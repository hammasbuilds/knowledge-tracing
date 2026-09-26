"""Deep Knowledge Tracing (Piech et al. 2015): an LSTM over (skill, correct) tokens.

Written directly against numpy, backward pass included, so it runs on CPU with
no deep-learning framework. The input at each step is the previous attempt as
one of ``2 * n_skills`` tokens (plus a start token); the output is one logit
per skill, of which only the logit of the skill actually attempted next is
scored. Training uses truncated back-propagation through time over chunks of
``bptt`` steps with the hidden state carried across chunks, Adam, gradient
clipping and early stopping on validation AUC.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..data import Log
from ..metrics import auc
from ..seq import positions


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class Batch:
    tokens: np.ndarray  # (B, T) input token per step
    skill: np.ndarray  # (B, T) skill whose correctness is predicted at this step
    y: np.ndarray  # (B, T) that correctness
    valid: np.ndarray  # (B, T) bool
    rows: np.ndarray  # (B, T) log row predicted at this step, -1 for padding


def make_batches(log: Log, batch_size: int, rng: np.random.Generator | None = None) -> list[Batch]:
    """Group students of similar length into padded batches.

    Step ``t`` of a student reads token ``t`` (start token, then attempts
    ``0..t-1``) and predicts attempt ``t``.
    """
    start = 2 * log.n_skills
    bounds = log.user_bounds()
    lengths = np.diff(bounds)
    users = np.flatnonzero(lengths > 0)
    users = users[np.argsort(lengths[users], kind="stable")]
    groups = [users[i : i + batch_size] for i in range(0, len(users), batch_size)]
    if rng is not None:
        rng.shuffle(groups)
    out = []
    for g in groups:
        t_max = int(lengths[g].max())
        b = len(g)
        tokens = np.full((b, t_max), start, dtype=np.int64)
        skill = np.zeros((b, t_max), dtype=np.int64)
        y = np.zeros((b, t_max), dtype=np.float64)
        valid = np.zeros((b, t_max), dtype=bool)
        rows = np.full((b, t_max), -1, dtype=np.int64)
        for r, u in enumerate(g):
            lo, hi = bounds[u], bounds[u + 1]
            n = hi - lo
            tok = 2 * log.skill[lo:hi].astype(np.int64) + log.correct[lo:hi]
            tokens[r, 1:n] = tok[:-1]
            skill[r, :n] = log.skill[lo:hi]
            y[r, :n] = log.correct[lo:hi]
            valid[r, :n] = True
            rows[r, :n] = np.arange(lo, hi)
        out.append(Batch(tokens, skill, y, valid, rows))
    return out


@dataclass
class LSTMParams:
    wx: np.ndarray  # (V, 4H)
    wh: np.ndarray  # (H, 4H)
    b: np.ndarray  # (4H,)
    wo: np.ndarray  # (H, S)
    bo: np.ndarray  # (S,)

    NAMES = ("wx", "wh", "b", "wo", "bo")

    @classmethod
    def init(
        cls, n_tokens: int, hidden: int, n_out: int, rng: np.random.Generator, dtype
    ) -> LSTMParams:
        s = 1.0 / np.sqrt(hidden)
        b = np.zeros(4 * hidden, dtype=dtype)
        b[hidden : 2 * hidden] = 1.0  # forget-gate bias
        return cls(
            wx=rng.uniform(-s, s, (n_tokens, 4 * hidden)).astype(dtype),
            wh=rng.uniform(-s, s, (hidden, 4 * hidden)).astype(dtype),
            b=b,
            wo=rng.uniform(-s, s, (hidden, n_out)).astype(dtype),
            bo=np.zeros(n_out, dtype=dtype),
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in self.NAMES}


def lstm_chunk(
    p: LSTMParams,
    tokens: np.ndarray,
    skill: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    h0: np.ndarray,
    c0: np.ndarray,
    norm: float,
    grad: bool = True,
) -> tuple[float, dict[str, np.ndarray] | None, np.ndarray, np.ndarray, np.ndarray]:
    """Forward (and optionally backward) over one chunk of steps.

    Returns ``(loss_sum, grads, h_T, c_T, hidden)`` where ``hidden`` is (B, T, H):
    the state each step's prediction was read from. ``grads`` are of
    ``loss_sum / norm``.
    """
    bsz, steps = tokens.shape
    hdim = p.wh.shape[0]
    dtype = p.wh.dtype
    hs = np.empty((bsz, steps, hdim), dtype=dtype)
    cache = []
    h, c = h0, c0
    logits = np.empty((bsz, steps), dtype=dtype)
    for t in range(steps):
        z = p.wx[tokens[:, t]] + h @ p.wh + p.b
        i = _sigmoid(z[:, :hdim])
        f = _sigmoid(z[:, hdim : 2 * hdim])
        g = np.tanh(z[:, 2 * hdim : 3 * hdim])
        o = _sigmoid(z[:, 3 * hdim :])
        c_new = f * c + i * g
        tc = np.tanh(c_new)
        h_new = o * tc
        if grad:
            cache.append((h, c, i, f, g, o, tc))
        h, c = h_new, c_new
        hs[:, t] = h
        logits[:, t] = np.einsum("bh,hb->b", h, p.wo[:, skill[:, t]]) + p.bo[skill[:, t]]
    pr = _sigmoid(logits)
    ll = np.where(y == 1, np.log(pr + 1e-12), np.log(1 - pr + 1e-12))
    loss = float(-(ll * valid).sum())
    if not grad:
        return loss, None, h, c, hs
    g_all = {k: np.zeros_like(v) for k, v in p.arrays().items()}
    dlogit = ((pr - y) * valid / norm).astype(dtype)  # (B, T)
    # output layer, all steps at once
    flat_s = skill.reshape(-1)
    flat_h = hs.reshape(-1, hdim)
    flat_d = dlogit.reshape(-1)
    np.add.at(g_all["wo"].T, flat_s, flat_h * flat_d[:, None])
    g_all["bo"] += np.bincount(flat_s, weights=flat_d, minlength=len(p.bo)).astype(dtype)
    dz_all = np.empty((bsz, steps, 4 * hdim), dtype=dtype)
    dh_next = np.zeros((bsz, hdim), dtype=dtype)
    dc_next = np.zeros((bsz, hdim), dtype=dtype)
    for t in range(steps - 1, -1, -1):
        h_prev, c_prev, i, f, g, o, tc = cache[t]
        dh = dlogit[:, t, None] * p.wo[:, skill[:, t]].T + dh_next
        do = dh * tc
        dc = dh * o * (1 - tc * tc) + dc_next
        dz = dz_all[:, t]
        dz[:, :hdim] = dc * g * i * (1 - i)
        dz[:, hdim : 2 * hdim] = dc * c_prev * f * (1 - f)
        dz[:, 2 * hdim : 3 * hdim] = dc * i * (1 - g * g)
        dz[:, 3 * hdim :] = do * o * (1 - o)
        g_all["wh"] += h_prev.T @ dz
        dh_next = dz @ p.wh.T
        dc_next = dc * f
    flat_dz = dz_all.reshape(-1, 4 * hdim)
    g_all["b"] += flat_dz.sum(axis=0)
    np.add.at(g_all["wx"], tokens.reshape(-1), flat_dz)
    return loss, g_all, h, c, hs


@dataclass
class DKT:
    hidden: int = 64
    lr: float = 3e-3
    batch_size: int = 32
    bptt: int = 100
    max_epochs: int = 20
    patience: int = 3
    clip: float = 5.0
    l2: float = 1e-6
    seed: int = 0
    dtype: type = np.float32
    name: str = "DKT"
    verbose: bool = False
    params: LSTMParams | None = None
    n_skills: int = 0
    history: list[dict] = field(default_factory=list)

    def fit(self, train: Log, val: Log | None = None) -> DKT:
        rng = np.random.default_rng(self.seed)
        self.n_skills = train.n_skills
        self.params = LSTMParams.init(
            2 * train.n_skills + 1, self.hidden, train.n_skills, rng, self.dtype
        )
        adam_m = {k: np.zeros_like(v) for k, v in self.params.arrays().items()}
        adam_v = {k: np.zeros_like(v) for k, v in self.params.arrays().items()}
        step = 0
        best = (-np.inf, None, 0)
        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            total, count = 0.0, 0
            for batch in make_batches(train, self.batch_size, rng):
                h = np.zeros((len(batch.tokens), self.hidden), dtype=self.dtype)
                c = np.zeros_like(h)
                for s in range(0, batch.tokens.shape[1], self.bptt):
                    sl = slice(s, s + self.bptt)
                    n_valid = float(batch.valid[:, sl].sum())
                    if n_valid == 0:
                        break
                    loss, grads, h, c, _ = lstm_chunk(
                        self.params,
                        batch.tokens[:, sl],
                        batch.skill[:, sl],
                        batch.y[:, sl],
                        batch.valid[:, sl],
                        h,
                        c,
                        n_valid,
                    )
                    total += loss
                    count += int(n_valid)
                    step += 1
                    self._adam(grads, adam_m, adam_v, step)
            record = {
                "epoch": epoch,
                "train_nll": total / max(count, 1),
                "seconds": time.time() - t0,
            }
            if val is not None and len(val):
                v_auc = auc(val.correct, self.predict(val))
                record["val_auc"] = v_auc
                if v_auc > best[0]:
                    best = (v_auc, _copy(self.params), epoch)
                elif epoch - best[2] >= self.patience:
                    self.history.append(record)
                    break
            self.history.append(record)
            if self.verbose:
                print(f"  DKT epoch {epoch}: {record}", flush=True)
        if best[1] is not None:
            self.params = best[1]
        return self

    def _adam(self, grads: dict, m: dict, v: dict, step: int) -> None:
        assert self.params is not None
        arrays = self.params.arrays()
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        scale = min(1.0, self.clip / (norm + 1e-12))
        b1, b2 = 0.9, 0.999
        for k, g in grads.items():
            g = g * scale + self.l2 * arrays[k]
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * g * g
            mh = m[k] / (1 - b1**step)
            vh = v[k] / (1 - b2**step)
            arrays[k] -= (self.lr * mh / (np.sqrt(vh) + 1e-8)).astype(arrays[k].dtype)

    def hidden_states(self, log: Log) -> np.ndarray:
        """(n_rows, H): the state used to predict each row at horizon 1."""
        if self.params is None:
            raise RuntimeError("DKT is not fitted")
        if log.n_skills != self.n_skills:
            raise ValueError(f"log has {log.n_skills} skills, model was trained on {self.n_skills}")
        out = np.zeros((len(log), self.hidden), dtype=self.dtype)
        for batch in make_batches(log, 256):
            h = np.zeros((len(batch.tokens), self.hidden), dtype=self.dtype)
            c = np.zeros_like(h)
            for s in range(0, batch.tokens.shape[1], 512):
                sl = slice(s, s + 512)
                _, _, h, c, hs = lstm_chunk(
                    self.params,
                    batch.tokens[:, sl],
                    batch.skill[:, sl],
                    batch.y[:, sl],
                    batch.valid[:, sl],
                    h,
                    c,
                    1.0,
                    grad=False,
                )
                rows = batch.rows[:, sl]
                ok = rows >= 0
                out[rows[ok]] = hs[ok]
        return out

    def predict(self, log: Log, horizon: int = 1) -> np.ndarray:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        assert self.params is not None
        hs = self.hidden_states(log)
        pos = positions(log)
        # Row t at horizon k reads the state that predicted row t-k+1, i.e. the
        # state after attempts 0..t-k. For t < k that is the start state, which
        # is exactly the state that predicted the student's first row.
        src = np.arange(len(log)) - np.minimum(pos, horizon - 1)
        state = hs[src].astype(np.float64)
        s = log.skill
        z = (
            np.einsum("nh,hn->n", state, self.params.wo[:, s].astype(np.float64))
            + self.params.bo[s]
        )
        return _sigmoid(z)

    def save(self, path: str) -> None:
        assert self.params is not None
        np.savez_compressed(path, n_skills=self.n_skills, **self.params.arrays())

    @classmethod
    def load(cls, path: str) -> DKT:
        d = np.load(path)
        m = cls(hidden=d["wh"].shape[0], dtype=d["wh"].dtype.type)
        m.n_skills = int(d["n_skills"])
        m.params = LSTMParams(**{k: d[k] for k in LSTMParams.NAMES})
        return m


def _copy(p: LSTMParams) -> LSTMParams:
    return LSTMParams(**{k: v.copy() for k, v in p.arrays().items()})
