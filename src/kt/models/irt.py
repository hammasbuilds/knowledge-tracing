"""Item response theory with student ability tracked online.

``P(correct) = sigmoid(a_s * theta_u - b_i)``,  ``b_i = beta_s + delta_i``

Item difficulty is a skill difficulty plus an item offset shrunk towards zero,
so an item seen once in training does not get an extreme difficulty of its own
(and an unseen item falls back to its skill). ``a_s`` is fixed at 1 for 1PL and
learned per skill for 2PL.

Training fits student abilities jointly (MAP, ``theta ~ N(0, 1)``). At
prediction time a held-out student has no fitted ability, so ``theta`` is
re-estimated from that student's own earlier answers as a posterior mean over
a grid (EAP) - the only honest way to use IRT on a student it has not seen.
``ability="fitted"`` instead reuses the training ability of any student seen
in training; that is exactly the leak a row-level split permits, and it is
kept so the leak can be measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data import Log
from ..seq import positions

GRID = np.linspace(-4.0, 4.0, 61)
LOG_PRIOR = -0.5 * GRID**2 - np.log(np.sum(np.exp(-0.5 * GRID**2)))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class IRT:
    two_pl: bool = False
    item_sd: float = 1.0
    steps: int = 600
    lr: float = 0.05
    ability: str = "online"  # or "fitted"
    name: str = "IRT-1PL"
    theta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    theta_seen: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    beta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    delta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    log_a: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        if self.ability not in ("online", "fitted"):
            raise ValueError("ability must be 'online' or 'fitted'")

    def fit(self, train: Log, val: Log | None = None) -> IRT:
        u, i, s = train.user, train.item, train.skill
        y = train.correct.astype(np.float64)
        n = len(train)
        params = {
            "theta": np.zeros(train.n_users),
            "beta": np.zeros(train.n_skills),
            "delta": np.zeros(train.n_items),
            "log_a": np.zeros(train.n_skills),
        }
        # prior precisions, per parameter group (MAP = penalised likelihood)
        prec = {"theta": 1.0, "beta": 1e-4, "delta": 1.0 / self.item_sd**2, "log_a": 4.0}
        trainable = ["theta", "beta", "delta"] + (["log_a"] if self.two_pl else [])
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v2 = {k: np.zeros_like(v) for k, v in params.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        for step in range(1, self.steps + 1):
            a = np.exp(params["log_a"][s])
            z = a * params["theta"][u] - params["beta"][s] - params["delta"][i]
            r = _sigmoid(z) - y  # dNLL/dz
            grads = {
                "theta": np.bincount(u, r * a, train.n_users),
                "beta": -np.bincount(s, r, train.n_skills),
                "delta": -np.bincount(i, r, train.n_items),
                "log_a": np.bincount(s, r * a * params["theta"][u], train.n_skills),
            }
            for k in trainable:
                g = (grads[k] + prec[k] * params[k]) / n
                m[k] = b1 * m[k] + (1 - b1) * g
                v2[k] = b2 * v2[k] + (1 - b2) * g * g
                mh = m[k] / (1 - b1**step)
                vh = v2[k] / (1 - b2**step)
                params[k] -= self.lr * mh / (np.sqrt(vh) + eps)
        self.theta = params["theta"]
        self.theta_seen = np.bincount(u, minlength=train.n_users) > 0
        seen_s = np.bincount(s, minlength=train.n_skills) > 0
        mean_beta = float(params["beta"][seen_s].mean()) if seen_s.any() else 0.0
        self.beta = np.where(seen_s, params["beta"], mean_beta)
        self.delta = params["delta"]
        self.log_a = params["log_a"]
        return self

    def _ab(self, log: Log) -> tuple[np.ndarray, np.ndarray]:
        s, i = log.skill, log.item
        beta = np.full(len(log), float(self.beta.mean()) if len(self.beta) else 0.0)
        ok = s < len(self.beta)
        beta[ok] = self.beta[s[ok]]
        delta = np.zeros(len(log))
        ok_i = i < len(self.delta)
        delta[ok_i] = self.delta[i[ok_i]]
        a = np.ones(len(log))
        a[ok] = np.exp(self.log_a[s[ok]])
        return a, beta + delta

    def predict(self, log: Log, horizon: int = 1) -> np.ndarray:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        a, b = self._ab(log)
        out = np.empty(len(log))
        bounds = log.user_bounds()
        pos = positions(log)
        for u in range(log.n_users):
            lo, hi = bounds[u], bounds[u + 1]
            if lo == hi:
                continue
            if self.ability == "fitted" and u < len(self.theta_seen) and self.theta_seen[u]:
                out[lo:hi] = _sigmoid(a[lo:hi] * self.theta[u] - b[lo:hi])
                continue
            pg = _sigmoid(a[lo:hi, None] * GRID[None, :] - b[lo:hi, None])  # (T, G)
            y = log.correct[lo:hi, None]
            ll = np.where(y == 1, np.log(pg + 1e-12), np.log(1 - pg + 1e-12))
            cum = np.vstack([np.zeros((1, len(GRID))), np.cumsum(ll, axis=0)])  # cum[t] = rows < t
            t_hist = np.maximum(pos[lo:hi] - horizon + 1, 0)  # rows < t-h+1, i.e. <= t-h
            logpost = LOG_PRIOR[None, :] + cum[t_hist]
            logpost -= logpost.max(axis=1, keepdims=True)
            w = np.exp(logpost)
            w /= w.sum(axis=1, keepdims=True)
            out[lo:hi] = np.sum(w * pg, axis=1)
        return out

    def predict_skills(
        self, skill: np.ndarray, correct: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """P(correct) on a new item of each target skill, given one student's history.

        The ability posterior is formed from the history only (EAP over the
        grid); new items carry their skill's difficulty (item offset 0).
        """
        ok = skill < len(self.beta)
        a_h = np.where(ok, np.exp(self.log_a[np.minimum(skill, len(self.beta) - 1)]), 1.0)
        b_h = np.where(ok, self.beta[np.minimum(skill, len(self.beta) - 1)], self.beta.mean())
        logpost = LOG_PRIOR.copy()
        if len(skill):
            pg = _sigmoid(a_h[:, None] * GRID[None, :] - b_h[:, None])
            y = correct[:, None]
            logpost += np.where(y == 1, np.log(pg + 1e-12), np.log(1 - pg + 1e-12)).sum(axis=0)
        w = np.exp(logpost - logpost.max())
        w /= w.sum()
        a_t = np.exp(self.log_a[targets])
        return (_sigmoid(a_t[:, None] * GRID[None, :] - self.beta[targets][:, None]) * w).sum(
            axis=1
        )

    def skill_level(self) -> IRT:
        """This model with its item offsets dropped: every item gets its skill's
        difficulty. It is what :meth:`to_json` saves and what the tutor runs
        (a history names skills, not items), so the study scores it separately
        from the item-level model."""
        m = IRT(two_pl=self.two_pl, item_sd=self.item_sd, name=f"{self.name} (skill-level)")
        m.beta, m.log_a = self.beta.copy(), self.log_a.copy()
        return m

    def to_json(self) -> dict:
        """The skill-level model only (see :meth:`skill_level`): item offsets and
        training abilities are left out. It is a weaker model than the item-level
        IRT the study headlines, and is scored on its own as ``irt_skill_level``."""
        return {
            "model": "irt",
            "two_pl": self.two_pl,
            "item_sd": self.item_sd,
            "beta": self.beta.tolist(),
            "log_a": self.log_a.tolist(),
        }

    @classmethod
    def from_json(cls, d: dict) -> IRT:
        m = cls(two_pl=bool(d["two_pl"]), item_sd=float(d["item_sd"]))
        m.name = "IRT-2PL" if m.two_pl else "IRT-1PL"
        m.beta = np.asarray(d["beta"], dtype=np.float64)
        m.log_a = np.asarray(d["log_a"], dtype=np.float64)
        return m
