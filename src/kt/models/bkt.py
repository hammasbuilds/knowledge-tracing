"""Bayesian Knowledge Tracing, fitted per skill with EM (Baum-Welch).

A two-state hidden Markov model per skill: the student either knows the skill
or not. Four parameters per skill:

* ``prior``  P(L0)  - knows it before the first attempt
* ``learn``  P(T)   - moves from unknown to known after an attempt
* ``guess``  P(G)   - correct while not knowing
* ``slip``   P(S)   - wrong while knowing

No forgetting. BKT is not identifiable without constraints: "knows it and
slips a lot" and "does not know it and guesses a lot" can fit the same data
(Beck & Chang 2007). The standard fix, used here, bounds ``guess`` and
``slip`` so a known skill is always more likely to be answered correctly than
an unknown one. The M-step decouples per parameter and each parameter's
objective is a concave binomial likelihood, so clipping the closed-form update
to the bounds *is* the exact constrained M-step, not an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data import Log
from ..seq import SkillSeqs, TimeMajor, observed_in_seq, skill_seqs, time_major

PARAMS = ("prior", "learn", "guess", "slip")
DEFAULT_INIT = {"prior": 0.4, "learn": 0.1, "guess": 0.2, "slip": 0.1}
TINY = 1e-6


@dataclass
class BKTBounds:
    guess_max: float = 0.3
    slip_max: float = 0.3
    floor: float = 1e-4

    def clip(self, p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        f = self.floor
        return {
            "prior": np.clip(p["prior"], f, 1 - f),
            "learn": np.clip(p["learn"], f, 1 - f),
            "guess": np.clip(p["guess"], f, self.guess_max),
            "slip": np.clip(p["slip"], f, self.slip_max),
        }


UNBOUNDED = BKTBounds(guess_max=1 - 1e-4, slip_max=1 - 1e-4)


def _emit(
    correct: np.ndarray, guess: np.ndarray, slip: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """P(obs | known), P(obs | unknown)."""
    c = correct.astype(np.float64)
    return c * (1 - slip) + (1 - c) * slip, c * guess + (1 - c) * (1 - guess)


@dataclass
class BKT:
    """Per-skill BKT. Skills unseen in training fall back to pooled parameters."""

    bounds: BKTBounds = field(default_factory=BKTBounds)
    max_iter: int = 100
    tol: float = 1e-7
    name: str = "BKT"
    params: dict[str, np.ndarray] = field(default_factory=dict)
    pooled: dict[str, float] = field(default_factory=dict)
    fit_info: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- fitting
    def _em(self, log: Log, ss: SkillSeqs, tm: TimeMajor, n_skills: int, pooled: bool) -> dict:
        n_groups = 1 if pooled else n_skills
        group_of_seq = np.zeros(ss.n, dtype=np.int64) if pooled else ss.seq_skill.astype(np.int64)
        p = {k: np.full(n_groups, v) for k, v in DEFAULT_INIT.items()}
        p = self.bounds.clip(p)
        correct = log.correct
        sorted_seq = tm.seq_order
        max_len = len(tm.count)
        prev_ll = -np.inf
        ll = -np.inf
        it = 0
        for it in range(1, self.max_iter + 1):
            gs = group_of_seq[sorted_seq]  # group per active slot, prefix-aligned
            # forward (scaled): a = P(known | obs up to t)
            alpha = np.empty(len(tm.flat))
            prior_known = np.empty(len(tm.flat))  # P(known at t | obs before t)
            scale = np.empty(len(tm.flat))
            for t in range(max_len):
                sl = tm.step(t)
                n = tm.count[t]
                g = gs[:n]
                if t == 0:
                    pk = p["prior"][g]
                else:
                    prev = alpha[tm.offset[t - 1] : tm.offset[t - 1] + n]
                    pk = prev + (1 - prev) * p["learn"][g]
                ek, eu = _emit(correct[tm.flat[sl]], p["guess"][g], p["slip"][g])
                num = pk * ek
                z = num + (1 - pk) * eu
                prior_known[sl] = pk
                alpha[sl] = num / z
                scale[sl] = z
            ll = float(np.log(scale).sum())
            # backward: beta_k, beta_u (scaled by the same z)
            post_k = np.empty(len(tm.flat))
            xi_ul = np.zeros(len(tm.flat))  # expected unknown->known transitions into t
            bk = np.ones(tm.count[max_len - 1]) if max_len else np.ones(0)
            bu = np.ones_like(bk)
            for t in range(max_len - 1, -1, -1):
                sl = tm.step(t)
                n = tm.count[t]
                g = gs[:n]
                a = alpha[sl]
                if t < max_len - 1:
                    n_next = tm.count[t + 1]
                    nb_k = np.ones(n)
                    nb_u = np.ones(n)
                    sl1 = tm.step(t + 1)
                    g1 = gs[:n_next]
                    ek1, eu1 = _emit(correct[tm.flat[sl1]], p["guess"][g1], p["slip"][g1])
                    z1 = scale[sl1]
                    lr = p["learn"][g1]
                    wk = ek1 * bk / z1  # emission * beta at t+1, known
                    wu = eu1 * bu / z1
                    nb_k[:n_next] = wk
                    nb_u[:n_next] = lr * wk + (1 - lr) * wu
                    xi_ul[sl1] = (1 - a[:n_next]) * lr * wk
                    bk, bu = nb_k, nb_u
                else:
                    bk = np.ones(n)
                    bu = np.ones(n)
                post_k[sl] = a * bk / (a * bk + (1 - a) * bu)
            # M-step, accumulated per group
            row_group = np.empty(len(tm.flat), dtype=np.int64)
            for t in range(max_len):
                row_group[tm.step(t)] = gs[: tm.count[t]]
            c = correct[tm.flat].astype(np.float64)
            first = np.zeros(len(tm.flat), dtype=bool)
            first[tm.step(0)] = True
            not_last_unknown = np.zeros(len(tm.flat))
            for t in range(max_len - 1):
                sl = tm.step(t)
                n_next = tm.count[t + 1]
                not_last_unknown[tm.offset[t] : tm.offset[t] + n_next] = 1 - post_k[sl][:n_next]
            new = {
                "prior": _group_mean(post_k[first], row_group[first], n_groups, p["prior"]),
                "learn": _ratio(xi_ul, not_last_unknown, row_group, n_groups, p["learn"]),
                "guess": _ratio((1 - post_k) * c, 1 - post_k, row_group, n_groups, p["guess"]),
                "slip": _ratio(post_k * (1 - c), post_k, row_group, n_groups, p["slip"]),
            }
            p = self.bounds.clip(new)
            if ll - prev_ll < self.tol * max(1.0, abs(ll)) and it > 1:
                break
            prev_ll = ll
        return {"params": p, "loglik": ll, "iterations": it}

    def fit(self, train: Log, val: Log | None = None) -> BKT:
        ss = skill_seqs(train)
        tm = time_major(ss)
        per = self._em(train, ss, tm, train.n_skills, pooled=False)
        pool = self._em(train, ss, tm, train.n_skills, pooled=True)
        seen = np.bincount(train.skill, minlength=train.n_skills) > 0
        self.pooled = {k: float(v[0]) for k, v in pool["params"].items()}
        self.params = {k: np.where(seen, v, self.pooled[k]) for k, v in per["params"].items()}
        g, s = self.params["guess"][seen], self.params["slip"][seen]
        self.fit_info = {
            "loglik": per["loglik"],
            "iterations": per["iterations"],
            "skills_fitted": int(seen.sum()),
            "at_guess_bound": int(np.sum(np.isclose(g, self.bounds.guess_max))),
            "at_slip_bound": int(np.sum(np.isclose(s, self.bounds.slip_max))),
            "degenerate_g_plus_s_ge_1": int(np.sum(g + s >= 1)),
        }
        return self

    # ------------------------------------------------------------ prediction
    def _p(self, name: str, skill: np.ndarray) -> np.ndarray:
        arr = self.params[name]
        out = np.full(len(skill), self.pooled[name])
        ok = skill < len(arr)
        out[ok] = arr[skill[ok]]
        return out

    def knowledge(self, log: Log) -> tuple[np.ndarray, np.ndarray]:
        """Filtered P(known) before (``prior``) and after (``post``) each row."""
        ss = skill_seqs(log)
        tm = time_major(ss)
        prior_known = np.empty(len(log))
        post = np.empty(len(log))
        learn_r = self._p("learn", log.skill)
        guess_r = self._p("guess", log.skill)
        slip_r = self._p("slip", log.skill)
        prior_r = self._p("prior", log.skill)
        last = np.empty(0)
        for t in range(len(tm.count)):
            rows = tm.flat[tm.step(t)]
            pk = (
                prior_r[rows]
                if t == 0
                else last[: len(rows)] + (1 - last[: len(rows)]) * learn_r[rows]
            )
            ek, eu = _emit(log.correct[rows], guess_r[rows], slip_r[rows])
            a = pk * ek / (pk * ek + (1 - pk) * eu)
            prior_known[rows] = pk
            post[rows] = a
            last = a
        return prior_known, post

    def predict(self, log: Log, horizon: int = 1) -> np.ndarray:
        ss = skill_seqs(log)
        prior_known, _ = self.knowledge(log)
        q_obs = observed_in_seq(log, ss, horizon)
        gap = ss.idx - q_obs  # unobserved same-skill opportunities before this row
        src = ss.order[ss.starts[ss.seq] + q_obs]  # row whose prior is the last informed one
        learn = self._p("learn", log.skill)
        known = 1 - (1 - prior_known[src]) * (1 - learn) ** gap
        guess, slip = self._p("guess", log.skill), self._p("slip", log.skill)
        return known * (1 - slip) + (1 - known) * guess

    def to_json(self) -> dict:
        return {
            "model": "bkt",
            "bounds": {"guess_max": self.bounds.guess_max, "slip_max": self.bounds.slip_max},
            "pooled": self.pooled,
            "params": {k: v.tolist() for k, v in self.params.items()},
        }

    @classmethod
    def from_json(cls, d: dict) -> BKT:
        m = cls(bounds=BKTBounds(**d["bounds"]))
        m.pooled = {k: float(v) for k, v in d["pooled"].items()}
        m.params = {k: np.asarray(v, dtype=np.float64) for k, v in d["params"].items()}
        return m


def _group_mean(v: np.ndarray, g: np.ndarray, n: int, fallback: np.ndarray) -> np.ndarray:
    return _ratio(v, np.ones_like(v), g, n, fallback)


def _ratio(
    num: np.ndarray, den: np.ndarray, g: np.ndarray, n: int, fallback: np.ndarray
) -> np.ndarray:
    a = np.bincount(g, weights=num, minlength=n)
    b = np.bincount(g, weights=den, minlength=n)
    return np.where(b > TINY, a / np.maximum(b, TINY), fallback)
