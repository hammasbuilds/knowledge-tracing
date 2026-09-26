"""Performance Factors Analysis (Pavlik, Cen & Koedinger 2009).

``logit P(correct) = beta_s + gamma_s * successes_s + rho_s * failures_s``

where the counts are the student's earlier successes and failures on skill
``s``. Every row touches exactly three parameters of one skill, so the
likelihood separates by skill and each skill is a three-parameter logistic
regression, fitted here by Newton's method with a small L2 penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data import Log
from ..seq import observed_in_seq, skill_seqs


def counts(log: Log, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Earlier successes and failures on the row's skill, visible at ``horizon``."""
    ss = skill_seqs(log)
    q = observed_in_seq(log, ss, horizon)
    c_sorted = log.correct[ss.order].astype(np.int64)
    cum = np.concatenate(([0], np.cumsum(c_sorted)))
    start = ss.starts[ss.seq]
    succ = cum[start + q] - cum[start]
    return succ.astype(np.float64), (q - succ).astype(np.float64)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float, iters: int = 50) -> np.ndarray:
    """Newton-Raphson for L2-penalised logistic regression (intercept penalised too)."""
    w = np.zeros(x.shape[1])
    for _ in range(iters):
        p = _sigmoid(x @ w)
        grad = x.T @ (p - y) + l2 * w
        hess = (x * (p * (1 - p))[:, None]).T @ x + l2 * np.eye(x.shape[1])
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


@dataclass
class PFA:
    l2: float = 1.0
    name: str = "PFA"
    weights: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    pooled: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def fit(self, train: Log, val: Log | None = None) -> PFA:
        succ, fail = counts(train)
        x = np.column_stack([np.ones(len(train)), succ, fail])
        y = train.correct.astype(np.float64)
        self.pooled = fit_logistic(x, y, self.l2)
        self.weights = np.tile(self.pooled, (train.n_skills, 1))
        order = np.argsort(train.skill, kind="stable")
        cuts = np.flatnonzero(np.diff(train.skill[order])) + 1
        for rows in np.split(order, cuts):
            if len(rows):
                self.weights[train.skill[rows[0]]] = fit_logistic(x[rows], y[rows], self.l2)
        return self

    def predict(self, log: Log, horizon: int = 1) -> np.ndarray:
        succ, fail = counts(log, horizon)
        w = np.tile(self.pooled, (len(log), 1))
        ok = log.skill < len(self.weights)
        w[ok] = self.weights[log.skill[ok]]
        return _sigmoid(w[:, 0] + w[:, 1] * succ + w[:, 2] * fail)
