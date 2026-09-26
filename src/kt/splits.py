"""Train / validation / test splits.

The honest split is by student: a test student is never seen in training, so
every prediction about them has to come from their own history. The leaky
split assigns individual rows at random, which lets any model that stores
per-student parameters fit a student on answers they give *after* the one
being predicted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Log

SPLIT_NAMES = ("train", "val", "test")


def fractions_for(n_units: int) -> tuple[float, float, float]:
    """House split policy: 80/15/5 below 10k units, 90/7/3 at 10k and above."""
    return (0.8, 0.15, 0.05) if n_units < 10_000 else (0.9, 0.07, 0.03)


@dataclass(frozen=True)
class Split:
    """Per-row assignment to train (0), val (1) or test (2)."""

    kind: str  # "student" or "row"
    assign: np.ndarray  # int8 per row
    fractions: tuple[float, float, float]
    seed: int

    def mask(self, name: str) -> np.ndarray:
        return self.assign == SPLIT_NAMES.index(name)

    def summary(self, log: Log) -> dict[str, dict[str, int]]:
        out = {}
        for name in SPLIT_NAMES:
            m = self.mask(name)
            out[name] = {"rows": int(m.sum()), "students": int(len(np.unique(log.user[m])))}
        return out


def _cut(n: int, fractions: tuple[float, float, float], rng: np.random.Generator) -> np.ndarray:
    if not np.isclose(sum(fractions), 1.0) or min(fractions) <= 0:
        raise ValueError(f"fractions must be positive and sum to 1, got {fractions}")
    if n < 3:
        raise ValueError(f"need at least 3 units to split three ways, got {n}")
    order = rng.permutation(n)
    n_test = max(1, round(n * fractions[2]))
    n_val = max(1, round(n * fractions[1]))
    assign = np.zeros(n, dtype=np.int8)
    assign[order[:n_test]] = 2
    assign[order[n_test : n_test + n_val]] = 1
    return assign


def split_by_student(
    log: Log, seed: int = 0, fractions: tuple[float, float, float] | None = None
) -> Split:
    fr = fractions or fractions_for(log.n_users)
    per_user = _cut(log.n_users, fr, np.random.default_rng(seed))
    return Split("student", per_user[log.user], fr, seed)


def split_by_row(
    log: Log, seed: int = 0, fractions: tuple[float, float, float] | None = None
) -> Split:
    """Leaky baseline: rows are assigned at random, ignoring who answered them.

    Rows of one attempt (expanded variant) stay together so the leak measured
    is the student leak, not a copy of the same answer.
    """
    fr = fractions or fractions_for(log.n_users)
    starts = np.flatnonzero(log.first_row_of_attempt())
    per_attempt = _cut(len(starts), fr, np.random.default_rng(seed))
    lengths = np.diff(np.append(starts, len(log)))
    return Split("row", np.repeat(per_attempt, lengths), fr, seed)
