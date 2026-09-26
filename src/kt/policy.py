"""Choosing the next exercise, and judging that choice without a simulator.

Two policies a tutor can run on top of any model that outputs P(correct):

* **target difficulty** - serve the skill whose predicted P(correct) is closest
  to a target (0.7 by default: hard enough to learn from, easy enough not to
  discourage);
* **mastery** - keep practising a skill until the predicted P(correct) on it
  reaches a threshold, then stop.

Neither can be evaluated by simply replaying a log, because the log's
exercise order was chosen by someone else. What *can* be measured on logged
data, with no simulated student, is whether the model's promise holds:

* :func:`band_check` - among logged attempts the model predicted at ~0.7, how
  often was the student actually right? If a model says 0.7 and students
  score 0.85, a 0.7-targeting tutor built on it serves exercises that are too
  easy.
* :func:`mastery_tradeoff` - the effort/score trade-off of González-Brenes &
  Huang's Leopard evaluation: for each threshold, how many practice attempts
  happen before the model declares mastery (effort), and how often the student
  is right on the attempts *after* the declaration (score).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Log
from .seq import skill_seqs


@dataclass(frozen=True)
class Recommendation:
    skill: str
    p_correct: float
    reason: str


def recommend(
    p_by_skill: dict[str, float],
    target: float = 0.7,
    mastered: set[str] | None = None,
) -> Recommendation:
    """Pick the unmastered skill whose predicted P(correct) is closest to ``target``."""
    if not 0 < target < 1:
        raise ValueError("target must be strictly between 0 and 1")
    pool = {s: p for s, p in p_by_skill.items() if s not in (mastered or set())}
    if not pool:
        raise ValueError("no candidate skills left (all mastered or none given)")
    skill = min(sorted(pool), key=lambda s: abs(pool[s] - target))
    p = pool[skill]
    return Recommendation(
        skill, p, f"predicted P(correct) {p:.2f} is closest to target {target:.2f}"
    )


def band_check(y: np.ndarray, p: np.ndarray, target: float = 0.7, width: float = 0.05) -> dict:
    """Observed correct-rate among attempts predicted within ``target +- width``."""
    m = np.abs(p - target) <= width
    n = int(m.sum())
    if n == 0:
        return {"target": target, "width": width, "n": 0, "share": 0.0}
    rate = float(y[m].mean())
    se = float(np.sqrt(rate * (1 - rate) / n))
    return {
        "target": target,
        "width": width,
        "n": n,
        "share": n / len(y),
        "mean_predicted": float(p[m].mean()),
        "observed_rate": rate,
        "gap": rate - float(p[m].mean()),
        "naive_se": se,
    }


def band_check_ci(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    target: float = 0.7,
    width: float = 0.05,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """:func:`band_check` with a student-level bootstrap CI on the observed rate."""
    base = band_check(y, p, target, width)
    if base["n"] == 0:
        return base
    m = np.abs(p - target) <= width
    ys, gs = y[m].astype(np.float64), groups[m]
    uniq, inv = np.unique(gs, return_inverse=True)
    sums = np.bincount(inv, weights=ys)
    cnts = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        rates.append(float((w * sums).sum() / max((w * cnts).sum(), 1)))
    base["observed_rate_ci"] = [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))]
    base["students"] = int(len(uniq))
    return base


def mastery_tradeoff(
    log: Log, p: np.ndarray, thresholds: tuple[float, ...] = (0.8, 0.85, 0.9, 0.95)
) -> dict:
    """Leopard-style effort/score per threshold, over (student, skill) sequences.

    ``p[j]`` must be the model's prediction for row ``j`` made before seeing it.
    Mastery is declared at the first row whose prediction reaches the
    threshold. Effort is the number of attempts before that row (the full
    sequence length when mastery is never declared). Score is the correct rate
    over the declared row and everything after it on that skill.
    Also reports the same numbers for the log's own stopping rule proxy,
    three correct in a row (ASSISTments skill builders end there).
    """
    ss = skill_seqs(log)
    y_sorted = log.correct[ss.order].astype(np.float64)
    p_sorted = p[ss.order]
    starts, lengths = ss.starts, ss.lengths
    out: dict = {"sequences": int(ss.n), "thresholds": {}}

    never = np.iinfo(np.int64).max
    cum = np.concatenate(([0.0], np.cumsum(y_sorted)))
    ends = starts + lengths

    def summarise(first_idx: np.ndarray) -> dict:
        declared = first_idx != never
        effort = np.minimum(first_idx, lengths)
        s_at = starts + effort
        after_c = cum[ends] - cum[s_at]
        n_after = float((ends - s_at)[declared].sum())
        return {
            "declared_share": float(declared.mean()),
            "mean_effort": float(effort.mean()),
            "post_declaration_attempts": int(n_after),
            "post_declaration_correct_rate": float(after_c[declared].sum() / n_after)
            if n_after
            else None,
        }

    seq_of_sorted = np.repeat(np.arange(ss.n), lengths)
    idx_sorted = np.arange(len(p_sorted)) - starts[seq_of_sorted]
    for th in thresholds:
        hit = p_sorted >= th
        first = np.full(ss.n, never)
        np.minimum.at(first, seq_of_sorted[hit], idx_sorted[hit])
        out["thresholds"][str(th)] = summarise(first)
    # three-correct-in-a-row: declared *after* the third consecutive correct
    streak = np.zeros(len(y_sorted), dtype=np.int64)
    for j in range(len(y_sorted)):
        prev = streak[j - 1] if j and idx_sorted[j] > 0 else 0
        streak[j] = prev + 1 if y_sorted[j] == 1 else 0
    done = streak >= 3
    first = np.full(ss.n, never)
    np.minimum.at(first, seq_of_sorted[done], idx_sorted[done] + 1)
    out["three_in_a_row"] = summarise(first)
    return out
