"""Index helpers shared by the sequential models.

All models answer the same question for a row ``j`` at position ``t`` in its
student's sequence: *what is P(correct) given only rows at positions
``<= t - horizon``?* ``horizon=1`` is ordinary next-attempt prediction; larger
horizons hide the most recent ``horizon - 1`` attempts, which is what a tutor
planning several exercises ahead actually has.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Log


def positions(log: Log) -> np.ndarray:
    """Position of each row inside its student's sequence (0-based)."""
    bounds = log.user_bounds()
    return np.arange(len(log)) - np.repeat(bounds[:-1], np.diff(bounds))


@dataclass(frozen=True)
class SkillSeqs:
    """Rows grouped into (student, skill) sequences.

    ``seq[j]`` is the sequence id of row ``j``; ``idx[j]`` its index inside
    that sequence; ``order`` lists rows sequence by sequence, in time order.
    """

    seq: np.ndarray
    idx: np.ndarray
    order: np.ndarray
    starts: np.ndarray  # start of each sequence inside ``order``
    lengths: np.ndarray
    seq_skill: np.ndarray

    @property
    def n(self) -> int:
        return len(self.lengths)


def skill_seqs(log: Log) -> SkillSeqs:
    key = log.user.astype(np.int64) * (log.n_skills + 1) + log.skill
    order = np.lexsort((np.arange(len(log)), key))  # stable: time order within a key
    k_sorted = key[order]
    new = np.ones(len(order), dtype=bool)
    new[1:] = k_sorted[1:] != k_sorted[:-1]
    seq_sorted = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    lengths = np.diff(np.append(starts, len(order)))
    seq = np.empty(len(log), dtype=np.int64)
    seq[order] = seq_sorted
    idx = np.empty(len(log), dtype=np.int64)
    idx[order] = np.arange(len(order)) - starts[seq_sorted]
    return SkillSeqs(seq, idx, order, starts, lengths, log.skill[order[starts]])


def observed_in_seq(log: Log, ss: SkillSeqs, horizon: int) -> np.ndarray:
    """For each row, how many earlier rows of its (student, skill) sequence are
    visible when only rows at positions ``<= t - horizon`` may be used."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if horizon == 1:
        return ss.idx.copy()
    pos = positions(log)
    # sorted (seq, pos) keys of every row, in ``order``
    big = int(pos.max()) + 2 if len(pos) else 1
    keys = ss.seq[ss.order] * big + pos[ss.order]
    query = ss.seq * big + (pos - horizon)
    # a negative ``pos - horizon`` would land in the previous sequence's keys
    return np.maximum(np.searchsorted(keys, query, side="right") - ss.starts[ss.seq], 0)


@dataclass(frozen=True)
class TimeMajor:
    """Sequences laid out time-major for vectorised recursions.

    Sequences are sorted by descending length, so at step ``t`` the active
    sequences are a prefix of length ``count[t]``. ``flat[offset[t] + i]`` is the
    log row of sequence ``i`` (in that sorted order) at step ``t``.
    """

    flat: np.ndarray
    offset: np.ndarray
    count: np.ndarray
    seq_order: np.ndarray  # sequence ids in descending-length order

    def step(self, t: int) -> slice:
        return slice(self.offset[t], self.offset[t] + self.count[t])


def time_major(ss: SkillSeqs) -> TimeMajor:
    seq_order = np.argsort(-ss.lengths, kind="stable")
    max_len = int(ss.lengths.max()) if ss.n else 0
    asc = np.sort(ss.lengths)
    count = ss.n - np.searchsorted(asc, np.arange(max_len), side="right")
    offset = np.concatenate(([0], np.cumsum(count)[:-1])).astype(np.int64)
    flat = np.empty(int(count.sum()), dtype=np.int64)
    for t in range(max_len):
        active = seq_order[: count[t]]
        flat[offset[t] : offset[t] + count[t]] = ss.order[ss.starts[active] + t]
    return TimeMajor(flat, offset, count, seq_order)
