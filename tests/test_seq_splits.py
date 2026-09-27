import numpy as np
import pytest
from helpers import tiny_log

from kt.seq import observed_in_seq, positions, skill_seqs, time_major
from kt.splits import fractions_for, split_by_row, split_by_student
from kt.synthetic import simulate_bkt


def test_positions_and_skill_sequence_index():
    log = tiny_log([0, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 0], users=[0, 0, 0, 1, 1, 1])
    assert positions(log).tolist() == [0, 1, 2, 0, 1, 2]
    ss = skill_seqs(log)
    assert ss.idx.tolist() == [0, 0, 1, 0, 0, 1]
    assert ss.n == 4


def test_observed_in_seq_matches_brute_force():
    rng = np.random.default_rng(0)
    users = np.sort(rng.integers(0, 5, 200)).tolist()
    log = tiny_log(rng.integers(0, 4, 200).tolist(), rng.integers(0, 2, 200).tolist(), users)
    ss = skill_seqs(log)
    pos = positions(log)
    for h in (1, 2, 5):
        got = observed_in_seq(log, ss, h)
        for j in range(len(log)):
            same = [
                r
                for r in range(len(log))
                if log.user[r] == log.user[j]
                and log.skill[r] == log.skill[j]
                and pos[r] <= pos[j] - h
            ]
            assert got[j] == len(same)
    with pytest.raises(ValueError):
        observed_in_seq(log, ss, 0)


def test_time_major_covers_every_row_once():
    log = simulate_bkt(
        {"prior": [0.3] * 3, "learn": [0.1] * 3, "guess": [0.2] * 3, "slip": [0.1] * 3},
        20,
        4,
        seed=0,
    )
    tm = time_major(skill_seqs(log))
    assert sorted(tm.flat.tolist()) == list(range(len(log)))
    assert np.all(np.diff(tm.count) <= 0)


def test_student_split_is_disjoint_and_follows_policy():
    log = simulate_bkt({"prior": [0.3], "learn": [0.1], "guess": [0.2], "slip": [0.1]}, 400, 3)
    sp = split_by_student(log, seed=3)
    users = {n: set(np.unique(log.user[sp.mask(n)])) for n in ("train", "val", "test")}
    assert not users["train"] & users["test"] and not users["val"] & users["test"]
    assert len(users["test"]) == 20 and len(users["val"]) == 60
    assert fractions_for(9_999) == (0.8, 0.15, 0.05)
    assert fractions_for(10_000) == (0.9, 0.07, 0.03)


def test_row_split_mixes_students_but_keeps_attempts_whole():
    log = tiny_log([0, 1, 0, 1] * 50, [1, 0] * 100, users=[0] * 100 + [1] * 100)
    log.attempt[:] = np.repeat(np.arange(100), 2)  # pairs of rows share an attempt
    sp = split_by_row(log, seed=0)
    assert np.all(sp.assign[0::2] == sp.assign[1::2])
    assert len(np.unique(log.user[sp.mask("test")])) == 2
    with pytest.raises(ValueError):
        split_by_student(tiny_log([0, 0], [1, 0], [0, 1]))
