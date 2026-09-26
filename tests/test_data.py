import zipfile

import numpy as np
import pytest

from kt.data import COLLAPSED, EXPANDED, RAW, DataError, load_algebra, load_assistments
from helpers import assist_csv

ROWS = [
    (1, "u1", "p10", 1, "5", "A"),
    (1, "u1", "p10", 1, "7", "B"),  # multi-skill attempt
    (1, "u1", "p10", 1, "5", "A"),  # exact duplicate record
    (2, "u1", "p11", 0, "", ""),  # no skill at all
    (3, "u1", "p12", 0, "5", "A"),
    (4, "u2", "p10", 1, "7", "B"),
    (4, "u2", "p10", 0, "5", "A"),  # same order_id, different outcome: conflicting
    (5, "u2", "p13", 1, "7", "B"),
    (6, "u2", "p13", 2, "7", "B"),  # non-binary correct
    (0, "u2", "p14", 0, "5", "A"),  # earlier order_id listed later in the file
]


@pytest.fixture()
def assist_path(tmp_path):
    p = tmp_path / "skill_builder.csv"
    p.write_text(assist_csv(ROWS), encoding="latin-1")
    return p


def test_cleaning_counts(assist_path):
    log = load_assistments(assist_path, COLLAPSED)
    c = log.cleaning
    assert c["raw_rows"] == 10
    assert c["missing_skill_rows"] == 1
    assert c["exact_duplicate_rows"] == 1
    assert c["non_binary_rows"] == 1
    assert c["conflicting_attempts"] == 1
    assert c["attempts_without_any_skill"] == 1
    assert c["attempts"] == 4
    assert c["multi_skill_attempts"] == 1


def test_three_variants_row_counts(assist_path):
    assert len(load_assistments(assist_path, COLLAPSED)) == 4
    assert len(load_assistments(assist_path, EXPANDED)) == 5
    raw = load_assistments(assist_path, RAW)
    assert len(raw) == 6
    # the raw copies of attempt 1 are back to back and share one outcome
    first = raw.first_row_of_attempt()
    assert first.sum() == 4
    assert np.all(raw.correct[~first] == raw.correct[np.flatnonzero(~first) - 1])


def test_rows_sorted_by_user_then_time_and_collapsed_skill_names(assist_path):
    log = load_assistments(assist_path, COLLAPSED)
    names = [log.user_names[u] for u in log.user]
    assert names == ["u1", "u1", "u2", "u2"]
    assert log.attempt.tolist() == [1, 3, 0, 5]
    assert log.skill_names[log.skill[0]] == "5:A+7:B"
    assert log.correct.tolist() == [1, 0, 0, 1]


def test_reads_through_zip(assist_path, tmp_path):
    z = tmp_path / "data.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(assist_path, "inner/skill_builder.csv")
    assert len(load_assistments(z, COLLAPSED)) == 4


def test_missing_file_and_wrong_columns(tmp_path):
    with pytest.raises(DataError, match="does not exist"):
        load_assistments(tmp_path / "nope.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(DataError, match="columns"):
        load_assistments(bad)
    with pytest.raises(DataError, match="unknown variant"):
        load_assistments(_ok(tmp_path), "weird")


def _ok(tmp_path):
    p = tmp_path / "ok.csv"
    p.write_text(assist_csv(ROWS[:1]), encoding="latin-1")
    return p


ALGEBRA_HEADER = (
    "Row\tAnon Student Id\tProblem Hierarchy\tProblem Name\tProblem View\tStep Name\t"
    "First Transaction Time\tCorrect First Attempt\tKC(Default)\n"
)


def test_algebra_multi_kc_missing_kc_and_time_order(tmp_path):
    rows = [
        "1\ts1\tU1\tP1\t1\tx=1\t2005-09-09 12:25:00.0\t1\tKC-a~~KC-b",
        "2\ts1\tU1\tP1\t1\tx=2\t2005-09-09 12:24:00.0\t0\tKC-a",
        "3\ts1\tU1\tP1\t1\tx=3\t2005-09-09 12:26:00.0\t1\t",
        "4\ts2\tU1\tP2\t1\ty=1\t2005-09-10 10:00:00.0\t1\tKC-b~~KC-b",
    ]
    p = tmp_path / "algebra_2005_2006_train.txt"
    p.write_text(ALGEBRA_HEADER + "\n".join(rows) + "\n", encoding="latin-1")
    col = load_algebra(p, COLLAPSED)
    assert col.cleaning["missing_skill_rows"] == 1
    assert col.cleaning["exact_duplicate_rows"] == 1
    assert col.attempt.tolist() == [2, 1, 4]  # s1 sorted by time, then s2
    assert col.skill_names[col.skill[1]] == "KC-a+KC-b"
    assert len(load_algebra(p, EXPANDED)) == 4
    assert len(load_algebra(p, RAW)) == 5
