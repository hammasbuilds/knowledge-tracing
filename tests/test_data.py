import zipfile

import numpy as np
import pytest
from helpers import assist_csv

from kt.data import COLLAPSED, EXPANDED, RAW, DataError, load_algebra, load_assistments

ROWS = [
    (1, "u1", "p10", 1, "5", "A"),
    (1, "u1", "p10", 1, "7", "B"),  # multi-skill attempt
    (1, "u1", "p10", 1, "5", "A"),  # repeated (order_id, skill) record
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
    assert c["repeated_order_skill_rows"] == 1
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
    assert col.cleaning["repeated_kc_in_step"] == 1
    assert col.attempt.tolist() == [2, 1, 4]  # s1 sorted by time, then s2
    assert col.skill_names[col.skill[1]] == "KC-a+KC-b"
    assert len(load_algebra(p, EXPANDED)) == 4
    assert len(load_algebra(p, RAW)) == 5


def test_zip_skips_macos_resource_forks(tmp_path):
    good = ALGEBRA_HEADER + "1\ts1\tU1\tP1\t1\tx=1\t2005-09-09 12:25:00.0\t1\tKC-a\n"
    z = tmp_path / "algebra.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("__MACOSX/a/._algebra_2005_2006_train.txt", b"\x00\x05\x16\x07junk")
        zf.writestr("a/algebra_2005_2006_train.txt", good)
    assert len(load_algebra(z, COLLAPSED)) == 1


def test_crosscheck_compares_students_problems_order_and_skill_ids(tmp_path):
    from kt.run import crosscheck_collapsed

    ours_rows = [
        (1, "u1", "p1", 1, "5", "A"),
        (1, "u1", "p1", 1, "7", "B"),
        (2, "u1", "p2", 0, "5", "A"),
        (3, "u2", "p3", 1, "7", "B"),
    ]
    official = [
        (1, "u1", "p1", 1, "5_7", "A"),
        (2, "u1", "p2", 0, "5", "A"),
        (3, "u2", "p3", 1, "7", "B"),
    ]

    def load(rows, name):
        p = tmp_path / name
        p.write_text(assist_csv(rows), encoding="latin-1")
        return load_assistments(p, COLLAPSED)

    ours = load(ours_rows, "ours.csv")
    same = crosscheck_collapsed(ours, load(official, "off.csv"), "off.csv")
    assert same["same_attempt_ids_in_same_order"] and same["same_students"]
    assert same["same_problems"] and same["same_outcomes"]
    assert same["rows_with_different_skill_ids"] == 0
    wrong = [official[0], (2, "u1", "p9", 0, "5", "A"), (3, "u2", "p3", 1, "9", "B")]
    diff = crosscheck_collapsed(ours, load(wrong, "wrong.csv"), "wrong.csv")
    assert not diff["same_problems"] and diff["rows_with_different_skill_ids"] == 1


def test_duplicate_audit_separates_multi_skill_rows_from_repeated_records(tmp_path):
    from kt.audit import assistments_file_audit, duplicate_structure, repeated_record_mask

    rows = [
        (1, "u1", "p1", 1, "5", "A"),
        (2, "u1", "p2", 0, "5", "A"),
        (1, "u1", "p1", 1, "7", "B"),  # multi-skill: second tag of attempt 1
        (1, "u1", "p1", 1, "5", "A"),  # repeated record of (1, 5)
        (1, "u1", "p1", 1, "5", "A"),  # and again
        (3, "u2", "p3", 1, "7", "B"),
    ]
    text = assist_csv(rows).splitlines()
    text[5] = text[5].replace(",1,1,5,A", ",1,9,5,A")  # a copy whose attempt_count differs
    p = tmp_path / "sb.csv"
    p.write_text("\n".join(text) + "\n", encoding="latin-1")
    raw = load_assistments(p, RAW)
    d = duplicate_structure(raw)
    assert d["attempts"] == 3 and d["multi_skill_extra_rows"] == 1
    assert d["repeated_record_extra_rows"] == 2 and d["repeated_attempts"] == 1
    assert d["copies_per_repeated_group"]["max"] == 3
    assert d["students_owning_repeated_records"] == 1
    assert repeated_record_mask(raw).sum() == 3
    audit = assistments_file_audit(p)
    assert audit["repeated_record_copies"] == 2 and audit["byte_identical_copies"] == 1
    assert audit["columns_differing_in_copies"] == {"attempt_count": 1}
    assert audit["file_order_repeat_previous_order_id"] == 2  # rows 4-5 follow order 1
    with pytest.raises(ValueError):
        duplicate_structure(load_assistments(p, COLLAPSED))
