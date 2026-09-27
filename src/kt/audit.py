"""Where the duplicated rows in ASSISTments 2009 come from, and who owns them.

The original release contains two different kinds of repeated rows, which
matter differently:

* **multi-skill rows** - one attempt tagged with k skills is written k times,
  once per skill. Legitimate in a per-skill model's eyes (the attempt really
  is evidence about each skill), but a sequence model sees the same answer k
  times in a row.
* **repeated records** - the same (attempt, skill) row written again, many
  times over. These are the "duplicated data records" the publisher later
  removed. They are not byte-identical: bookkeeping columns such as
  ``opportunity`` differ between the copies.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from .data import RAW, DataError, Log
from .splits import SPLIT_NAMES, Split


def exact_duplicate_mask(log: Log) -> np.ndarray:
    """True on every row whose (attempt, skill) occurs more than once."""
    key = log.attempt.astype(np.int64) * (log.n_skills + 1) + log.skill
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return cnt[inv] > 1


def duplicate_structure(raw: Log) -> dict:
    """Split the raw file's extra rows into multi-skill rows and repeated records."""
    if raw.variant != RAW:
        raise ValueError("duplicate_structure needs the raw variant")
    key = raw.attempt.astype(np.int64) * (raw.n_skills + 1) + raw.skill
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    first = raw.first_row_of_attempt()
    attempts = int(first.sum())
    distinct_pairs = len(uniq)
    dup_groups = cnt > 1
    dup_attempts = np.unique(raw.attempt[cnt[inv] > 1])
    dup_rows_user = raw.user[cnt[inv] > 1]
    extra_per_user = np.bincount(dup_rows_user, minlength=raw.n_users)
    owners = np.flatnonzero(extra_per_user)
    top = np.sort(extra_per_user)[::-1]
    return {
        "rows": len(raw),
        "attempts": attempts,
        "multi_skill_extra_rows": distinct_pairs - attempts,
        "repeated_record_extra_rows": len(raw) - distinct_pairs,
        "repeated_groups": int(dup_groups.sum()),
        "repeated_attempts": len(dup_attempts),
        "copies_per_repeated_group": {
            "median": float(np.median(cnt[dup_groups])) if dup_groups.any() else 0.0,
            "p90": float(np.percentile(cnt[dup_groups], 90)) if dup_groups.any() else 0.0,
            "max": int(cnt.max()) if len(cnt) else 0,
        },
        "students_owning_repeated_records": len(owners),
        "share_of_repeated_rows_owned_by_top_10_students": float(
            top[:10].sum() / max(top.sum(), 1)
        ),
    }


def split_concentration(raw: Log, split: Split) -> dict:
    """Per split: rows per attempt, and how much of it is owned by the students
    who have repeated records."""
    dup = exact_duplicate_mask(raw)
    owners = np.unique(raw.user[dup])
    out = {}
    for name in SPLIT_NAMES:
        m = split.mask(name)
        first = raw.first_row_of_attempt() & m
        owned = m & np.isin(raw.user, owners)
        out[name] = {
            "rows": int(m.sum()),
            "attempts": int(first.sum()),
            "rows_per_attempt": float(m.sum() / max(first.sum(), 1)),
            "students": int(len(np.unique(raw.user[m]))),
            "students_with_repeated_records": int(
                len(np.intersect1d(np.unique(raw.user[m]), owners))
            ),
            "rows_of_those_students": int(owned.sum()),
            "share_of_rows": float(owned.sum() / max(m.sum(), 1)),
        }
    return out


def assistments_file_audit(path: str | Path) -> dict:
    """Read the ASSISTments csv in file order (no sorting) and report:

    * how often a row repeats the previous row's order_id *in file order*
      (the loaders sort by student and time, as DKT pipelines do, which puts
      every copy next to its original);
    * which columns differ between copies of one (order_id, skill_id) row.
    """
    path = Path(path)
    if not path.exists() or path.suffix.lower() != ".csv":
        raise DataError(f"{path}: the file audit reads the ASSISTments csv directly")
    rows_with_skill = 0
    repeats_prev = 0
    prev = None
    first_seen: dict[tuple[str, str], dict[str, str]] = {}
    differing: Counter[str] = Counter()
    copies = 0
    identical = 0
    with path.open(encoding="latin-1", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = [c for c in (reader.fieldnames or []) if c]
        for rec in reader:
            sid = rec["skill_id"].strip()
            if sid in ("", "NA"):
                continue
            rows_with_skill += 1
            oid = rec["order_id"]
            if prev == oid:
                repeats_prev += 1
            prev = oid
            key = (oid, sid)
            base = first_seen.get(key)
            if base is None:
                first_seen[key] = {c: rec[c] for c in cols}
                continue
            copies += 1
            diff = [c for c in cols if rec[c] != base[c]]
            identical += not diff
            differing.update(diff)
    return {
        "rows_with_skill": rows_with_skill,
        "file_order_repeat_previous_order_id": repeats_prev,
        "file_order_repeat_share": repeats_prev / max(rows_with_skill, 1),
        "repeated_record_copies": copies,
        "byte_identical_copies": identical,
        "columns_differing_in_copies": dict(differing.most_common()),
    }
