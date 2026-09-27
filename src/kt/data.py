"""Loading student answer logs into one in-memory shape.

Every dataset becomes a :class:`Log`: flat, row-aligned numpy arrays sorted by
student and then by time, plus the vocabularies needed to turn ids back into
names. A row is one scored attempt. Two public datasets are supported:

* ASSISTments 2009-2010 skill-builder (``skill_builder_data_corrected.csv``).
* KDD Cup 2010 Algebra I 2005-2006 (``algebra_2005_2006_train.txt``).

Both ship multi-skill attempts. ASSISTments repeats the whole row once per
skill (same ``order_id``); the KDD file joins the skills with ``~~`` on one row.
:func:`load_assistments` and :func:`load_algebra` return the log in one of two
explicit shapes (see :data:`VARIANTS`) so the effect of that choice can be
measured instead of assumed.
"""

from __future__ import annotations

import csv
import io
import sys
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RAW = "raw"
"""Every row that has a skill, exactly as shipped: duplicated records are kept
and a multi-skill attempt appears once per skill tag. This is what the original
DKT paper trained and scored on."""

EXPANDED = "expanded"
"""One row per (attempt, skill): a multi-skill attempt appears once per skill,
back to back, with the same outcome, but exact duplicate records are removed.
This is the ASSISTments "corrected" release."""

COLLAPSED = "collapsed"
"""One row per attempt; a multi-skill attempt gets a single skill whose name is
the sorted combination of its skills (what the ASSISTments team's own
``*_collapsed.csv`` release does)."""

VARIANTS = (RAW, EXPANDED, COLLAPSED)


class DataError(ValueError):
    """Raised when an input file is missing or not in the expected format."""


@dataclass
class Log:
    """Row-aligned attempt log, sorted by (student, time)."""

    name: str
    variant: str
    user: np.ndarray  # int32, dense 0..n_users-1
    item: np.ndarray  # int32, dense 0..n_items-1
    skill: np.ndarray  # int32, dense 0..n_skills-1
    correct: np.ndarray  # int8, 0/1
    attempt: np.ndarray  # int64, id of the underlying attempt (shared by expanded rows)
    skill_names: list[str]
    item_names: list[str]
    user_names: list[str]
    cleaning: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.user)
        for name in ("item", "skill", "correct", "attempt"):
            if len(getattr(self, name)) != n:
                raise DataError(
                    f"column {name!r} has {len(getattr(self, name))} rows, expected {n}"
                )
        if n and np.any(np.diff(self.user) < 0):
            raise DataError("rows must be sorted by user")

    def __len__(self) -> int:
        return len(self.user)

    @property
    def n_users(self) -> int:
        return len(self.user_names)

    @property
    def n_items(self) -> int:
        return len(self.item_names)

    @property
    def n_skills(self) -> int:
        return len(self.skill_names)

    def user_bounds(self) -> np.ndarray:
        """``bounds[u]:bounds[u+1]`` is the row range of user ``u``."""
        return np.searchsorted(self.user, np.arange(self.n_users + 1)).astype(np.int64)

    def subset_users(self, users: np.ndarray) -> Log:
        """Rows of the given users only, keeping every vocabulary unchanged."""
        mask = np.isin(self.user, users)
        return self.subset_rows(mask)

    def subset_rows(self, mask: np.ndarray) -> Log:
        return Log(
            name=self.name,
            variant=self.variant,
            user=self.user[mask],
            item=self.item[mask],
            skill=self.skill[mask],
            correct=self.correct[mask],
            attempt=self.attempt[mask],
            skill_names=self.skill_names,
            item_names=self.item_names,
            user_names=self.user_names,
            cleaning=dict(self.cleaning),
        )

    def first_row_of_attempt(self) -> np.ndarray:
        """Boolean mask, True on the first row of every attempt.

        In the collapsed variant every row is its own attempt, so the mask is
        all True. In the expanded variant it marks one row per attempt, which is
        what an honest per-attempt score has to be computed on.
        """
        first = np.ones(len(self), dtype=bool)
        if len(self) > 1:
            first[1:] = (self.attempt[1:] != self.attempt[:-1]) | (self.user[1:] != self.user[:-1])
        return first


@dataclass(slots=True)
class _Row:
    user: str
    item: str
    skills: tuple[str, ...]
    raw_skills: tuple[str, ...]
    correct: int
    attempt: int
    order: tuple


def _build(name: str, variant: str, rows: list[_Row], cleaning: dict[str, int]) -> Log:
    if variant not in VARIANTS:
        raise DataError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    if not rows:
        raise DataError(f"{name}: no usable rows")
    rows.sort(key=lambda r: (r.user, r.order))
    users: dict[str, int] = {}
    items: dict[str, int] = {}
    skills: dict[str, int] = {}
    cols: dict[str, list[int]] = {k: [] for k in ("user", "item", "skill", "correct", "attempt")}
    multi = 0
    for r in rows:
        if len(r.skills) > 1:
            multi += 1
        if variant == RAW:
            emitted = r.raw_skills
        elif variant == EXPANDED:
            emitted = r.skills
        else:
            emitted = ("+".join(r.skills),)
        for s in emitted:
            cols["user"].append(users.setdefault(r.user, len(users)))
            cols["item"].append(items.setdefault(r.item, len(items)))
            cols["skill"].append(skills.setdefault(s, len(skills)))
            cols["correct"].append(r.correct)
            cols["attempt"].append(r.attempt)
    cleaning = dict(cleaning)
    cleaning["attempts"] = len(rows)
    cleaning["multi_skill_attempts"] = multi
    cleaning["rows"] = len(cols["user"])
    return Log(
        name=name,
        variant=variant,
        user=np.asarray(cols["user"], dtype=np.int32),
        item=np.asarray(cols["item"], dtype=np.int32),
        skill=np.asarray(cols["skill"], dtype=np.int32),
        correct=np.asarray(cols["correct"], dtype=np.int8),
        attempt=np.asarray(cols["attempt"], dtype=np.int64),
        skill_names=list(skills),
        item_names=list(items),
        user_names=list(users),
        cleaning=cleaning,
    )


def _open_text(path: Path, member_suffix: str) -> Iterator[str]:
    """Yield lines of ``path``, reading through a zip archive if needed."""
    if not path.exists():
        raise DataError(f"{path} does not exist (run scripts/fetch_data.sh first)")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            # skip macOS resource forks (__MACOSX/._name), which share the suffix
            names = [
                n
                for n in zf.namelist()
                if n.endswith(member_suffix)
                and not n.startswith("__MACOSX")
                and not n.rsplit("/", 1)[-1].startswith("._")
            ]
            if not names:
                raise DataError(f"{path}: no member ending in {member_suffix!r}")
            with zf.open(names[0]) as fh:
                yield from io.TextIOWrapper(fh, encoding="latin-1", newline="")
    else:
        with path.open(encoding="latin-1", newline="") as fh:
            yield from fh


def _id_key(sid: str) -> tuple[int, int, str]:
    """Numeric ids sort numerically; composite ids such as ``1_13`` sort after."""
    return (0, int(sid), "") if sid.isdigit() else (1, 0, sid)


ASSIST_REQUIRED = ("order_id", "user_id", "problem_id", "skill_id", "skill_name", "correct")


def read_assistments_rows(lines: Iterable[str]) -> tuple[list[_Row], dict[str, int]]:
    """Group raw ASSISTments 2009 rows into attempts.

    Rows sharing an ``order_id`` are one attempt; their distinct skills form
    that attempt's skill set. Handled explicitly and counted:

    * ``missing_skill_rows``: rows with an empty or ``NA`` ``skill_id`` (dropped; an
      attempt keeps its other skills if it had any).
    * ``exact_duplicate_rows``: a second row with the same ``order_id`` *and*
      skill (dropped - it carries no new information).
    * ``conflicting_attempts``: an ``order_id`` whose rows disagree on user,
      problem or correctness (dropped whole - it cannot be trusted).
    * ``non_binary_rows``: ``correct`` not 0/1 (dropped).
    """
    reader = csv.DictReader(lines)
    if reader.fieldnames is None or any(c not in reader.fieldnames for c in ASSIST_REQUIRED):
        raise DataError(f"not an ASSISTments 2009 file; need columns {ASSIST_REQUIRED}")
    counts = {
        "raw_rows": 0,
        "missing_skill_rows": 0,
        "exact_duplicate_rows": 0,
        "non_binary_rows": 0,
        "conflicting_attempts": 0,
    }
    groups: dict[int, dict] = {}
    for rec in reader:
        counts["raw_rows"] += 1
        if rec["correct"] not in ("0", "1"):
            counts["non_binary_rows"] += 1
            continue
        oid = int(rec["order_id"])
        key = (rec["user_id"], rec["problem_id"], int(rec["correct"]))
        g = groups.get(oid)
        if g is None:
            g = groups[oid] = {"key": key, "skills": {}, "raw": [], "bad": False}
        elif g["key"] != key:
            g["bad"] = True
        sid = rec["skill_id"].strip()
        if sid in ("", "NA"):
            counts["missing_skill_rows"] += 1
            continue
        label = f"{sid}:{rec['skill_name'].strip()}"
        g["raw"].append(label)
        if sid in g["skills"]:
            counts["exact_duplicate_rows"] += 1
            continue
        g["skills"][sid] = label
    rows: list[_Row] = []
    no_skill = 0
    for oid, g in groups.items():
        if g["bad"]:
            counts["conflicting_attempts"] += 1
            continue
        if not g["skills"]:
            no_skill += 1
            continue
        user, problem, correct = g["key"]
        skills = tuple(g["skills"][k] for k in sorted(g["skills"], key=_id_key))
        rows.append(_Row(user, problem, skills, tuple(g["raw"]), correct, oid, (oid,)))
    counts["attempts_without_any_skill"] = no_skill
    return rows, counts


def load_assistments(path: str | Path, variant: str = COLLAPSED) -> Log:
    """Load ASSISTments 2009-2010 skill-builder (csv or the distributed zip)."""
    rows, counts = read_assistments_rows(_open_text(Path(path), ".csv"))
    return _build("assist09", variant, rows, counts)


ALGEBRA_REQUIRED = (
    "Row",
    "Anon Student Id",
    "Problem Hierarchy",
    "Problem Name",
    "Step Name",
    "Correct First Attempt",
    "KC(Default)",
    "First Transaction Time",
)


def read_algebra_rows(lines: Iterable[str]) -> tuple[list[_Row], dict[str, int]]:
    """Parse the KDD Cup 2010 step-level training file.

    An item is a (problem, step) pair. Skills come from ``KC(Default)`` split on
    ``~~``; steps with no KC are dropped and counted, as are repeated KCs within
    one step (``exact_duplicate_rows``). Rows are ordered by first transaction
    time, ties broken by the file's row number.
    """
    reader = csv.DictReader(lines, delimiter="\t")
    if reader.fieldnames is None or any(c not in reader.fieldnames for c in ALGEBRA_REQUIRED):
        raise DataError(f"not a KDD Cup 2010 step file; need columns {ALGEBRA_REQUIRED}")
    counts = {
        "raw_rows": 0,
        "missing_skill_rows": 0,
        "exact_duplicate_rows": 0,
        "non_binary_rows": 0,
    }
    rows: list[_Row] = []
    for rec in reader:
        counts["raw_rows"] += 1
        cfa = rec["Correct First Attempt"]
        if cfa not in ("0", "1"):
            counts["non_binary_rows"] += 1
            continue
        kc = (rec["KC(Default)"] or "").strip()
        if not kc:
            counts["missing_skill_rows"] += 1
            continue
        parts = [p.strip() for p in kc.split("~~") if p.strip()]
        uniq = tuple(sorted(set(parts)))
        counts["exact_duplicate_rows"] += len(parts) - len(uniq)
        row_id = int(rec["Row"])
        order = (rec["First Transaction Time"] or "", row_id)
        item = sys.intern(f"{rec['Problem Hierarchy']}|{rec['Problem Name']}|{rec['Step Name']}")
        rows.append(
            _Row(
                sys.intern(rec["Anon Student Id"]),
                item,
                uniq,
                tuple(parts),
                int(cfa),
                row_id,
                order,
            )
        )
    return rows, counts


def load_algebra(path: str | Path, variant: str = COLLAPSED) -> Log:
    """Load KDD Cup 2010 Algebra I 2005-2006 (train file, txt or zip)."""
    rows, counts = read_algebra_rows(_open_text(Path(path), "_train.txt"))
    return _build("algebra05", variant, rows, counts)


LOADERS = {"assist09": load_assistments, "algebra05": load_algebra}
