"""What a tutor needs at run time: load a saved model, read a student's history,
say how well each skill is known, and pick the next one.

Two saved models can drive it:

* **IRT** (default): ``P(correct)`` from an ability re-estimated on the student's
  own history. The best-ranked and best-calibrated model in the study. Mastery
  is judged on ``P(correct)``.
* **BKT**: mastery is judged on ``P(known)``, the latent state, never on
  ``P(correct)``: BKT's P(correct) is capped at ``1 - slip`` (0.70 for skills
  whose slip sits at the 0.3 bound), so a P(correct) threshold of 0.95 would
  be unreachable for most skills and a mastered skill would stay in rotation.

DKT is not offered here: it reads the history as a sequence of (skill,
correct) tokens and has no per-skill state to report or threshold.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from .data import DataError
from .models.bkt import BKT
from .models.irt import GRID, IRT
from .policy import recommend

BUILTIN = {"irt": "assist09_irt.json", "bkt": "assist09_bkt.json"}


@dataclass
class SkillStatus:
    skill: str
    p_correct: float
    mastery_value: float  # P(known) for BKT, P(correct) for IRT
    ceiling: float  # highest mastery_value this skill can reach
    practised: bool


@dataclass
class TutorModel:
    kind: str  # "bkt" or "irt"
    model: BKT | IRT
    skill_names: list[str]
    source: str

    @property
    def mastery_label(self) -> str:
        return "P(known)" if self.kind == "bkt" else "P(correct)"

    def status(self, history: list[tuple[str, int]]) -> list[SkillStatus]:
        ids = np.array([resolve_skill(s, self.skill_names) for s, _ in history], dtype=np.int64)
        correct = np.array([c for _, c in history], dtype=np.int8)
        targets = np.arange(len(self.skill_names))
        if self.kind == "bkt":
            assert isinstance(self.model, BKT)
            p, known = self.model.predict_skills(ids, correct, targets)
            mastery, ceiling = known, np.ones(len(targets))
        else:
            assert isinstance(self.model, IRT)
            p = self.model.predict_skills(ids, correct, targets)
            mastery = p
            a = np.exp(self.model.log_a)
            ceiling = 1.0 / (1.0 + np.exp(-(a * GRID[-1] - self.model.beta)))
        seen = set(ids.tolist())
        return [
            SkillStatus(n, float(p[i]), float(mastery[i]), float(ceiling[i]), i in seen)
            for i, n in enumerate(self.skill_names)
        ]


def load_model(spec: str) -> TutorModel:
    """``spec`` is ``irt`` / ``bkt`` (the packaged ASSISTments 2009 models) or a
    path to a JSON file written by ``kt study``."""
    if spec in BUILTIN:
        ref = resources.files("kt") / "pretrained" / BUILTIN[spec]
        text, source = ref.read_text(encoding="utf-8"), f"packaged {BUILTIN[spec]}"
    else:
        path = Path(spec)
        if not path.exists():
            raise DataError(f"model {spec!r} not found: use 'irt', 'bkt' or a JSON path")
        if path.suffix.lower() == ".npz":
            raise DataError(
                f"{path} is a DKT model; recommend needs a BKT or IRT JSON (it reports a "
                "per-skill state, which DKT does not have)"
            )
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise DataError(f"{path} is not a text JSON model file") from None
        source = str(path)
    try:
        d = json.loads(text)
        kind = d["model"]
        names = list(d["skill_names"])
    except (json.JSONDecodeError, KeyError, TypeError):
        raise DataError(
            f"{source} is not a kt model JSON (need 'model' and 'skill_names')"
        ) from None
    if kind == "bkt":
        model: BKT | IRT = BKT.from_json(d)
    elif kind == "irt":
        model = IRT.from_json(d)
    else:
        raise DataError(f"{source}: unsupported model {kind!r} (bkt or irt)")
    return TutorModel(kind, model, names, source)


def read_history(path: Path) -> list[tuple[str, int]]:
    """CSV with ``skill`` and ``correct`` (0/1) columns, oldest first.

    Header names are matched case-insensitively and a UTF-8 byte-order mark (as
    Excel writes) is ignored.
    """
    if not path.exists():
        raise DataError(f"history file {path} does not exist")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise DataError(f"{path} is empty")
        cols = {h.strip().lower(): i for i, h in enumerate(header)}
        if not {"skill", "correct"} <= set(cols):
            raise DataError(
                f"{path}: need a header with 'skill' and 'correct' columns, got {header}"
            )
        out = []
        for line, rec in enumerate(reader, 2):
            if not any(x.strip() for x in rec):
                continue
            if len(rec) <= max(cols["skill"], cols["correct"]):
                raise DataError(f"{path} line {line}: too few columns")
            c = rec[cols["correct"]].strip()
            if c not in ("0", "1"):
                raise DataError(f"{path} line {line}: correct must be 0 or 1, got {c!r}")
            out.append((rec[cols["skill"]].strip(), int(c)))
    return out


def resolve_skill(query: str, names: list[str]) -> int:
    """Match a skill by exact name, by its id prefix (``311``), or by its label."""
    for i, n in enumerate(names):
        if query == n or query == n.split(":", 1)[0] or query == n.split(":", 1)[-1]:
            return i
    lowered = [i for i, n in enumerate(names) if query.lower() in n.lower()]
    if len(lowered) == 1:
        return lowered[0]
    hint = f"; {len(lowered)} names contain it" if lowered else ""
    raise DataError(f"unknown skill {query!r}{hint}. Run `kt skills` to list them.")


@dataclass
class Plan:
    table: list[SkillStatus]
    mastered: set[str]
    unreachable: set[str]
    next_skill: str | None
    reason: str


def plan_next(
    tm: TutorModel,
    history: list[tuple[str, int]],
    target: float = 0.7,
    mastery: float = 0.95,
    candidates: list[str] | None = None,
) -> Plan:
    """Pick the unmastered candidate whose P(correct) is closest to ``target``.

    Candidates default to the skills already practised (or all skills for an
    empty history). A skill whose ceiling is below ``mastery`` can never be
    declared mastered and is flagged, so the caller knows it will stay in rotation.
    """
    if not 0 < mastery <= 1:
        raise ValueError("mastery must be in (0, 1]")
    table = tm.status(history)
    by_name = {s.skill: s for s in table}
    if candidates:
        pool = [tm.skill_names[resolve_skill(c, tm.skill_names)] for c in candidates]
    else:
        pool = [s.skill for s in table if s.practised] or list(by_name)
    rows = [by_name[n] for n in dict.fromkeys(pool)]
    mastered = {s.skill for s in rows if s.mastery_value >= mastery}
    unreachable = {s.skill for s in rows if s.ceiling < mastery}
    if len(mastered) == len(rows):
        return Plan(rows, mastered, unreachable, None, "every candidate skill is mastered")
    rec = recommend({s.skill: s.p_correct for s in rows}, target, mastered)
    return Plan(rows, mastered, unreachable, rec.skill, rec.reason)
