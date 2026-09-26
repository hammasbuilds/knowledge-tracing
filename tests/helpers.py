"""Shared test helpers (plain functions, importable from any test)."""

from __future__ import annotations

import numpy as np
from kt.data import COLLAPSED, Log

TRUE_BKT = {
    "prior": [0.2, 0.5, 0.1],
    "learn": [0.15, 0.05, 0.3],
    "guess": [0.2, 0.25, 0.1],
    "slip": [0.08, 0.1, 0.05],
}

ASSIST_HEADER = (
    "order_id,assignment_id,user_id,assistment_id,problem_id,original,correct,"
    "attempt_count,skill_id,skill_name\n"
)


def assist_csv(rows: list[tuple]) -> str:
    """rows: (order_id, user, problem, correct, skill_id, skill_name)."""
    lines = [ASSIST_HEADER]
    for oid, user, prob, correct, sid, sname in rows:
        lines.append(f"{oid},1,{user},1,{prob},1,{correct},1,{sid},{sname}\n")
    return "".join(lines)


def tiny_log(skills: list[int], correct: list[int], users: list[int] | None = None) -> Log:
    n = len(skills)
    users = users if users is not None else [0] * n
    n_skills = max(skills) + 1
    return Log(
        name="tiny",
        variant=COLLAPSED,
        user=np.asarray(users, dtype=np.int32),
        item=np.asarray(skills, dtype=np.int32),
        skill=np.asarray(skills, dtype=np.int32),
        correct=np.asarray(correct, dtype=np.int8),
        attempt=np.arange(n, dtype=np.int64),
        skill_names=[f"s{i}" for i in range(n_skills)],
        item_names=[f"s{i}" for i in range(n_skills)],
        user_names=[f"u{i}" for i in range(max(users) + 1)],
    )
