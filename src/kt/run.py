"""End-to-end study for one dataset: every experiment, written to ``results/``."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .audit import assistments_file_audit, duplicate_structure
from .data import COLLAPSED, EXPANDED, LOADERS, RAW, DataError, Log
from .models.bkt import BKT
from .models.dkt import DKT
from .models.irt import IRT
from .policy import band_check_ci, mastery_tradeoff
from .study import (
    Candidate,
    bkt_identifiability,
    build_selected,
    evaluate_horizons,
    evaluate_row_split,
    evaluate_student_split,
)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, default=_jsonable) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def describe(log: Log) -> dict:
    first = log.first_row_of_attempt()
    return {
        "variant": log.variant,
        "rows": len(log),
        "attempts": int(first.sum()),
        "rows_per_attempt": len(log) / max(int(first.sum()), 1),
        "students": log.n_users,
        "items": log.n_items,
        "skills": log.n_skills,
        "correct_rate": float(log.correct.mean()),
        "cleaning": log.cleaning,
    }


def adjacent_copy_rate(log: Log) -> dict:
    """How often a row repeats the previous row's attempt (same order_id).

    Those rows are answered by the row before them: their outcome is identical
    by construction. A sequence model can learn to copy it; a per-skill model
    cannot, because the copy carries a different skill tag (or, for exact
    duplicates, the same one).
    """
    first = log.first_row_of_attempt()
    repeat = ~first
    return {
        "rows": len(log),
        "repeat_rows": int(repeat.sum()),
        "repeat_share": float(repeat.mean()),
    }


def _skill_ids(name: str) -> frozenset[str]:
    """Skill ids of a collapsed skill name: ours join ``id:name`` labels with
    ``+``; the official file writes a joint id such as ``1_13``."""
    ids: set[str] = set()
    for part in name.split("+"):
        ids.update(part.split(":", 1)[0].split("_"))
    return frozenset(ids)


def crosscheck_collapsed(ours: Log, official: Log, path: str | Path) -> dict:
    """Compare our collapse of the raw file with the publisher's collapsed release,
    row by row in sequence order: attempt id, student, problem, outcome, skill ids."""
    same_len = len(ours) == len(official)
    checks = {"same_row_count": same_len}
    if same_len:
        checks["same_attempt_ids_in_same_order"] = bool(
            np.array_equal(ours.attempt, official.attempt)
        )
        checks["same_students"] = all(
            ours.user_names[a] == official.user_names[b]
            for a, b in zip(ours.user, official.user, strict=True)
        )
        checks["same_problems"] = all(
            ours.item_names[a] == official.item_names[b]
            for a, b in zip(ours.item, official.item, strict=True)
        )
        checks["same_outcomes"] = bool(np.array_equal(ours.correct, official.correct))
        ours_ids = [_skill_ids(n) for n in ours.skill_names]
        off_ids = [_skill_ids(n) for n in official.skill_names]
        mismatched = sum(
            ours_ids[a] != off_ids[b] for a, b in zip(ours.skill, official.skill, strict=True)
        )
        checks["rows_with_different_skill_ids"] = int(mismatched)
    return {
        "official_file": Path(path).name,
        "ours": {
            "rows": len(ours),
            "students": ours.n_users,
            "skills": ours.n_skills,
            "items": ours.n_items,
        },
        "official": {
            "rows": len(official),
            "students": official.n_users,
            "skills": official.n_skills,
            "items": official.n_items,
        },
        **checks,
    }


def run_dataset(
    name: str,
    path: str | Path,
    out_dir: Path,
    variants: tuple[str, ...] = (RAW, EXPANDED, COLLAPSED),
    seed: int = 0,
    quick: bool = False,
    n_boot: int = 1000,
    models_dir: Path | None = None,
    verbose: bool = True,
    crosscheck: str | Path | None = None,
) -> dict:
    """Run every experiment for one dataset; returns a compact summary."""
    loader = LOADERS[name]
    t0 = time.time()
    logs = {v: loader(path, v) for v in variants}
    head = logs[COLLAPSED]
    data_info = {
        v: {**describe(lg), "adjacent_copies": adjacent_copy_rate(lg)} for v, lg in logs.items()
    }
    info: dict = {"dataset": name, "source": Path(path).name, "variants": data_info}
    if crosscheck is not None:
        info["crosscheck"] = crosscheck_collapsed(head, loader(crosscheck, COLLAPSED), crosscheck)
    if RAW in logs:
        info["duplicates"] = duplicate_structure(logs[RAW])
        if name == "assist09" and Path(path).suffix.lower() == ".csv":
            info["file_audit"] = assistments_file_audit(path)
    write_json(out_dir / "data.json", info)
    if verbose:
        print(
            f"[{name}] loaded {', '.join(f'{v}={len(lg)}' for v, lg in logs.items())} "
            f"rows in {time.time() - t0:.0f}s",
            flush=True,
        )

    # 1. headline: collapsed, split by student, full hyper-parameter grid
    if verbose:
        print(f"[{name}] student split, {COLLAPSED}", flush=True)
    head_res, head_extra = evaluate_student_split(head, seed, quick, verbose, n_boot)
    write_json(out_dir / f"student_split_{COLLAPSED}.json", head_res)
    selected = {fam: sel["selected"] for fam, sel in head_res["selection"].items()}
    models, parts = head_extra["models"], head_extra["parts"]

    # 1b. is the ranking an accident of which 208 students landed in test?
    for extra_seed in (seed + 1, seed + 2):
        if verbose:
            print(f"[{name}] student split, {COLLAPSED}, seed {extra_seed}", flush=True)
        res, _ = _student_split_fixed(head, selected, extra_seed, quick, verbose, n_boot)
        write_json(out_dir / f"student_split_{COLLAPSED}_seed{extra_seed}.json", res)

    # 2. the same models on the variants with duplicated rows
    for v in variants:
        if v == COLLAPSED:
            continue
        if verbose:
            print(f"[{name}] student split, {v}", flush=True)
        res, _ = _student_split_fixed(logs[v], selected, seed, quick, verbose, n_boot)
        write_json(out_dir / f"student_split_{v}.json", res)
        # the duplicate effect on the other two student splits as well
        for extra_seed in (seed + 1, seed + 2):
            if verbose:
                print(f"[{name}] student split, {v}, seed {extra_seed}", flush=True)
            res, _ = _student_split_fixed(logs[v], selected, extra_seed, quick, verbose, n_boot)
            write_json(out_dir / f"student_split_{v}_seed{extra_seed}.json", res)

    np.savez_compressed(
        out_dir / "test_predictions_collapsed.npz",
        rows=np.flatnonzero(head_extra["split"].mask("test")),
        **preds_by_family(head_extra["preds_test"]),
    )

    # 3. leakage: random row split
    if verbose:
        print(f"[{name}] row split", flush=True)
    write_json(
        out_dir / "row_split_collapsed.json",
        evaluate_row_split(head, selected, seed, quick, verbose),
    )

    # 4. next-attempt vs later-attempt
    if verbose:
        print(f"[{name}] horizons", flush=True)
    write_json(
        out_dir / "horizons_collapsed.json",
        {"dataset": name, **evaluate_horizons(models, parts["test"], n_boot=n_boot, seed=seed)},
    )

    # 5. policy checks on the test students
    test = parts["test"]
    preds = head_extra["preds_test"]
    policy = {
        "dataset": name,
        "band_0.7": {
            fam: band_check_ci(test.correct, p, test.user, 0.7, 0.05, n_boot, seed)
            for fam, p in preds.items()
        },
        "mastery": {fam: mastery_tradeoff(test, p) for fam, p in preds.items()},
    }
    write_json(out_dir / "policy_collapsed.json", policy)

    # 6. BKT identifiability
    write_json(
        out_dir / "bkt_identifiability.json", {"dataset": name, **bkt_identifiability(parts)}
    )

    if models_dir is not None:
        models_dir.mkdir(parents=True, exist_ok=True)
        bkt: BKT = models["BKT"]
        write_json(
            models_dir / f"{name}_bkt.json", {**bkt.to_json(), "skill_names": head.skill_names}
        )
        irt: IRT = models["IRT-1PL"]
        write_json(
            models_dir / f"{name}_irt.json", {**irt.to_json(), "skill_names": head.skill_names}
        )
        dkt: DKT = models["DKT"]
        dkt.save(str(models_dir / f"{name}_dkt.npz"))
    if verbose:
        print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)
    return head_res


def preds_by_family(preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"p_{k}": v for k, v in preds.items()}


def load_test_predictions(name: str, path: str | Path, results_dir: Path) -> tuple[Log, dict]:
    """The collapsed test split and the saved test predictions of every model.

    Rebuilds the split from the data (it is deterministic given the seed) and
    checks it is the one the predictions were made on.
    """
    from .splits import split_by_student

    saved_path = results_dir / name / "test_predictions_collapsed.npz"
    if not saved_path.exists():
        raise DataError(f"{saved_path} missing: run `kt study --dataset {name}` first")
    info = json.loads((results_dir / name / f"student_split_{COLLAPSED}.json").read_text("utf-8"))
    log = LOADERS[name](path, COLLAPSED)
    split = split_by_student(log, info["split"]["seed"])
    saved = np.load(saved_path)
    rows = np.flatnonzero(split.mask("test"))
    if not np.array_equal(rows, saved["rows"]):
        raise DataError("saved predictions were made on a different split or data file")
    preds = {k[2:]: saved[k] for k in saved.files if k.startswith("p_")}
    return log.subset_rows(split.mask("test")), preds


def _student_split_fixed(
    log: Log, selected: dict, seed: int, quick: bool, verbose: bool, n_boot: int
):
    """Student split with hyper-parameters fixed to those chosen on the collapsed
    variant (each model still early-stops / converges on this variant's data)."""
    built = build_selected(selected, quick)
    fixed = {fam: [Candidate(fam, selected[fam], lambda m=m: m)] for fam, m in built.items()}
    return evaluate_student_split(log, seed, quick, verbose, n_boot, families=fixed)
