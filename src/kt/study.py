"""The experiments behind the README, one function each.

Every function takes a loaded :class:`~kt.data.Log`, fits the model zoo on the
train split, selects hyper-parameters on the validation split only, and
returns a JSON-serialisable dict with train, val and test scores.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .audit import exact_duplicate_mask, split_concentration
from .data import Log
from .metrics import auc, bootstrap, calibration, score
from .models.baseline import ItemMean
from .models.bkt import BKT, UNBOUNDED
from .models.dkt import DKT
from .models.irt import IRT
from .models.pfa import PFA
from .seq import positions
from .splits import SPLIT_NAMES, Split, split_by_row, split_by_student

HORIZONS = (1, 2, 3, 5, 10)
EARLY = 5  # "first attempts" = positions 0..EARLY-1 in the student's sequence


@dataclass
class Candidate:
    family: str
    config: dict
    build: Callable[[], object]


def zoo(quick: bool = False) -> dict[str, list[Candidate]]:
    """Model families and the hyper-parameter grid each is tuned over on val."""
    dkt_grid = (
        [{"hidden": 32, "lr": 3e-3}]
        if quick
        else [{"hidden": h, "lr": lr} for h in (32, 64, 128) for lr in (1e-3, 3e-3)]
    )
    epochs = 3 if quick else 30
    return {
        "ItemMean": [
            Candidate("ItemMean", {"strength": k}, lambda k=k: ItemMean(strength=k))
            for k in (2.0, 10.0)
        ],
        "BKT": [Candidate("BKT", {"guess_max": 0.3, "slip_max": 0.3}, lambda: BKT())],
        "PFA": [Candidate("PFA", {"l2": v}, lambda v=v: PFA(l2=v)) for v in (0.1, 1.0, 10.0)],
        "IRT-1PL": [
            Candidate("IRT-1PL", {"item_sd": sd}, lambda sd=sd: IRT(item_sd=sd, name="IRT-1PL"))
            for sd in (0.5, 1.0, 2.0)
        ],
        "IRT-2PL": [
            Candidate(
                "IRT-2PL",
                {"item_sd": sd},
                lambda sd=sd: IRT(item_sd=sd, two_pl=True, name="IRT-2PL"),
            )
            for sd in (0.5, 1.0, 2.0)
        ],
        "DKT": [
            Candidate("DKT", cfg, lambda cfg=cfg: DKT(max_epochs=epochs, **cfg)) for cfg in dkt_grid
        ],
    }


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, flush=True)


def fit_family(cands: list[Candidate], train: Log, val: Log, verbose: bool = False):
    """Fit every candidate, keep the one with the best validation AUC."""
    tried = []
    best = None
    for c in cands:
        t0 = time.time()
        model = c.build()
        model.fit(train, val)
        v = auc(val.correct, model.predict(val))
        entry = {"config": c.config, "val_auc": v, "seconds": round(time.time() - t0, 1)}
        if isinstance(model, DKT):
            entry["epochs_run"] = len(model.history)
            entry["best_epoch"] = int(np.argmax([h.get("val_auc", -1) for h in model.history]) + 1)
        tried.append(entry)
        _log(f"    {c.family} {c.config}: val AUC {v:.4f} ({entry['seconds']}s)", verbose)
        if best is None or v > best[0]:
            best = (v, model, c.config)
    assert best is not None
    return best[1], {"selected": best[2], "grid": tried}


def _parts(log: Log, split: Split) -> dict[str, Log]:
    return {name: log.subset_rows(split.mask(name)) for name in SPLIT_NAMES}


def evaluate_student_split(
    log: Log,
    seed: int = 0,
    quick: bool = False,
    verbose: bool = False,
    n_boot: int = 1000,
    families: dict[str, list[Candidate]] | None = None,
) -> tuple[dict, dict]:
    """Headline experiment: split by student, tune on val, score train/val/test.

    Returns ``(results, extras)``; extras hold the fitted models, the split
    parts and the test predictions keyed by model family.
    """
    split = split_by_student(log, seed)
    parts = _parts(log, split)
    fams = families if families is not None else zoo(quick)
    models, selection = {}, {}
    for fam, cands in fams.items():
        _log(f"  fitting {fam}", verbose)
        models[fam], selection[fam] = fit_family(cands, parts["train"], parts["val"], verbose)
    scores: dict = {fam: {} for fam in models}
    preds_test: dict[str, np.ndarray] = {}
    for fam, m in models.items():
        for name, part in parts.items():
            p = m.predict(part)
            scores[fam][name] = score(part.correct, p)
            if name == "test":
                preds_test[fam] = p
                scores[fam]["test_calibration"] = calibration(part.correct, p)
    test = parts["test"]
    out: dict = {
        "dataset": log.name,
        "variant": log.variant,
        "split": {
            "kind": "student",
            "seed": seed,
            "fractions": split.fractions,
            "sizes": split.summary(log),
        },
        "selection": selection,
        "scores": scores,
    }
    first = test.first_row_of_attempt()
    if not first.all():
        out["scores_first_row_of_attempt"] = {
            fam: score(test.correct[first], p[first]) for fam, p in preds_test.items()
        }
    pairs = [("DKT", f) for f in models if f != "DKT"]
    out["bootstrap_test"] = bootstrap(test.correct, preds_test, test.user, n_boot, seed, pairs)
    if not first.all():
        out["bootstrap_test_first_row"] = bootstrap(
            test.correct[first],
            {k: v[first] for k, v in preds_test.items()},
            test.user[first],
            n_boot,
            seed,
            pairs,
        )
    dup = exact_duplicate_mask(log)
    if dup.any() and (keep_any := ~np.isin(test.user, np.unique(log.user[dup]))).any():
        # how much of the result is carried by the few students with repeated records?
        # (skipped when every test student has them - nothing would be left to score)
        owners = np.unique(log.user[dup])
        keep = keep_any
        out["repeated_records"] = {
            "concentration": split_concentration(log, split),
            "test_students_excluded": int(len(np.intersect1d(np.unique(test.user), owners))),
            "scores_without_those_students": {
                fam: score(test.correct[keep], p[keep]) for fam, p in preds_test.items()
            },
            "scores_without_those_students_first_row": {
                fam: score(test.correct[keep & first], p[keep & first])
                for fam, p in preds_test.items()
            },
        }
    extras = {"models": models, "parts": parts, "split": split}
    return out, {"preds_test": preds_test, **extras}


def bkt_identifiability(parts: dict[str, Log]) -> dict:
    """Bounded vs unbounded EM: how many skills go degenerate, and does it matter?"""
    out = {}
    for label, model in (("bounded", BKT()), ("unbounded", BKT(bounds=UNBOUNDED))):
        model.fit(parts["train"], parts["val"])
        out[label] = {
            "fit": model.fit_info,
            "val": score(parts["val"].correct, model.predict(parts["val"])),
            "test": score(parts["test"].correct, model.predict(parts["test"])),
        }
    return out


def evaluate_row_split(
    log: Log, selected: dict, seed: int = 0, quick: bool = False, verbose: bool = False
) -> dict:
    """Leaky split: rows assigned at random. Models are fitted on train rows and
    predict test rows with the student's full history (train and test rows)."""
    split = split_by_row(log, seed)
    train = log.subset_rows(split.mask("train"))
    val_mask, test_mask = split.mask("val"), split.mask("test")
    built = build_selected(selected, quick)
    built["IRT-1PL (fitted ability)"] = IRT(
        item_sd=selected["IRT-1PL"]["item_sd"], ability="fitted"
    )
    built["IRT-2PL (fitted ability)"] = IRT(
        item_sd=selected["IRT-2PL"]["item_sd"], two_pl=True, ability="fitted"
    )
    val_log = log.subset_rows(val_mask)
    early = test_mask & (positions(log) < EARLY)
    scores = {}
    for name, m in built.items():
        _log(f"  row split: fitting {name}", verbose)
        m.fit(train, val_log)
        p = m.predict(log)
        scores[name] = {
            "train": score(log.correct[split.mask("train")], p[split.mask("train")]),
            "val": score(log.correct[val_mask], p[val_mask]),
            "test": score(log.correct[test_mask], p[test_mask]),
            # a student's first attempts are where a fitted ability knows most
            # that the history does not
            f"test_first_{EARLY}_attempts": score(log.correct[early], p[early]),
        }
    return {
        "dataset": log.name,
        "variant": log.variant,
        "split": {
            "kind": "row",
            "seed": seed,
            "fractions": split.fractions,
            "sizes": split.summary(log),
        },
        "scores": scores,
    }


def build_selected(selected: dict, quick: bool) -> dict:
    epochs = 3 if quick else 30
    s = selected
    return {
        "ItemMean": ItemMean(**s["ItemMean"]),
        "BKT": BKT(),
        "PFA": PFA(**s["PFA"]),
        "IRT-1PL": IRT(item_sd=s["IRT-1PL"]["item_sd"], name="IRT-1PL"),
        "IRT-2PL": IRT(item_sd=s["IRT-2PL"]["item_sd"], two_pl=True, name="IRT-2PL"),
        "DKT": DKT(max_epochs=epochs, **s["DKT"]),
    }


def evaluate_horizons(
    models: dict, test: Log, min_pos: int = max(HORIZONS), n_boot: int = 1000, seed: int = 0
) -> dict:
    """Next-attempt vs later-attempt prediction on one fixed set of test rows.

    Only rows with at least ``min_pos`` earlier attempts are scored, so every
    horizon is evaluated on exactly the same rows and the curve is not a
    change of population.
    """
    keep = positions(test) >= min_pos
    y = test.correct[keep]
    out: dict = {
        "rows": int(keep.sum()),
        "students": int(len(np.unique(test.user[keep]))),
        "min_position": min_pos,
        "by_horizon": {},
    }
    for k in HORIZONS:
        preds = {fam: m.predict(test, k)[keep] for fam, m in models.items()}
        pairs = [("DKT", f) for f in preds if f != "DKT"]
        out["by_horizon"][str(k)] = {
            "scores": {fam: score(y, p) for fam, p in preds.items()},
            "bootstrap": bootstrap(y, preds, test.user[keep], n_boot, seed, pairs),
        }
    return out
