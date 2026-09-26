"""Scoring: AUC, RMSE, log loss, calibration, and student-level bootstrap CIs."""

from __future__ import annotations

import numpy as np

EPS = 1e-7


def _check(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs p {p.shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError("predictions contain NaN or inf")
    return y, p


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum statistic, ties averaged.

    Returns NaN when only one class is present (AUC is undefined there).
    """
    y, p = _check(y, p)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    ranks = np.empty(len(p), dtype=np.float64)
    # average rank for tied scores
    boundaries = np.flatnonzero(np.diff(ps)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(ps)]))
    avg = (starts + ends - 1) / 2.0 + 1.0
    ranks[order] = np.repeat(avg, ends - starts)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _check(y, p)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _check(y, p)
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Equal-width reliability bins and expected calibration error (ECE)."""
    y, p = _check(y, p)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    bins = []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += n / len(y) * abs(conf - acc)
        bins.append({"lo": b / n_bins, "hi": (b + 1) / n_bins, "n": n, "mean_p": conf, "rate": acc})
    return {"ece": float(ece), "bins": bins}


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = _check(y, p)
    return {
        "n": int(len(y)),
        "auc": auc(y, p),
        "rmse": rmse(y, p),
        "log_loss": log_loss(y, p),
        "ece": calibration(y, p)["ece"],
        "base_rate": float(y.mean()) if len(y) else float("nan"),
    }


def _groups(groups: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(groups, kind="mergesort")
    g = groups[order]
    cuts = np.flatnonzero(np.diff(g)) + 1
    return np.split(order, cuts)


def bootstrap(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    pairs: list[tuple[str, str]] | None = None,
) -> dict:
    """Cluster bootstrap over students: resample whole students with replacement.

    Rows of one student are not independent, so resampling rows would give CIs
    that are too narrow. Returns 95% percentile intervals for AUC and RMSE of
    each model, and for the paired AUC difference of every pair in ``pairs``
    (computed on the same resample, so shared noise cancels).
    """
    y = np.asarray(y, dtype=np.float64)
    members = _groups(np.asarray(groups))
    rng = np.random.default_rng(seed)
    names = list(preds)
    stats: dict[str, dict[str, list[float]]] = {k: {"auc": [], "rmse": []} for k in names}
    diffs: dict[str, list[float]] = {f"{a} - {b}": [] for a, b in (pairs or [])}
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        yb = y[idx]
        aucs = {}
        for k in names:
            pb = preds[k][idx]
            aucs[k] = auc(yb, pb)
            stats[k]["auc"].append(aucs[k])
            stats[k]["rmse"].append(float(np.sqrt(np.mean((yb - pb) ** 2))))
        for a, b in pairs or []:
            diffs[f"{a} - {b}"].append(aucs[a] - aucs[b])

    def ci(v: list[float]) -> list[float]:
        arr = np.asarray(v)
        arr = arr[np.isfinite(arr)]
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    out: dict = {"n_boot": n_boot, "n_students": len(members), "models": {}, "auc_diff": {}}
    for k in names:
        out["models"][k] = {"auc_ci": ci(stats[k]["auc"]), "rmse_ci": ci(stats[k]["rmse"])}
    for key, v in diffs.items():
        a, b = key.split(" - ")
        point = auc(y, preds[a]) - auc(y, preds[b])
        lo, hi = ci(v)
        out["auc_diff"][key] = {"point": point, "ci": [lo, hi], "excludes_zero": lo > 0 or hi < 0}
    return out
