import numpy as np
import pytest

from kt.metrics import auc, bootstrap, calibration, log_loss, rmse, score


def brute_auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return wins / (len(pos) * len(neg))


def test_auc_matches_pairwise_definition_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.round(rng.random(400), 1)  # many ties
    assert auc(y, p) == pytest.approx(brute_auc(y, p), abs=1e-12)


def test_auc_extremes_and_undefined():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc(y, np.full(4, 0.5)) == 0.5
    assert np.isnan(auc(np.ones(3), np.array([0.1, 0.2, 0.3])))


def test_rmse_logloss_and_nan_guard():
    y = np.array([1, 0])
    assert rmse(y, np.array([1.0, 0.0])) == 0.0
    assert log_loss(y, np.array([0.5, 0.5])) == pytest.approx(np.log(2))
    with pytest.raises(ValueError):
        rmse(y, np.array([np.nan, 0.1]))
    with pytest.raises(ValueError):
        auc(y, np.array([0.1]))


def test_calibration_is_zero_for_a_calibrated_predictor():
    rng = np.random.default_rng(1)
    p = np.repeat([0.15, 0.55, 0.85], 20000)
    y = (rng.random(len(p)) < p).astype(int)
    assert calibration(y, p)["ece"] < 0.01
    assert calibration(y, np.clip(p + 0.2, 0, 1))["ece"] > 0.15


def test_score_keys():
    s = score(np.array([0, 1, 1]), np.array([0.2, 0.7, 0.9]))
    assert set(s) == {"n", "auc", "rmse", "log_loss", "ece", "base_rate"}
    assert s["n"] == 3


def test_bootstrap_ci_brackets_point_and_detects_a_real_difference():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(200), 20)
    y = rng.integers(0, 2, len(groups))
    good = np.clip(y * 0.6 + rng.random(len(y)) * 0.4, 0, 1)
    noise = rng.random(len(y))
    out = bootstrap(y, {"good": good, "noise": noise}, groups, n_boot=200, pairs=[("good", "noise")])
    lo, hi = out["models"]["good"]["auc_ci"]
    assert lo <= auc(y, good) <= hi
    d = out["auc_diff"]["good - noise"]
    assert d["excludes_zero"] and d["ci"][0] > 0.3
    lo_n, hi_n = out["models"]["noise"]["auc_ci"]
    assert lo_n < 0.5 < hi_n
