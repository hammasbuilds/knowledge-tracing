import numpy as np
import pytest
from helpers import tiny_log

from kt.policy import band_check, band_check_ci, mastery_tradeoff, recommend


def test_recommend_picks_closest_to_target_and_skips_mastered():
    p = {"fractions": 0.35, "decimals": 0.68, "angles": 0.97, "area": 0.8}
    assert recommend(p).skill == "decimals"
    assert recommend(p, target=0.9).skill == "angles"
    assert recommend(p, target=0.9, mastered={"angles"}).skill == "area"
    with pytest.raises(ValueError):
        recommend(p, target=1.0)
    with pytest.raises(ValueError):
        recommend({"a": 0.5}, mastered={"a"})


def test_recommend_tie_break_is_deterministic():
    assert recommend({"b": 0.25, "a": 0.75}, target=0.5).skill == "a"


def test_band_check_reports_the_gap():
    p = np.array([0.7] * 100 + [0.2] * 100)
    y = np.array([1] * 90 + [0] * 10 + [0] * 100)
    out = band_check(y, p)
    assert out["n"] == 100 and out["share"] == 0.5
    assert out["observed_rate"] == pytest.approx(0.9)
    assert out["gap"] == pytest.approx(0.2)
    assert band_check(y, np.full(200, 0.1))["n"] == 0


def test_band_check_ci_brackets_the_rate():
    rng = np.random.default_rng(0)
    p = np.full(2000, 0.7)
    y = (rng.random(2000) < 0.7).astype(int)
    out = band_check_ci(y, p, np.repeat(np.arange(100), 20), n_boot=200)
    lo, hi = out["observed_rate_ci"]
    assert lo < out["observed_rate"] < hi and lo < 0.7 < hi


def test_mastery_tradeoff_effort_and_score():
    # one student, one skill: predictions rise, threshold 0.8 first reached at index 2
    log = tiny_log([0] * 5, [0, 1, 1, 0, 1])
    p = np.array([0.3, 0.6, 0.85, 0.9, 0.95])
    out = mastery_tradeoff(log, p, thresholds=(0.8, 0.99))
    t = out["thresholds"]["0.8"]
    assert t["mean_effort"] == 2 and t["declared_share"] == 1.0
    assert t["post_declaration_attempts"] == 3
    assert t["post_declaration_correct_rate"] == pytest.approx(2 / 3)
    never = out["thresholds"]["0.99"]
    assert never["declared_share"] == 0.0 and never["mean_effort"] == 5
    assert never["post_declaration_correct_rate"] is None


def test_three_in_a_row_rule():
    log = tiny_log([0] * 6 + [1] * 3, [1, 1, 0, 1, 1, 1, 1, 0, 1])
    out = mastery_tradeoff(log, np.zeros(9))["three_in_a_row"]
    # skill 0 reaches 3-in-a-row after its 6th attempt; skill 1 never does
    assert out["declared_share"] == 0.5
    assert out["mean_effort"] == pytest.approx((6 + 3) / 2)
    assert out["post_declaration_attempts"] == 0
