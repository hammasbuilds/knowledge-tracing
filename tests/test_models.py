import numpy as np
import pytest
from helpers import TRUE_BKT, tiny_log

from kt.metrics import auc
from kt.models.baseline import ItemMean
from kt.models.bkt import BKT, UNBOUNDED, BKTBounds
from kt.models.dkt import DKT, LSTMParams, lstm_chunk
from kt.models.irt import IRT
from kt.models.pfa import PFA, counts
from kt.seq import positions


def test_bkt_em_recovers_generating_parameters(synth):
    m = BKT().fit(synth)
    for k, true in TRUE_BKT.items():
        assert np.max(np.abs(m.params[k] - np.asarray(true))) < 0.05, k


def test_bkt_bounds_are_enforced():
    m = BKT(bounds=BKTBounds(guess_max=0.1, slip_max=0.05))
    log = tiny_log([0] * 40, ([0, 1] * 20), users=[0] * 20 + [1] * 20)
    m.fit(log)
    assert m.params["guess"][0] <= 0.1 + 1e-12 and m.params["slip"][0] <= 0.05 + 1e-12


def reference_bkt(p, obs):
    """Straight-line BKT filtering for one skill sequence."""
    known, out = p["prior"], []
    for c in obs:
        out.append(known * (1 - p["slip"]) + (1 - known) * p["guess"])
        if c:
            post = known * (1 - p["slip"]) / (known * (1 - p["slip"]) + (1 - known) * p["guess"])
        else:
            post = known * p["slip"] / (known * p["slip"] + (1 - known) * (1 - p["guess"]))
        known = post + (1 - post) * p["learn"]
    return out


def test_bkt_prediction_matches_reference_filter(synth):
    m = BKT().fit(synth)
    log = tiny_log([0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 1])
    pred = m.predict(log)
    for s in (0, 1):
        rows = np.flatnonzero(log.skill == s)
        p = {k: m.params[k][s] for k in m.params}
        assert np.allclose(pred[rows], reference_bkt(p, log.correct[rows]))


def test_bkt_horizon_marginalises_hidden_attempts(synth):
    m = BKT().fit(synth)
    log = tiny_log([0, 0, 0], [1, 1, 1])
    p = {k: m.params[k][0] for k in m.params}
    # horizon 2 for the third row: only row 0 observed, row 1 is a hidden opportunity
    after0 = reference_bkt(p, [1, 1])[1]  # P(correct) at row 1 given row 0
    known1 = (after0 - p["guess"]) / (1 - p["slip"] - p["guess"])
    known2 = known1 + (1 - known1) * p["learn"]
    expect = known2 * (1 - p["slip"]) + (1 - known2) * p["guess"]
    assert m.predict(log, horizon=2)[2] == pytest.approx(expect)


def test_unbounded_em_can_leave_the_identifiable_region():
    # half the students already know the skill and slip often; bounded EM must stay put
    rng = np.random.default_rng(0)
    skills, correct, users = [], [], []
    for u in range(300):
        for _ in range(8):
            skills.append(0)
            correct.append(int(rng.random() < 0.55))
            users.append(u)
    log = tiny_log(skills, correct, users)
    bounded = BKT().fit(log)
    free = BKT(bounds=UNBOUNDED).fit(log)
    assert bounded.params["guess"][0] <= 0.3 and bounded.params["slip"][0] <= 0.3
    assert free.params["guess"][0] > 0.3 or free.params["slip"][0] > 0.3


def test_pfa_counts_match_brute_force():
    log = tiny_log([0, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 1])
    s, f = counts(log)
    assert s.tolist() == [0, 0, 1, 2, 0, 2]
    assert f.tolist() == [0, 0, 0, 0, 1, 1]
    s2, f2 = counts(log, horizon=3)
    assert s2.tolist() == [0, 0, 0, 1, 0, 2]  # row 5 sees rows 0..2 only
    assert f2.tolist() == [0, 0, 0, 0, 1, 0]


def test_pfa_learns_that_success_raises_p(synth, synth_test):
    m = PFA().fit(synth)
    assert np.all(m.weights[:, 1] > 0)
    assert auc(synth_test.correct, m.predict(synth_test)) > 0.7


def test_irt_orders_item_difficulty():
    rng = np.random.default_rng(0)
    n_users, diffs = 500, np.array([-1.5, 0.0, 1.5])
    theta = rng.normal(size=n_users)
    skills, correct, users = [], [], []
    for u in range(n_users):
        for i in range(3):
            for _ in range(3):
                skills.append(i)
                correct.append(int(rng.random() < 1 / (1 + np.exp(-(theta[u] - diffs[i])))))
                users.append(u)
    m = IRT(steps=400).fit(tiny_log(skills, correct, users))
    b = m.beta + m.delta
    assert b[0] < b[1] < b[2]


def test_irt_fitted_ability_leaks_but_online_does_not():
    # student 0 gets everything right; fitted mode knows it even for their first row
    log = tiny_log([0, 1] * 20, [1] * 20 + [0] * 20, users=[0] * 20 + [1] * 20)
    fitted = IRT(ability="fitted", steps=300).fit(log)
    online = IRT(steps=300).fit(log)
    first_rows = np.flatnonzero(positions(log) == 0)
    assert np.all(online.predict(log)[first_rows] == pytest.approx(online.predict(log)[first_rows][0]))
    pf = fitted.predict(log)[first_rows]
    assert pf[0] > pf[1] + 0.3
    with pytest.raises(ValueError):
        IRT(ability="magic")


MODELS = [
    lambda: ItemMean(),
    lambda: BKT(),
    lambda: PFA(),
    lambda: IRT(steps=100),
    lambda: IRT(steps=100, two_pl=True),
    lambda: DKT(hidden=8, max_epochs=1),
]


@pytest.mark.parametrize("make", MODELS)
@pytest.mark.parametrize("horizon", [1, 3])
def test_no_model_reads_the_answers_it_predicts(make, horizon, synth):
    """Flipping outcomes at positions > t - horizon must not move the prediction at t."""
    m = make().fit(synth.subset_users(np.arange(200)))
    log = synth.subset_users(np.arange(200, 230))
    pos = positions(log)
    base = m.predict(log, horizon)
    for t in (0, 4, 12):
        flipped = log.subset_rows(np.ones(len(log), dtype=bool))
        flipped.correct = log.correct.copy()
        hidden = pos > t - horizon
        flipped.correct[hidden] = 1 - flipped.correct[hidden]
        at_t = pos == t
        assert np.allclose(m.predict(flipped, horizon)[at_t], base[at_t], atol=1e-6)


def test_history_actually_moves_predictions(synth):
    m = BKT().fit(synth)
    good = m.predict(tiny_log([0] * 6, [1] * 6))[-1]
    bad = m.predict(tiny_log([0] * 6, [0] * 6))[-1]
    assert good > bad + 0.3


def test_lstm_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    p = LSTMParams.init(7, 5, 3, rng, np.float64)
    tok, sk = rng.integers(0, 7, (4, 6)), rng.integers(0, 3, (4, 6))
    y = rng.integers(0, 2, (4, 6)).astype(float)
    valid = rng.random((4, 6)) < 0.8
    h0, c0 = rng.normal(size=(4, 5)) * 0.1, rng.normal(size=(4, 5)) * 0.1
    _, grads, *_ = lstm_chunk(p, tok, sk, y, valid, h0, c0, 3.0)
    worst = 0.0
    for name, arr in p.arrays().items():
        for _ in range(5):
            idx = tuple(rng.integers(0, d) for d in arr.shape)
            old = arr[idx]
            arr[idx] = old + 1e-6
            up = lstm_chunk(p, tok, sk, y, valid, h0, c0, 3.0, grad=False)[0]
            arr[idx] = old - 1e-6
            down = lstm_chunk(p, tok, sk, y, valid, h0, c0, 3.0, grad=False)[0]
            arr[idx] = old
            num = (up - down) / 2e-6 / 3.0
            worst = max(worst, abs(num - grads[name][idx]) / (abs(num) + abs(grads[name][idx]) + 1e-9))
    assert worst < 1e-5


def test_dkt_learns_and_round_trips(synth, synth_test, tmp_path):
    m = DKT(hidden=16, max_epochs=4).fit(synth, synth_test)
    p = m.predict(synth_test)
    assert auc(synth_test.correct, p) > 0.75
    m.save(str(tmp_path / "dkt.npz"))
    again = DKT.load(str(tmp_path / "dkt.npz"))
    assert np.allclose(again.predict(synth_test), p)
    with pytest.raises(ValueError):
        m.predict(synth_test, horizon=0)
