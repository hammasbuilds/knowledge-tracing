import json
from pathlib import Path

import numpy as np
import pytest
from helpers import tiny_log

from kt.cli import main
from kt.data import DataError
from kt.models.bkt import BKT
from kt.models.irt import IRT
from kt.tutor import TutorModel, load_model, plan_next, read_history

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["1:Adding", "2:Fractions", "3:Angles"]


def bkt_model(slip_last: float = 0.3) -> TutorModel:
    m = BKT()
    m.params = {
        "prior": np.array([0.3, 0.3, 0.3]),
        "learn": np.array([0.2, 0.2, 0.2]),
        "guess": np.array([0.2, 0.2, 0.3]),
        "slip": np.array([0.1, 0.1, slip_last]),
    }
    m.pooled = {"prior": 0.3, "learn": 0.2, "guess": 0.2, "slip": 0.1}
    return TutorModel("bkt", m, NAMES, "test")


def test_bkt_predict_skills_matches_the_sequence_filter():
    tm = bkt_model()
    hist_s = [0, 1, 0, 0, 2, 1]
    hist_c = [0, 1, 1, 1, 0, 1]
    log = tiny_log(hist_s + [0, 1, 2], hist_c + [0, 0, 0])
    expect = tm.model.predict(log)[-3:]
    p, known = tm.model.predict_skills(np.array(hist_s), np.array(hist_c), np.arange(3))
    assert np.allclose(p, expect)
    assert np.all((known >= 0) & (known <= 1))


def test_bkt_mastery_uses_p_known_so_a_capped_skill_can_be_mastered():
    tm = bkt_model(slip_last=0.3)  # P(correct) on "Angles" can never exceed 0.70
    history = [("Angles", 1)] * 12 + [("Adding", 0), ("Fractions", 0)]
    plan = plan_next(tm, history, target=0.7, mastery=0.95)
    angles = next(s for s in plan.table if s.skill == "3:Angles")
    assert angles.p_correct <= 0.70 + 1e-9
    assert angles.mastery_value > 0.95
    assert "3:Angles" in plan.mastered
    assert plan.next_skill != "3:Angles"  # 0.70 == target, but it is mastered
    assert not plan.unreachable


def test_all_mastered_returns_no_next_exercise():
    tm = bkt_model()
    history = [(n, 1) for n in ("Adding", "Fractions", "Angles") for _ in range(15)]
    plan = plan_next(tm, history)
    assert plan.next_skill is None and len(plan.mastered) == 3


def irt_model() -> TutorModel:
    m = IRT()
    m.beta = np.array([-1.0, 0.0, 3.5])
    m.log_a = np.zeros(3)
    return TutorModel("irt", m, NAMES, "test")


def test_irt_ability_moves_with_history_and_flags_unreachable_mastery():
    tm = irt_model()
    none = {s.skill: s for s in tm.status([])}
    good = {s.skill: s for s in tm.status([("Adding", 1)] * 8)}
    bad = {s.skill: s for s in tm.status([("Adding", 0)] * 8)}
    assert (
        bad["2:Fractions"].p_correct < none["2:Fractions"].p_correct < good["2:Fractions"].p_correct
    )
    plan = plan_next(tm, [("Angles", 1)] * 3, mastery=0.95)
    assert "3:Angles" in plan.unreachable  # sigmoid(4 - 3.5) < 0.95 even at the top of the grid


def test_irt_predict_skills_matches_online_predict():
    rng = np.random.default_rng(0)
    users = [0] * 40
    skills = rng.integers(0, 3, 40).tolist()
    correct = rng.integers(0, 2, 40).tolist()
    m = IRT(steps=100).fit(tiny_log(skills, correct, users))
    m.delta = np.zeros(0)  # new items: skill difficulty only
    log = tiny_log(skills[:10] + [2], correct[:10] + [0])
    online = m.predict(log)[-1]
    direct = m.predict_skills(np.array(skills[:10]), np.array(correct[:10]), np.array([2]))[0]
    assert online == pytest.approx(direct)
    again = IRT.from_json(m.to_json())
    assert np.allclose(
        again.predict_skills(np.array([0]), np.array([1]), np.arange(3)),
        m.predict_skills(np.array([0]), np.array([1]), np.arange(3)),
    )


def test_read_history_accepts_excel_bom_and_any_header_case(tmp_path):
    p = tmp_path / "h.csv"
    p.write_bytes("\ufeffSkill , CORRECT\nAdding,1\n\n Fractions ,0\n".encode())
    assert read_history(p) == [("Adding", 1), ("Fractions", 0)]
    with pytest.raises(DataError, match="does not exist"):
        read_history(tmp_path / "missing.csv")
    (tmp_path / "e.csv").write_text("", encoding="utf-8")
    with pytest.raises(DataError, match="empty"):
        read_history(tmp_path / "e.csv")
    (tmp_path / "s.csv").write_text("skill,correct\nAdding\n", encoding="utf-8")
    with pytest.raises(DataError, match="too few columns"):
        read_history(tmp_path / "s.csv")


def test_load_model_rejects_wrong_files_cleanly(tmp_path):
    with pytest.raises(DataError, match="not found"):
        load_model(str(tmp_path / "nope.json"))
    npz = tmp_path / "dkt.npz"
    npz.write_bytes(b"PK\x03\x04binary")
    with pytest.raises(DataError, match="DKT"):
        load_model(str(npz))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(DataError, match="not a kt model"):
        load_model(str(bad))
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"model": "dkt", "skill_names": []}), encoding="utf-8")
    with pytest.raises(DataError, match="unsupported"):
        load_model(str(other))


@pytest.mark.parametrize("kind", ["irt", "bkt"])
def test_packaged_models_load_and_match_the_committed_ones(kind):
    tm = load_model(kind)
    assert tm.kind == kind and len(tm.skill_names) == 149
    packaged = (ROOT / "src" / "kt" / "pretrained" / f"assist09_{kind}.json").read_text("utf-8")
    committed = (ROOT / "models" / f"assist09_{kind}.json").read_text("utf-8")
    assert json.loads(packaged) == json.loads(committed)


def test_cli_errors_are_clean(tmp_path, capsys):
    assert main(["recommend", "--history", str(tmp_path / "missing.csv")]) == 2
    assert "does not exist" in capsys.readouterr().err
    h = tmp_path / "h.csv"
    h.write_text("skill,correct\nVenn Diagram,1\n", encoding="utf-8")
    npz = tmp_path / "m.npz"
    npz.write_bytes(b"PK")
    assert main(["recommend", "--model", str(npz), "--history", str(h)]) == 2
    assert "DKT" in capsys.readouterr().err
    assert main(["recommend", "--history", str(h), "--mastery", "1.5"]) == 2
    assert "mastery" in capsys.readouterr().err


def test_cli_recommend_with_packaged_irt(tmp_path, capsys):
    h = tmp_path / "h.csv"
    h.write_bytes(
        "\ufeffSKILL,Correct\nVenn Diagram,1\nVenn Diagram,1\nArea Rectangle,0\n".encode()
    )
    assert main(["recommend", "--history", str(h)]) == 0
    out = capsys.readouterr().out
    assert "model: IRT" in out and "next exercise:" in out
