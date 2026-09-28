import json

import numpy as np
import pytest
from helpers import ASSIST_HEADER, TRUE_BKT

from kt.cli import main
from kt.run import load_test_predictions, run_dataset
from kt.synthetic import simulate_bkt


@pytest.fixture(scope="module")
def assist_file(tmp_path_factory):
    """A small ASSISTments-format file: BKT students, every 7th attempt tagged
    with a second skill, every 11th record duplicated."""
    log = simulate_bkt(TRUE_BKT, n_students=160, attempts_per_skill=6, seed=5)
    lines = [ASSIST_HEADER]
    for j in range(len(log)):
        u, s, c = log.user[j], log.skill[j], log.correct[j]
        row = f"{j + 1},1,{u},1,p{s}-{j % 4},1,{c},1,{s},skill {s}\n"
        lines.append(row)
        if j % 7 == 0:
            lines.append(f"{j + 1},1,{u},1,p{s}-{j % 4},1,{c},1,9,shared skill\n")
        if j % 11 == 0 and u % 3 == 0:  # only some students have repeated records
            lines.append(row)
    path = tmp_path_factory.mktemp("data") / "skill_builder_data.csv"
    path.write_text("".join(lines), encoding="latin-1")
    return path


@pytest.fixture(scope="module")
def study(assist_file, tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    models = tmp_path_factory.mktemp("models")
    run_dataset(
        "assist09",
        assist_file,
        out / "assist09",
        quick=True,
        n_boot=30,
        models_dir=models,
        verbose=False,
    )
    return out, models


def test_study_writes_every_result_file(study):
    out, models = study
    names = {p.name for p in (out / "assist09").iterdir()}
    assert {
        "data.json",
        "student_split_collapsed.json",
        "student_split_expanded.json",
        "student_split_raw.json",
        "row_split_collapsed.json",
        "horizons_collapsed.json",
        "policy_collapsed.json",
        "bkt_identifiability.json",
        "test_predictions_collapsed.npz",
    } <= names
    assert (models / "assist09_bkt.json").exists() and (models / "assist09_dkt.npz").exists()


def test_study_numbers_are_sane(study):
    out, _ = study
    res = json.loads((out / "assist09" / "student_split_collapsed.json").read_text())
    for fam, s in res["scores"].items():
        for part in ("train", "val", "test"):
            assert 0.4 < s[part]["auc"] <= 1.0, (fam, part)
    assert res["scores"]["BKT"]["test"]["auc"] > res["scores"]["ItemMean"]["test"]["auc"]
    data = json.loads((out / "assist09" / "data.json").read_text())
    v = data["variants"]
    assert v["raw"]["rows"] > v["expanded"]["rows"] > v["collapsed"]["rows"]
    assert v["collapsed"]["adjacent_copies"]["repeat_rows"] == 0
    raw = json.loads((out / "assist09" / "student_split_raw.json").read_text())
    assert "scores_first_row_of_attempt" in raw
    assert (out / "assist09" / "student_split_raw_seed2.json").exists()
    rr = raw["repeated_records"]
    assert 0 < rr["test_students_excluded"] < raw["split"]["sizes"]["test"]["students"]
    assert rr["concentration"]["test"]["rows_per_attempt"] > 1
    assert (
        json.loads((out / "assist09" / "data.json").read_text())["duplicates"]["repeated_groups"]
        > 0
    )
    # the skill-level IRT that `kt recommend --model irt` runs is scored on every split
    sk = res["irt_skill_level"]
    assert set(sk) == {"train", "val", "test"} and all(0.4 < sk[p]["auc"] <= 1 for p in sk)


def test_saved_predictions_reload_on_the_same_split(study, assist_file):
    out, _ = study
    test, preds = load_test_predictions("assist09", assist_file, out)
    assert set(preds) >= {"BKT", "DKT", "PFA"}
    assert all(len(p) == len(test) for p in preds.values())


def test_cli_llm_build_and_dry_run(study, assist_file, capsys):
    out, _ = study
    rc = main(
        [
            "llm",
            "build",
            "--data",
            str(assist_file),
            "--results",
            str(out),
            "--out",
            str(out / "llm"),
            "-n",
            "25",
        ]
    )
    assert rc == 0
    info = json.loads((out / "llm" / "llm_sample.json").read_text(encoding="utf-8"))
    assert 0 <= info["eligible_test_correct_rate_per_student"] <= 1
    rc = main(["llm", "run", "--out", str(out / "llm"), "--cache", str(out / "cache"), "--dry-run"])
    assert rc == 0
    assert "model calls needed: 25" in capsys.readouterr().out


def test_cli_recommend(study, tmp_path, capsys):
    _, models = study
    hist = tmp_path / "h.csv"
    hist.write_text(
        "skill,correct\nskill 0,1\nskill 0,1\nskill 0,1\nskill 1,0\nskill 1,0\n"
        "skill 2,1\nskill 2,0\n",
        encoding="utf-8",
    )
    rc = main(["recommend", "--model", str(models / "assist09_bkt.json"), "--history", str(hist)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "next exercise:" in text and "<- next" in text


def test_cli_reports_bad_input_cleanly(study, tmp_path, capsys):
    _, models = study
    hist = tmp_path / "bad.csv"
    hist.write_text("skill,correct\nskill 0,yes\n", encoding="utf-8")
    assert (
        main(["recommend", "--model", str(models / "assist09_bkt.json"), "--history", str(hist)])
        == 2
    )
    assert "correct must be 0 or 1" in capsys.readouterr().err
    hist.write_text("skill,correct\nno such skill,1\n", encoding="utf-8")
    assert (
        main(["recommend", "--model", str(models / "assist09_bkt.json"), "--history", str(hist)])
        == 2
    )
    assert "unknown skill" in capsys.readouterr().err
    assert main(["inspect", "--data", str(tmp_path / "missing.csv")]) == 2


def test_study_variant_error_comes_after_the_data_check(assist_file, tmp_path, capsys):
    missing = str(tmp_path / "missing.csv")
    assert main(["study", "--data", missing, "--variant", "raw"]) == 2
    assert "does not exist" in capsys.readouterr().err
    assert main(["study", "--data", str(assist_file), "--variant", "raw"]) == 2
    assert capsys.readouterr().err.startswith("error: --variant must include 'collapsed'")


def test_quick_study_does_not_overwrite_the_packaged_models(assist_file, monkeypatch):
    import kt.run
    from kt.tutor import PRETRAINED

    seen = {}
    monkeypatch.setattr(kt.run, "run_dataset", lambda *a, **k: seen.update(k))
    assert main(["study", "--data", str(assist_file), "--quick"]) == 0
    assert seen["models_dir"] is None
    assert main(["study", "--data", str(assist_file)]) == 0
    assert seen["models_dir"] == PRETRAINED


def test_cli_help_exits_cleanly():
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0


def test_bkt_json_round_trip(study):
    from kt.models.bkt import BKT

    _, models = study
    spec = json.loads((models / "assist09_bkt.json").read_text())
    m = BKT.from_json(spec)
    assert np.all((m.params["guess"] > 0) & (m.params["guess"] <= 0.3))


def test_report_builds_every_table(study, capsys):
    out, _ = study
    assert main(["report", "--results", str(out)]) == 0
    text = (out / "SUMMARY.md").read_text(encoding="utf-8")
    for heading in (
        "split by student",
        "different student splits",
        "three ways of shaping the rows",
        "row-level random split",
        "attempts ahead",
        "predicted 0.70",
        "guess/slip bounds",
        "what the extra rows are",
        "those students removed",
        "on three student splits",
        "Validation AUC next to test AUC",
        "mean (range)",
        "per split seed",
        "as saved for `kt recommend --model irt`",
    ):
        assert heading in text, heading
    assert main(["report", "--results", str(out / "nope")]) == 2
