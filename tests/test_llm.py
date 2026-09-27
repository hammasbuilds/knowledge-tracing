import json

import numpy as np
import pytest
from helpers import tiny_log

from kt.llm import (
    CachedClient,
    build_prompt,
    cache_key,
    parse_probability,
    run_jobs,
    sample_jobs,
    score_jobs,
)


def test_prompt_lists_history_in_order_and_truncates():
    text = build_prompt([("Addition", 1), ("Subtraction", 0)], "Addition")
    assert "1. Addition: correct\n2. Subtraction: incorrect" in text
    assert "The next exercise practises: Addition" in text
    long = build_prompt([("A", 1)] * 40, "A", max_history=30)
    assert "(10 earlier exercises omitted)" in long and "30. A: correct" in long
    assert "(no earlier exercises)" in build_prompt([], "A")


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ('{"p_correct": 0.63}', 0.63),
        ('{"p_correct": "0.4"}', 0.4),
        ("p_correct: 63%", 0.63),
        ('{"p_correct": 80}', 0.8),
        ("I think 0.25", 0.25),
        ('{"p_correct": 1.7e3}', None),
        ("between 0.2 and 0.4", None),
        ("no idea", None),
        ('{"p_correct": 250}', None),
    ],
)
def test_parse_probability(text, want):
    got = parse_probability(text)
    assert got == pytest.approx(want) if want is not None else got is None


class FakeCall:
    def __init__(self):
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        # deterministic: more "incorrect" lines -> lower probability
        wrong = prompt.count(": incorrect")
        right = prompt.count(": correct")
        return json.dumps({"p_correct": round((right + 1) / (right + wrong + 2), 3)})


def test_cache_prevents_repeat_calls_and_keys_on_model_and_options(tmp_path):
    fake = FakeCall()
    client = CachedClient("m1", fake, tmp_path)
    a = client.generate("hello")
    b = client.generate("hello")
    assert a == b and fake.calls == 1 and client.hits == 1
    CachedClient("m2", fake, tmp_path).generate("hello")
    assert fake.calls == 2
    assert cache_key("m", "p", {"t": 0}) != cache_key("m", "p", {"t": 1})


def _log():
    rng = np.random.default_rng(0)
    users = np.repeat(np.arange(12), 15)
    return tiny_log(
        rng.integers(0, 3, len(users)).tolist(),
        rng.integers(0, 2, len(users)).tolist(),
        users.tolist(),
    )


def test_sample_jobs_prompt_holds_only_the_past():
    log = _log()
    preds = {"ItemMean": np.full(len(log), 0.5), "BKT": np.linspace(0, 1, len(log))}
    jobs = sample_jobs(log, preds, n=20, seed=1)
    assert len(jobs) == 20 and len({j["row"] for j in jobs}) == 20
    assert jobs == sample_jobs(log, preds, n=20, seed=1)
    for j in jobs:
        assert j["position"] >= 5
        assert j["prompt"].count(": correct") + j["prompt"].count(": incorrect") == j["position"]
        assert j["classical"]["BKT"] == pytest.approx(preds["BKT"][j["row"]])
    assert max(sum(j["student"] == s for j in jobs) for s in {j["student"] for j in jobs}) <= 2


def test_run_and_score_counts_unparsed_replies(tmp_path):
    log = _log()
    preds = {"ItemMean": np.full(len(log), 0.5), "BKT": log.correct * 0.6 + 0.2}
    jobs = sample_jobs(log, preds, n=40, seed=0)
    answers = run_jobs(jobs, CachedClient("fake", FakeCall(), tmp_path))
    answers[0]["p"] = None
    res = score_jobs(jobs, answers, n_boot=50)
    assert res["n"] == 40 and res["parsed_share"] == pytest.approx(39 / 40)
    assert res["scores"]["BKT"]["auc"] == 1.0  # it was given the answers: sanity of wiring
    assert "LLM - BKT" in res["bootstrap"]["auc_diff"]
    with pytest.raises(ValueError, match="no answer"):
        score_jobs(jobs, answers[1:])
