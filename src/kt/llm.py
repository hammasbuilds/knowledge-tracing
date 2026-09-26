"""Model arm: can an instruction-tuned LLM read an answer history and predict
the next answer as well as the classical models?

Built and tested with a fake client; the real run is queued in
``scripts/run_models.sh``. Every generation is cached on disk keyed by
(model, prompt hash, options) so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .data import Log
from .metrics import bootstrap, score
from .seq import positions

DEFAULT_MODEL = "qwen2.5:14b-instruct"
DEFAULT_HOST = "http://127.0.0.1:11434"
OPTIONS = {"temperature": 0.0, "num_predict": 64, "seed": 0}

PROMPT = """You are modelling a student in a maths tutoring system.
Below is the student's answer history, oldest first. Each line is one exercise:
the skill it practises and whether the first attempt was correct.

{history}

The next exercise practises: {target}

Estimate the probability that the student answers it correctly on the first attempt.
Reply with JSON only, exactly like {{"p_correct": 0.63}}."""


def _skill_label(name: str) -> str:
    """ASSISTments skill names are stored as ``id:name`` (or ``a:x+b:y``)."""
    return " + ".join(part.split(":", 1)[-1] for part in name.split("+"))


def build_prompt(history: list[tuple[str, int]], target: str, max_history: int = 30) -> str:
    shown = history[-max_history:]
    lines = []
    if len(history) > len(shown):
        lines.append(f"({len(history) - len(shown)} earlier exercises omitted)")
    lines += [f"{i}. {s}: {'correct' if c else 'incorrect'}" for i, (s, c) in enumerate(shown, 1)]
    if not history:
        lines.append("(no earlier exercises)")
    return PROMPT.format(history="\n".join(lines), target=target)


_NUM = re.compile(r"p_correct\"?\s*[:=]\s*\"?([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(%?)")
_BARE = re.compile(r"(?<![0-9.])([01](?:\.[0-9]+)?|\.[0-9]+)(?![0-9.])")


def parse_probability(text: str) -> float | None:
    """Pull a probability out of a model reply; None when there is none.

    Accepts the requested JSON, a percentage (``"p_correct": 63%``), or a lone
    number in [0, 1]. Anything outside [0, 1] is rejected rather than clipped.
    """
    m = _NUM.search(text)
    if m:
        v = float(m.group(1))
        if m.group(2) == "%" or v > 1:
            v = v / 100 if v <= 100 else float("nan")
        return v if 0.0 <= v <= 1.0 else None
    bare = _BARE.findall(text)
    if len(bare) == 1:
        v = float(bare[0])
        return v if 0.0 <= v <= 1.0 else None
    return None


class Client(Protocol):
    model: str

    def generate(self, prompt: str) -> str: ...


def cache_key(model: str, prompt: str, options: dict) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\0")
    h.update(hashlib.sha256(prompt.encode()).hexdigest().encode())
    h.update(b"\0")
    h.update(json.dumps(options, sort_keys=True).encode())
    return h.hexdigest()


@dataclass
class CachedClient:
    """Wraps a raw ``call(prompt) -> text`` with a one-file-per-generation cache."""

    model: str
    call: Callable[[str], str]
    cache_dir: Path
    options: dict = field(default_factory=lambda: dict(OPTIONS))
    hits: int = 0
    misses: int = 0

    def generate(self, prompt: str) -> str:
        key = cache_key(self.model, prompt, self.options)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        text = self.call(prompt)
        self.misses += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"model": self.model, "options": self.options, "prompt": prompt, "response": text}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        tmp.replace(path)
        return text


def ollama_call(
    model: str, host: str = DEFAULT_HOST, timeout: float = 300.0
) -> Callable[[str], str]:
    """A raw call to Ollama's /api/generate (non-streaming)."""

    def call(prompt: str) -> str:
        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": OPTIONS,
                "format": "json",
            }
        ).encode()
        req = urllib.request.Request(
            f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())["response"]

    return call


def sample_jobs(
    test: Log, preds: dict[str, np.ndarray], n: int, seed: int = 0, min_pos: int = 5
) -> list[dict]:
    """Draw ``n`` test rows (at least ``min_pos`` earlier attempts), stratified so
    no student contributes more than ``ceil(n / students)`` rows, and freeze the
    classical models' predictions for exactly those rows."""
    pos = positions(test)
    eligible = np.flatnonzero(pos >= min_pos)
    if len(eligible) == 0:
        raise ValueError("no test rows with enough history")
    rng = np.random.default_rng(seed)
    users = np.unique(test.user[eligible])
    cap = int(np.ceil(n / len(users)))
    picked: list[int] = []
    for u in rng.permutation(users):
        rows = eligible[test.user[eligible] == u]
        picked += rng.choice(rows, size=min(cap, len(rows)), replace=False).tolist()
    picked = sorted(rng.choice(picked, size=min(n, len(picked)), replace=False).tolist())
    bounds = test.user_bounds()
    jobs = []
    for j in picked:
        u = int(test.user[j])
        lo = int(bounds[u])
        hist = [
            (_skill_label(test.skill_names[test.skill[r]]), int(test.correct[r]))
            for r in range(lo, j)
        ]
        target = _skill_label(test.skill_names[test.skill[j]])
        jobs.append(
            {
                "row": int(j),
                "student": test.user_names[u],
                "position": int(pos[j]),
                "y": int(test.correct[j]),
                "prompt": build_prompt(hist, target),
                "classical": {k: float(v[j]) for k, v in preds.items()},
            }
        )
    return jobs


def run_jobs(jobs: list[dict], client: Client, progress: bool = False) -> list[dict]:
    out = []
    for i, job in enumerate(jobs, 1):
        text = client.generate(job["prompt"])
        out.append({"row": job["row"], "response": text, "p": parse_probability(text)})
        if progress and i % 25 == 0:
            print(f"  {i}/{len(jobs)}", flush=True)
    return out


def score_jobs(jobs: list[dict], answers: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """LLM vs classical on the same rows.

    An unparseable reply is replaced by the mean ItemMean prediction on the
    sample (a base-rate guess) and counted in ``parsed_share`` - never dropped,
    since dropping hard cases would flatter the LLM.
    """
    by_row = {a["row"]: a for a in answers}
    missing = [j["row"] for j in jobs if j["row"] not in by_row]
    if missing:
        raise ValueError(f"{len(missing)} jobs have no answer (first: row {missing[0]})")
    y = np.array([j["y"] for j in jobs], dtype=np.float64)
    parsed = np.array([by_row[j["row"]]["p"] is not None for j in jobs])
    fallback = float(np.mean([j["classical"]["ItemMean"] for j in jobs]))
    llm = np.array(
        [by_row[j["row"]]["p"] if ok else fallback for j, ok in zip(jobs, parsed, strict=True)]
    )
    preds = {"LLM": llm}
    for name in jobs[0]["classical"]:
        preds[name] = np.array([j["classical"][name] for j in jobs])
    groups = np.unique([j["student"] for j in jobs], return_inverse=True)[1]
    pairs = [("LLM", k) for k in preds if k != "LLM"]
    return {
        "n": len(jobs),
        "parsed_share": float(parsed.mean()),
        "distinct_llm_values": int(len(np.unique(llm))),
        "scores": {k: score(y, p) for k, p in preds.items()},
        "bootstrap": bootstrap(y, preds, groups, n_boot, seed, pairs),
    }
