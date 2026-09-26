"""Command line: ``kt study``, ``kt inspect``, ``kt recommend``, ``kt llm ...``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .data import COLLAPSED, LOADERS, VARIANTS, DataError, Log
from .models.bkt import BKT
from .policy import recommend

ROOT = Path(__file__).resolve().parents[2]


def _cmd_inspect(a: argparse.Namespace) -> int:
    from .run import adjacent_copy_rate, describe

    for v in a.variant or VARIANTS:
        log = LOADERS[a.dataset](a.data, v)
        print(json.dumps({**describe(log), "adjacent_copies": adjacent_copy_rate(log)}, indent=2))
    return 0


def _cmd_study(a: argparse.Namespace) -> int:
    from .run import run_dataset

    variants = tuple(a.variant) if a.variant else VARIANTS
    if COLLAPSED not in variants:
        raise SystemExit("--variant must include 'collapsed' (the headline variant)")
    run_dataset(
        a.dataset,
        a.data,
        Path(a.out) / a.dataset,
        variants=variants,
        seed=a.seed,
        quick=a.quick,
        n_boot=a.n_boot,
        models_dir=Path(a.models_dir) if a.models_dir else None,
    )
    return 0


def read_history(path: Path) -> list[tuple[str, int]]:
    """CSV with a header containing ``skill`` and ``correct`` (0/1), oldest first."""
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or not {"skill", "correct"} <= set(reader.fieldnames):
            raise DataError(f"{path}: need a header with 'skill' and 'correct' columns")
        out = []
        for i, rec in enumerate(reader, 2):
            if rec["correct"].strip() not in ("0", "1"):
                raise DataError(f"{path} line {i}: correct must be 0 or 1, got {rec['correct']!r}")
            out.append((rec["skill"].strip(), int(rec["correct"])))
    return out


def resolve_skill(query: str, names: list[str]) -> int:
    """Match a skill by exact name, by its id prefix (``311``), or by its label."""
    for i, n in enumerate(names):
        if query == n or query == n.split(":", 1)[0] or query == n.split(":", 1)[-1]:
            return i
    lowered = [i for i, n in enumerate(names) if query.lower() in n.lower()]
    if len(lowered) == 1:
        return lowered[0]
    hint = f"; {len(lowered)} names contain it" if lowered else ""
    raise DataError(f"unknown skill {query!r}{hint}. Run `kt skills` to list them.")


def knowledge_by_skill(model: BKT, names: list[str], history: list[tuple[str, int]]) -> dict:
    """Run BKT over the student's history and return P(correct next) per skill."""
    skill_ids = [resolve_skill(s, names) for s, _ in history]
    rows = len(history)
    log = Log(
        name="history",
        variant=COLLAPSED,
        user=np.zeros(rows + len(names), dtype=np.int32),
        item=np.zeros(rows + len(names), dtype=np.int32),
        skill=np.asarray(skill_ids + list(range(len(names))), dtype=np.int32),
        correct=np.asarray([c for _, c in history] + [0] * len(names), dtype=np.int8),
        attempt=np.arange(rows + len(names), dtype=np.int64),
        skill_names=names,
        item_names=["?"],
        user_names=["student"],
    )
    # one probe row per skill appended after the history; its prediction uses
    # only the real history (probe outcomes come after it and are never read)
    p = model.predict(log)[rows:]
    seen = {i for i in skill_ids}
    return {names[i]: {"p_correct": float(p[i]), "practised": i in seen} for i in range(len(names))}


def _cmd_recommend(a: argparse.Namespace) -> int:
    spec = json.loads(Path(a.model).read_text(encoding="utf-8"))
    model = BKT.from_json(spec)
    names = spec["skill_names"]
    history = read_history(Path(a.history))
    table = knowledge_by_skill(model, names, history)
    candidates = [names[resolve_skill(s, names)] for s in a.candidates] if a.candidates else None
    if candidates is None:
        candidates = [n for n, v in table.items() if v["practised"]] or list(table)
    mastered = {n for n in candidates if table[n]["p_correct"] >= a.mastery}
    pool = {n: table[n]["p_correct"] for n in candidates}
    rec = recommend(pool, a.target, mastered if len(mastered) < len(pool) else None)
    print(f"history: {len(history)} attempts over {len({s for s, _ in history})} skills")
    print(f"{'P(correct)':>10}  skill")
    for n in sorted(pool, key=pool.get):
        flag = "  <- next" if n == rec.skill else ("  (mastered)" if n in mastered else "")
        print(f"{pool[n]:>10.3f}  {n}{flag}")
    print(f"\nnext exercise: {rec.skill} ({rec.reason})")
    return 0


def _cmd_skills(a: argparse.Namespace) -> int:
    spec = json.loads(Path(a.model).read_text(encoding="utf-8"))
    for n in spec["skill_names"]:
        print(n)
    return 0


def _llm_paths(a: argparse.Namespace) -> tuple[Path, Path, Path]:
    out = Path(a.out)
    return out / "llm_jobs.jsonl", out / "llm_answers.jsonl", out / "llm_arm.json"


def _cmd_llm_build(a: argparse.Namespace) -> int:
    from .llm import sample_jobs
    from .run import load_test_predictions

    jobs_path, _, _ = _llm_paths(a)
    test, preds = load_test_predictions(a.dataset, a.data, Path(a.results))
    jobs = sample_jobs(test, preds, a.n, a.seed)
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    with jobs_path.open("w", encoding="utf-8") as fh:
        for j in jobs:
            fh.write(json.dumps(j) + "\n")
    print(
        f"wrote {len(jobs)} jobs from {len({j['student'] for j in jobs})} students to {jobs_path}"
    )
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise DataError(f"{path} does not exist")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _cmd_llm_run(a: argparse.Namespace) -> int:
    from .llm import CachedClient, ollama_call, run_jobs

    jobs_path, answers_path, _ = _llm_paths(a)
    jobs = _read_jsonl(jobs_path)
    if a.limit:
        jobs = jobs[: a.limit]
    cache = Path(a.cache)
    done = sum(
        1 for j in jobs if (cache / (k := _key(a.model, j["prompt"]))[:2] / f"{k}.json").exists()
    )
    if a.dry_run:
        print(f"model {a.model} at {a.host}")
        print(f"jobs: {len(jobs)}  cached: {done}  model calls needed: {len(jobs) - done}")
        return 0
    client = CachedClient(a.model, ollama_call(a.model, a.host), cache)
    answers = run_jobs(jobs, client, progress=True)
    with answers_path.open("w", encoding="utf-8") as fh:
        for ans in answers:
            fh.write(json.dumps(ans) + "\n")
    print(
        f"{len(answers)} answers ({client.misses} new calls, {client.hits} cached)"
        f" -> {answers_path}"
    )
    return 0


def _key(model: str, prompt: str) -> str:
    from .llm import OPTIONS, cache_key

    return cache_key(model, prompt, OPTIONS)


def _cmd_llm_score(a: argparse.Namespace) -> int:
    from .llm import score_jobs
    from .run import write_json

    jobs_path, answers_path, out_path = _llm_paths(a)
    res = score_jobs(_read_jsonl(jobs_path), _read_jsonl(answers_path), a.n_boot)
    res["model"] = a.model
    write_json(out_path, res)
    for k, s in res["scores"].items():
        print(f"{k:>10}  AUC {s['auc']:.3f}  RMSE {s['rmse']:.3f}")
    print(f"parsed {res['parsed_share']:.1%} of replies -> {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from .llm import DEFAULT_HOST, DEFAULT_MODEL

    p = argparse.ArgumentParser(
        prog="kt",
        description="Knowledge tracing from answer logs: BKT, PFA, IRT, DKT, "
        "and a next-exercise policy.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--dataset", choices=sorted(LOADERS), default="assist09")
        sp.add_argument(
            "--data",
            required=True,
            help="path to the dataset file (csv/txt, or the zip it ships in)",
        )

    sp = sub.add_parser("inspect", help="row counts, duplicates and multi-skill share per variant")
    data_args(sp)
    sp.add_argument("--variant", action="append", choices=VARIANTS)
    sp.set_defaults(func=_cmd_inspect)

    sp = sub.add_parser("study", help="run every experiment and write results/<dataset>/*.json")
    data_args(sp)
    sp.add_argument("--variant", action="append", choices=VARIANTS, help="repeatable; default all")
    sp.add_argument("--out", default=str(ROOT / "results"))
    sp.add_argument("--models-dir", default=str(ROOT / "models"), help="where to save BKT/DKT")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--n-boot", type=int, default=1000)
    sp.add_argument("--quick", action="store_true", help="tiny grids, 3 DKT epochs: a smoke run")
    sp.set_defaults(func=_cmd_study)

    sp = sub.add_parser("recommend", help="pick the next skill for a student from their history")
    sp.add_argument("--model", default=str(ROOT / "models" / "assist09_bkt.json"))
    sp.add_argument(
        "--history", required=True, help="CSV with columns skill,correct (oldest first)"
    )
    sp.add_argument("--target", type=float, default=0.7, help="target P(correct), default 0.7")
    sp.add_argument("--mastery", type=float, default=0.95, help="P(correct) treated as mastered")
    sp.add_argument(
        "--candidates", nargs="+", help="skills to choose from (default: practised ones)"
    )
    sp.set_defaults(func=_cmd_recommend)

    sp = sub.add_parser("skills", help="list the skills a saved model knows")
    sp.add_argument("--model", default=str(ROOT / "models" / "assist09_bkt.json"))
    sp.set_defaults(func=_cmd_skills)

    llm = sub.add_parser("llm", help="the LLM arm (build jobs, run them, score them)")
    lsub = llm.add_subparsers(dest="llm_cmd", required=True)

    def llm_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--out", default=str(ROOT / "results" / "assist09"))
        sp.add_argument("--model", default=DEFAULT_MODEL)

    sp = lsub.add_parser(
        "build", help="sample test rows and freeze prompts + classical predictions"
    )
    data_args(sp)
    llm_args(sp)
    sp.add_argument("--results", default=str(ROOT / "results"))
    sp.add_argument("-n", type=int, default=300)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=_cmd_llm_build)

    sp = lsub.add_parser("run", help="ask the model (cached, resumable)")
    llm_args(sp)
    sp.add_argument("--host", default=DEFAULT_HOST)
    sp.add_argument("--cache", default=str(ROOT / "results" / "llm_cache"))
    sp.add_argument("--limit", type=int, default=0)
    sp.add_argument("--dry-run", action="store_true", help="print the job count and exit")
    sp.set_defaults(func=_cmd_llm_run)

    sp = lsub.add_parser("score", help="compare the LLM with the classical models")
    llm_args(sp)
    sp.add_argument("--n-boot", type=int, default=1000)
    sp.set_defaults(func=_cmd_llm_score)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DataError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
