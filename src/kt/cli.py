"""Command line: ``kt study``, ``kt inspect``, ``kt recommend``, ``kt llm ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import COLLAPSED, LOADERS, VARIANTS, DataError
from .llm import LLMError
from .tutor import DEFAULT_MODEL, PRETRAINED, load_model, plan_next, read_history


def _cmd_inspect(a: argparse.Namespace) -> int:
    from .run import adjacent_copy_rate, describe

    for v in a.variant or VARIANTS:
        log = LOADERS[a.dataset](a.data, v)
        print(json.dumps({**describe(log), "adjacent_copies": adjacent_copy_rate(log)}, indent=2))
    return 0


def _cmd_study(a: argparse.Namespace) -> int:
    from .run import run_dataset

    if not Path(a.data).exists():
        raise DataError(f"{a.data} does not exist: scripts/fetch_data.sh downloads the datasets")
    variants = tuple(a.variant) if a.variant else VARIANTS
    if COLLAPSED not in variants:
        raise DataError("--variant must include 'collapsed' (the headline variant)")
    if a.models_dir:
        models_dir: Path | None = Path(a.models_dir)
    else:
        # a --quick smoke run must not overwrite the models `kt recommend` ships with
        models_dir = None if a.quick else PRETRAINED
    run_dataset(
        a.dataset,
        a.data,
        Path(a.out) / a.dataset,
        variants=variants,
        seed=a.seed,
        quick=a.quick,
        n_boot=a.n_boot,
        models_dir=models_dir,
        crosscheck=a.crosscheck,
    )
    return 0


def _cmd_recommend(a: argparse.Namespace) -> int:
    tm = load_model(a.model)
    history = read_history(Path(a.history))
    plan = plan_next(tm, history, a.target, a.mastery, a.candidates)
    print(f"model: {tm.kind.upper()} ({tm.source}); mastery = {tm.mastery_label} >= {a.mastery}")
    print(f"history: {len(history)} attempts over {len({s for s, _ in history})} skills")
    label = tm.mastery_label if tm.kind == "bkt" else ""
    print(f"{'P(correct)':>10}  {label:>8}  skill")
    for st in sorted(plan.table, key=lambda r: r.p_correct):
        if st.skill == plan.next_skill:
            flag = "  <- next"
        elif st.skill in plan.mastered:
            flag = "  (mastered)"
        elif st.skill in plan.unreachable:
            flag = f"  (cannot reach {a.mastery}: ceiling {st.ceiling:.2f})"
        else:
            flag = ""
        known = f"{st.mastery_value:>8.3f}" if tm.kind == "bkt" else " " * 8
        print(f"{st.p_correct:>10.3f}  {known}  {st.skill}{flag}")
    nxt = plan.next_skill if plan.next_skill is not None else "none"
    print()
    print(f"next exercise: {nxt} ({plan.reason})")
    return 0


def _cmd_report(a: argparse.Namespace) -> int:
    from .report import build

    results = Path(a.results)
    if not results.is_dir():
        raise DataError(f"{results} is not a directory")
    text = build(results)
    out = results / "SUMMARY.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"-> {out}")
    return 0


def _cmd_skills(a: argparse.Namespace) -> int:
    for n in load_model(a.model).skill_names:
        print(n)
    return 0


def _llm_paths(a: argparse.Namespace) -> tuple[Path, Path, Path]:
    out = Path(a.out)
    return out / "llm_jobs.jsonl", out / "llm_answers.jsonl", out / "llm_arm.json"


def _cmd_llm_build(a: argparse.Namespace) -> int:
    from .llm import describe_sample, sample_jobs
    from .run import load_test_predictions, write_json

    jobs_path, _, _ = _llm_paths(a)
    test, preds = load_test_predictions(a.dataset, a.data, Path(a.results))
    jobs = sample_jobs(test, preds, a.n, a.seed)
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    with jobs_path.open("w", encoding="utf-8") as fh:
        for j in jobs:
            fh.write(json.dumps(j) + "\n")
    info = describe_sample(jobs, test)
    write_json(jobs_path.parent / "llm_sample.json", info)
    print(
        f"wrote {len(jobs)} jobs from {info['students']} students to {jobs_path}; correct rate "
        f"{info['sample_correct_rate']:.3f} (test set {info['test_correct_rate']:.3f})"
    )
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise DataError(f"{path} does not exist")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _cmd_llm_run(a: argparse.Namespace) -> int:
    from .llm import CachedClient, ollama_call, request_spec, run_jobs

    jobs_path, answers_path, _ = _llm_paths(a)
    jobs = _read_jsonl(jobs_path)
    if a.limit:
        jobs = jobs[: a.limit]
    spec = request_spec(a.model)
    client = CachedClient(spec, ollama_call(spec, a.host), Path(a.cache))
    done = sum(1 for j in jobs if client.path_for(j["prompt"]).exists())
    if a.dry_run:
        print(f"model {a.model} at {a.host}")
        print(f"jobs: {len(jobs)}  cached: {done}  model calls needed: {len(jobs) - done}")
        return 0
    answers = run_jobs(jobs, client, progress=True)
    with answers_path.open("w", encoding="utf-8") as fh:
        for ans in answers:
            fh.write(json.dumps(ans) + "\n")
    print(
        f"{len(answers)} answers ({client.misses} new calls, {client.hits} cached)"
        f" -> {answers_path}"
    )
    return 0


def _cmd_llm_score(a: argparse.Namespace) -> int:
    from .llm import score_jobs
    from .run import write_json

    jobs_path, answers_path, out_path = _llm_paths(a)
    if not answers_path.exists():
        raise DataError(f"{answers_path} does not exist: run `kt llm run` (scripts/run_models.sh)")
    res = score_jobs(_read_jsonl(jobs_path), _read_jsonl(answers_path), a.n_boot)
    res["model"] = a.model
    write_json(out_path, res)
    for k, s in res["scores"].items():
        print(f"{k:>10}  AUC {s['auc']:.3f}  RMSE {s['rmse']:.3f}")
    print(f"parsed {res['parsed_share']:.1%} of replies -> {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from .llm import DEFAULT_HOST
    from .llm import DEFAULT_MODEL as LLM_MODEL

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
    sp.add_argument("--out", default="results")
    sp.add_argument(
        "--models-dir",
        help="where to save the fitted BKT, IRT and DKT (default: the package's pretrained/ "
        "directory, which `kt recommend` reads; --quick saves nothing unless this is given)",
    )
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--n-boot", type=int, default=1000)
    sp.add_argument("--quick", action="store_true", help="tiny grids, 3 DKT epochs: a smoke run")
    sp.add_argument(
        "--crosscheck",
        help="publisher's collapsed release; checked against our own collapse of --data",
    )
    sp.set_defaults(func=_cmd_study)

    sp = sub.add_parser("recommend", help="pick the next skill for a student from their history")
    model_help = (
        "'bkt' (default: per-skill) or 'irt' (skill-level, one shared ability) for the "
        "packaged ASSISTments 2009 models, or a model JSON written by `kt study`"
    )
    sp.add_argument("--model", default=DEFAULT_MODEL, help=model_help)
    sp.add_argument(
        "--history", required=True, help="CSV with columns skill,correct (oldest first)"
    )
    sp.add_argument("--target", type=float, default=0.7, help="target P(correct), default 0.7")
    sp.add_argument(
        "--mastery",
        type=float,
        default=0.95,
        help="mastered when P(known) (BKT) or P(correct) (IRT) reaches this; default 0.95",
    )
    sp.add_argument(
        "--candidates", nargs="+", help="skills to choose from (default: practised ones)"
    )
    sp.set_defaults(func=_cmd_recommend)

    sp = sub.add_parser("report", help="write results/SUMMARY.md from the result JSON files")
    sp.add_argument("--results", default="results")
    sp.set_defaults(func=_cmd_report)

    sp = sub.add_parser("skills", help="list the skills a saved model knows")
    sp.add_argument("--model", default=DEFAULT_MODEL, help=model_help)
    sp.set_defaults(func=_cmd_skills)

    llm = sub.add_parser("llm", help="the LLM arm (build jobs, run them, score them)")
    lsub = llm.add_subparsers(dest="llm_cmd", required=True)

    def llm_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--out", default="results/assist09")
        sp.add_argument("--model", default=LLM_MODEL)

    sp = lsub.add_parser(
        "build", help="sample test rows and freeze prompts + classical predictions"
    )
    data_args(sp)
    llm_args(sp)
    sp.add_argument("--results", default="results")
    sp.add_argument("-n", type=int, default=300)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=_cmd_llm_build)

    sp = lsub.add_parser("run", help="ask the model (cached, resumable)")
    llm_args(sp)
    sp.add_argument("--host", default=DEFAULT_HOST)
    sp.add_argument("--cache", default="results/llm_cache")
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
    except (DataError, LLMError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
