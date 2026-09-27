"""Turn ``results/<dataset>/*.json`` into the markdown tables the README quotes."""

from __future__ import annotations

import json
from pathlib import Path

FAMILIES = ("ItemMean", "BKT", "PFA", "IRT-1PL", "IRT-2PL", "DKT")


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _ci(ci: list[float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def headline(d: Path) -> list[str]:
    res = _load(d / "student_split_collapsed.json")
    if res is None:
        return []
    sizes = res["split"]["sizes"]
    boot = res["bootstrap_test"]
    out = [
        f"### {res['dataset']}: split by student, one row per attempt",
        "",
        "students train/val/test: "
        + " / ".join(str(sizes[k]["students"]) for k in ("train", "val", "test"))
        + "; rows: "
        + " / ".join(f"{sizes[k]['rows']:,}" for k in ("train", "val", "test")),
        "",
        "| model | selected | train AUC | val AUC | test AUC | test 95% CI "
        "| test RMSE | test ECE |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for fam in FAMILIES:
        s = res["scores"][fam]
        sel = ", ".join(f"{k}={v}" for k, v in res["selection"][fam]["selected"].items())
        out.append(
            f"| {fam} | {sel} | {s['train']['auc']:.3f} | {s['val']['auc']:.3f} | "
            f"{s['test']['auc']:.3f} | {_ci(boot['models'][fam]['auc_ci'])} | "
            f"{s['test']['rmse']:.3f} | {s['test']['ece']:.3f} |"
        )
    out += ["", "| paired test AUC difference | point | 95% CI |", "|---|---:|---|"]
    for k, v in boot["auc_diff"].items():
        out.append(f"| {k} | {v['point']:+.3f} | {_ci(v['ci'])} |")
    return out + [""]


def seeds(d: Path) -> list[str]:
    files = sorted(d.glob("student_split_collapsed*.json"))
    runs = [_load(f) for f in files]
    runs = [r for r in runs if r is not None]
    if len(runs) < 2:
        return []
    out = [
        f"### {runs[0]['dataset']}: test AUC on {len(runs)} different student splits",
        "",
        "| model | " + " | ".join(f"seed {r['split']['seed']}" for r in runs) + " |",
        "|---|" + "---:|" * len(runs),
    ]
    for f in FAMILIES:
        cells = []
        for r in runs:
            ci = r["bootstrap_test"]["models"][f]["auc_ci"]
            cells.append(f"{r['scores'][f]['test']['auc']:.3f} {_ci(ci)}")
        out.append(f"| {f} | " + " | ".join(cells) + " |")
    return out + [""]


def duplicates(d: Path) -> list[str]:
    data = _load(d / "data.json")
    rows = []
    for v in ("raw", "expanded", "collapsed"):
        res = _load(d / f"student_split_{v}.json")
        if res is None or data is None:
            continue
        rows.append((v, data["variants"][v], res))
    if not rows:
        return []
    out = [
        f"### {rows[0][2]['dataset']}: the same students, three ways of shaping the rows",
        "",
        "| rows are | rows | copies of previous row | "
        + " | ".join(f"{f} test AUC" for f in FAMILIES)
        + " |",
        "|---|---:|---:|" + "---:|" * len(FAMILIES),
    ]
    for v, info, res in rows:
        cells = " | ".join(f"{res['scores'][f]['test']['auc']:.3f}" for f in FAMILIES)
        out.append(
            f"| {v} | {info['rows']:,} | {info['adjacent_copies']['repeat_share']:.1%} | {cells} |"
        )
        if "scores_first_row_of_attempt" in res:
            fr = res["scores_first_row_of_attempt"]
            cells = " | ".join(f"{fr[f]['auc']:.3f}" for f in FAMILIES)
            out.append(f"| {v}, scored once per attempt | {fr['DKT']['n']:,} | - | {cells} |")
    out += ["", "| DKT minus model, test AUC | " + " | ".join(v for v, _, _ in rows) + " |"]
    out.append("|---|" + "---|" * len(rows))
    for f in FAMILIES[:-1]:
        cells = []
        for _, _, res in rows:
            dd = res["bootstrap_test"]["auc_diff"][f"DKT - {f}"]
            cells.append(f"{dd['point']:+.3f} {_ci(dd['ci'])}")
        out.append(f"| DKT - {f} | " + " | ".join(cells) + " |")
    return out + [""]


def leakage(d: Path) -> list[str]:
    row = _load(d / "row_split_collapsed.json")
    stu = _load(d / "student_split_collapsed.json")
    if row is None or stu is None:
        return []
    out = [
        f"### {row['dataset']}: row-level random split vs split by student (test AUC)",
        "",
        "| model | by student | by row | inflation | by row, first 5 attempts only |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in row["scores"].items():
        base = name.split(" (")[0]
        a = stu["scores"][base]["test"]["auc"]
        b = s["test"]["auc"]
        early = s.get("test_first_5_attempts", {}).get("auc")
        e = "-" if early is None else f"{early:.3f}"
        out.append(f"| {name} | {a:.3f} | {b:.3f} | {b - a:+.3f} | {e} |")
    return out + [""]


def horizons(d: Path) -> list[str]:
    h = _load(d / "horizons_collapsed.json")
    if h is None:
        return []
    ks = list(h["by_horizon"])
    out = [
        f"### {h['dataset']}: predicting k attempts ahead (same {h['rows']:,} test rows, "
        f"{h['students']} students, test AUC)",
        "",
        "| model | " + " | ".join(f"k={k}" for k in ks) + " |",
        "|---|" + "---:|" * len(ks),
    ]
    for f in FAMILIES:
        cells = " | ".join(f"{h['by_horizon'][k]['scores'][f]['auc']:.3f}" for k in ks)
        out.append(f"| {f} | {cells} |")
    return out + [""]


def policy(d: Path) -> list[str]:
    p = _load(d / "policy_collapsed.json")
    if p is None:
        return []
    out = [
        f"### {p['dataset']}: does 'predicted 0.70' mean 70% correct? (test attempts predicted "
        "in [0.65, 0.75])",
        "",
        "| model | attempts in band | share | mean predicted | observed | 95% CI |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for f in FAMILIES:
        b = p["band_0.7"][f]
        if b["n"] == 0:
            out.append(f"| {f} | 0 | 0 | - | - | - |")
            continue
        out.append(
            f"| {f} | {b['n']:,} | {b['share']:.1%} | {b['mean_predicted']:.3f} | "
            f"{b['observed_rate']:.3f} | {_ci(b['observed_rate_ci'])} |"
        )
    out += [
        "",
        "Mastery at P(correct) >= 0.9: attempts before declaring mastery (effort) and "
        "correct rate after it (score).",
        "",
        "| rule | sequences declared | mean effort | attempts after | correct after |",
        "|---|---:|---:|---:|---:|",
    ]
    for f in FAMILIES:
        t = p["mastery"][f]["thresholds"]["0.9"]
        rate = t["post_declaration_correct_rate"]
        out.append(
            f"| {f} >= 0.9 | {t['declared_share']:.1%} | {t['mean_effort']:.2f} | "
            f"{t['post_declaration_attempts']:,} | {'-' if rate is None else f'{rate:.3f}'} |"
        )
    t = p["mastery"]["BKT"]["three_in_a_row"]
    rate = t["post_declaration_correct_rate"]
    out.append(
        f"| three correct in a row | {t['declared_share']:.1%} | {t['mean_effort']:.2f} | "
        f"{t['post_declaration_attempts']:,} | {'-' if rate is None else f'{rate:.3f}'} |"
    )
    return out + [""]


def identifiability(d: Path) -> list[str]:
    b = _load(d / "bkt_identifiability.json")
    if b is None:
        return []
    out = [
        f"### {b['dataset']}: BKT with and without the guess/slip bounds",
        "",
        "| EM | skills | at guess bound | at slip bound | guess+slip >= 1 | converged | "
        "val AUC | test AUC |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for k in ("bounded", "unbounded"):
        f = b[k]["fit"]
        out.append(
            f"| {k} | {f['skills_fitted']} | {f['at_guess_bound']} | {f['at_slip_bound']} | "
            f"{f['degenerate_g_plus_s_ge_1']} | {f.get('converged')} ({f['iterations']} it) | "
            f"{b[k]['val']['auc']:.3f} | {b[k]['test']['auc']:.3f} |"
        )
    return out + [""]


def llm(d: Path) -> list[str]:
    r = _load(d / "llm_arm.json")
    if r is None:
        return []
    out = [
        f"### LLM arm ({r['model']}, {r['n']} sampled test attempts, "
        f"{r['parsed_share']:.1%} replies parsed)",
        "",
        "| predictor | AUC | 95% CI | RMSE |",
        "|---|---:|---|---:|",
    ]
    for k, s in r["scores"].items():
        out.append(
            f"| {k} | {s['auc']:.3f} | {_ci(r['bootstrap']['models'][k]['auc_ci'])} | "
            f"{s['rmse']:.3f} |"
        )
    return out + [""]


def build(results: Path) -> str:
    lines = ["# Results summary", "", "Generated by `kt report` from the JSON files beside it.", ""]
    for d in sorted(p for p in results.iterdir() if p.is_dir() and p.name != "llm_cache"):
        for section in (
            headline,
            seeds,
            duplicates,
            leakage,
            horizons,
            policy,
            identifiability,
            llm,
        ):
            lines += section(d)
    return "\n".join(lines).rstrip() + "\n"
