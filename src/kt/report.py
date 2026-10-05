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
    sk = res.get("irt_skill_level")
    if sk:
        out += [
            "",
            "IRT-1PL as saved for `kt recommend --model irt` (item offsets dropped, every item "
            f"at its skill's difficulty): train / val / test AUC {sk['train']['auc']:.3f} / "
            f"{sk['val']['auc']:.3f} / {sk['test']['auc']:.3f}, test RMSE "
            f"{sk['test']['rmse']:.3f}, test ECE {sk['test']['ece']:.3f}.",
        ]
    out += ["", "| paired test AUC difference | point | 95% CI |", "|---|---:|---|"]
    for k, v in boot["auc_diff"].items():
        out.append(f"| {k} | {v['point']:+.3f} | {_ci(v['ci'])} |")
    return out + [""]


def _mean_range(xs: list[float], signed: bool = False) -> str:
    f = "+.3f" if signed else ".3f"
    return f"{sum(xs) / len(xs):{f}} ({min(xs):{f}} to {max(xs):{f}})"


def seeds(d: Path) -> list[str]:
    files = sorted(d.glob("student_split_collapsed*.json"))
    runs = [_load(f) for f in files]
    runs = [r for r in runs if r is not None]
    if len(runs) < 2:
        return []
    out = [
        f"### {runs[0]['dataset']}: test AUC on {len(runs)} different student splits",
        "",
        "| model | " + " | ".join(f"seed {r['split']['seed']}" for r in runs) + " | mean (range) |",
        "|---|" + "---:|" * (len(runs) + 1),
    ]
    for f in FAMILIES:
        cells = []
        for r in runs:
            ci = r["bootstrap_test"]["models"][f]["auc_ci"]
            cells.append(f"{r['scores'][f]['test']['auc']:.3f} {_ci(ci)}")
        aucs = [r["scores"][f]["test"]["auc"] for r in runs]
        out.append(f"| {f} | " + " | ".join(cells) + f" | {_mean_range(aucs)} |")
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
    out += ["", "Validation AUC next to test AUC (the same models, the val students):", ""]
    out += [
        "| rows are | " + " | ".join(f"{f} val / test" for f in ("BKT", "IRT-1PL", "DKT")) + " |"
    ]
    out.append("|---|---|---|---|")
    for v, _, res in rows:
        cells = " | ".join(
            f"{res['scores'][f]['val']['auc']:.3f} / {res['scores'][f]['test']['auc']:.3f}"
            for f in ("BKT", "IRT-1PL", "DKT")
        )
        out.append(f"| {v} | {cells} |")
    return out + [""] + duplicate_detail(d, [v for v, _, _ in rows])


def duplicate_detail(d: Path, variants: list[str]) -> list[str]:
    """Where the repeated rows come from, how concentrated they are, and whether the
    DKT gap holds on every split seed."""
    data = _load(d / "data.json") or {}
    out: list[str] = []
    dup = data.get("duplicates")
    if dup:
        c = dup["copies_per_repeated_group"]
        out += [
            f"### {data['dataset']}: what the extra rows are",
            "",
            "| | rows |",
            "|---|---:|",
            f"| attempts | {dup['attempts']:,} |",
            f"| extra rows from multi-skill tagging (one row per skill) | "
            f"{dup['multi_skill_extra_rows']:,} |",
            f"| extra rows from repeated records ({dup['repeated_attempts']:,} attempts, "
            f"copied median {c['median']:.0f}x, max {c['max']}x) | "
            f"{dup['repeated_record_extra_rows']:,} |",
            f"| students owning any repeated record | {dup['students_owning_repeated_records']} |",
            f"| share of repeated rows owned by the top 10 of them | "
            f"{dup['share_of_repeated_rows_owned_by_top_10_students']:.1%} |",
            "",
        ]
    audit = data.get("file_audit")
    if audit:
        cols = ", ".join(list(audit["columns_differing_in_copies"])[:6]) or "none"
        out += [
            f"In file order (before any sorting) {audit['file_order_repeat_share']:.1%} of rows "
            f"repeat the previous row's order_id. Of {audit['repeated_record_copies']:,} "
            f"repeated-record copies, {audit['byte_identical_copies']:,} are identical to the "
            f"first; columns that differ: {cols}.",
            "",
        ]
    raw = _load(d / "student_split_raw.json")
    if raw and "repeated_records" in raw:
        rr = raw["repeated_records"]
        conc = rr["concentration"]
        out += [
            "| split | rows | attempts | rows per attempt | students | with repeated records "
            "| their share of rows |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for part in ("train", "val", "test"):
            x = conc[part]
            out.append(
                f"| {part} | {x['rows']:,} | {x['attempts']:,} | {x['rows_per_attempt']:.2f} | "
                f"{x['students']} | {x['students_with_repeated_records']} | "
                f"{x['share_of_rows']:.1%} |"
            )
        w, wf = rr["scores_without_those_students"], rr["scores_without_those_students_first_row"]
        out += [
            "",
            f"Raw test AUC with the {rr['test_students_excluded']} test students who own repeated "
            "records removed:",
            "",
            "| model | all raw rows | raw rows, those students removed | "
            "once per attempt, those students removed |",
            "|---|---:|---:|---:|",
        ]
        for f in FAMILIES:
            out.append(
                f"| {f} | {raw['scores'][f]['test']['auc']:.3f} | {w[f]['auc']:.3f} | "
                f"{wf[f]['auc']:.3f} |"
            )
        out.append("")
    seeds = []
    for v in variants:
        runs = [_load(d / f"student_split_{v}{sfx}.json") for sfx in ("", "_seed1", "_seed2")]
        if all(runs):
            seeds.append((v, runs))
    if seeds:
        out += [
            "DKT test AUC, and DKT - BKT and DKT - IRT-1PL, on three student splits "
            "(mean and range over the seeds):",
            "",
            "| rows are | seed 0 | seed 1 | seed 2 | mean (range) |",
            "|---|---|---|---|---|",
        ]
        for v, runs in seeds:
            aucs = [r["scores"]["DKT"]["test"]["auc"] for r in runs]
            cells = [f"{a:.3f}" for a in aucs]
            out.append(f"| {v}, DKT | " + " | ".join(cells) + f" | {_mean_range(aucs)} |")
            if all("scores_first_row_of_attempt" in r for r in runs):
                once = [r["scores_first_row_of_attempt"]["DKT"]["auc"] for r in runs]
                cells = [f"{a:.3f}" for a in once]
                out.append(
                    f"| {v}, DKT scored once per attempt | "
                    + " | ".join(cells)
                    + f" | {_mean_range(once)} |"
                )
            for other in ("BKT", "IRT-1PL"):
                cells, pts = [], []
                for r in runs:
                    dd = r["bootstrap_test"]["auc_diff"][f"DKT - {other}"]
                    cells.append(f"{dd['point']:+.3f} {_ci(dd['ci'])}")
                    pts.append(dd["point"])
                out.append(
                    f"| {v}, DKT - {other} | "
                    + " | ".join(cells)
                    + f" | {_mean_range(pts, signed=True)} |"
                )
        out.append("")
        out += concentration_by_seed(d)
    return out


def concentration_by_seed(d: Path) -> list[str]:
    """For each split seed of the raw variant: how much of val and test the students
    with repeated records own, and DKT's test AUC with and without them."""
    runs = [_load(d / f"student_split_raw{sfx}.json") for sfx in ("", "_seed1", "_seed2")]
    runs = [r for r in runs if r and "repeated_records" in r]
    if not runs:
        return []
    out = [
        "Raw rows, per split seed: the students who own repeated records, and DKT with and "
        "without them:",
        "",
        "| seed | test students with repeated records | their share of val rows | "
        "their share of test rows | DKT val AUC | DKT test AUC | DKT test AUC without them "
        "| DKT - BKT without them |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cols: dict[str, list[float]] = {k: [] for k in ("val_share", "test_share", "v", "t", "w", "g")}
    for r in runs:
        rr = r["repeated_records"]
        conc, w = rr["concentration"], rr["scores_without_those_students"]
        row = {
            "val_share": conc["val"]["share_of_rows"],
            "test_share": conc["test"]["share_of_rows"],
            "v": r["scores"]["DKT"]["val"]["auc"],
            "t": r["scores"]["DKT"]["test"]["auc"],
            "w": w["DKT"]["auc"],
            "g": w["DKT"]["auc"] - w["BKT"]["auc"],
        }
        for k, x in row.items():
            cols[k].append(x)
        out.append(
            f"| {r['split']['seed']} | {rr['test_students_excluded']} of "
            f"{conc['test']['students']} | {row['val_share']:.1%} | {row['test_share']:.1%} | "
            f"{row['v']:.3f} | {row['t']:.3f} | {row['w']:.3f} | {row['g']:+.3f} |"
        )
    if len(runs) > 1:
        mean = {k: sum(v) / len(v) for k, v in cols.items()}
        out.append(
            f"| mean | - | {mean['val_share']:.1%} | {mean['test_share']:.1%} | {mean['v']:.3f} "
            f"| {mean['t']:.3f} | {mean['w']:.3f} | {mean['g']:+.3f} |"
        )
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
    streak_share = t["declared_share"]
    rate = t["post_declaration_correct_rate"]
    out.append(
        f"| three correct in a row | {t['declared_share']:.1%} | {t['mean_effort']:.2f} | "
        f"{t['post_declaration_attempts']:,} | {'-' if rate is None else f'{rate:.3f}'} |"
    )
    out += [
        "",
        f"Each model thresholded to sign off the same {streak_share:.1%} of (student, skill) "
        "sequences as three-in-a-row:",
        "",
        "| model | threshold | mean effort | attempts after | correct after |",
        "|---|---:|---:|---:|---:|",
    ]
    for f in FAMILIES:
        m = p["mastery"][f].get("matched_to_three_in_a_row")
        if m is None:
            continue
        rate = m["post_declaration_correct_rate"]
        out.append(
            f"| {f} | {m['threshold']:.3f} | {m['mean_effort']:.2f} | "
            f"{m['post_declaration_attempts']:,} | {'-' if rate is None else f'{rate:.3f}'} |"
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


def crosscheck(d: Path) -> list[str]:
    data = _load(d / "data.json") or {}
    c = data.get("crosscheck")
    if not c:
        return []
    keys = [k for k in c if k.startswith("same_") or k.startswith("rows_with")]
    return [
        f"### {data['dataset']}: our collapse vs the publisher's {c['official_file']}",
        "",
        "| check | result |",
        "|---|---|",
        *[f"| {k} | {c[k]} |" for k in keys],
        "",
    ]


def llm_sample(d: Path) -> list[str]:
    s = _load(d / "llm_sample.json")
    if s is None:
        return []
    out = [
        f"### LLM arm sample ({s['n']} test attempts, {s['students']} students): correct rate "
        f"{s['sample_correct_rate']:.3f} vs {s['test_correct_rate']:.3f} on the whole test set",
        "",
    ]
    if "eligible_test_correct_rate_per_student" in s:
        out += [
            "Test rows eligible for the sample (enough history): correct rate "
            f"{s['eligible_test_correct_rate']:.3f} per attempt, "
            f"{s['eligible_test_correct_rate_per_student']:.3f} averaged per student (the "
            "sample takes at most a few rows per student, so it weights students equally).",
            "",
        ]
    out += [
        "| classical model on the sample | AUC | 95% CI | RMSE |",
        "|---|---:|---|---:|",
    ]
    for k, sc in s["classical_on_sample"].items():
        ci = s["bootstrap"]["models"][k]["auc_ci"]
        out.append(f"| {k} | {sc['auc']:.3f} | {_ci(ci)} | {sc['rmse']:.3f} |")
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
            crosscheck,
            leakage,
            horizons,
            policy,
            identifiability,
            llm_sample,
            llm,
        ):
            lines += section(d)
    return "\n".join(lines).rstrip() + "\n"
