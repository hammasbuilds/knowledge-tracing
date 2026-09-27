# STATUS

**READY-FOR-REVIEW** - the classical study (the headline) is complete on two real public
datasets. The LLM arm is built, tested with a fake client, its 300 jobs are frozen, and it is
queued in `scripts/run_models.sh`; it is a side arm, not the headline.

## Self-score (honest, after a hostile pass)

| Points | Criterion | Score | Reason |
|---:|---|---:|---|
| 15 | Works from a clean clone | 15 | Fresh `git clone` to a temp dir: `uv sync --offline`, `uv run pytest -q` (71 passed), `uv run python demo.py`, `uv run kt recommend --history examples/history.csv`, `ruff check` all succeed with no data present. Tests read no env vars and no data dirs. |
| 20 | Real data, real result | 20 | ASSISTments 2009 original release (525,534 rows) and KDD Cup Algebra 2005 (809,694 rows), both sha256-pinned in `scripts/fetch_data.sh`. Every README number is in `results/*/*.json` / `results/SUMMARY.md`, produced on this machine. Our collapse of the original file matches the publisher's collapsed release row for row. |
| 15 | Finding quality | 14 | Duplicates vs leakage vs horizon separated and each controlled (same test students across variants, same rows across horizons, same rows for the fitted-vs-online leak contrast); paired student-level bootstrap CIs; three split seeds; hyper-parameters tuned on val only; surprising numbers investigated (DKT lower under row split, raw ItemMean). -1: the mastery comparison has no CI and the policy evaluation is observational (stated). |
| 15 | Correctness | 14 | No-peeking flip test on every model at two horizons (it caught a real cross-student leak), LSTM gradient check, EM parameter recovery, reference BKT filter, brute-force AUC and index tests. -1: one-dataset-shaped assumptions (e.g. per-student ordering by order_id / transaction time) are tested on fixtures, not re-derived from the raw timestamps. |
| 10 | Usability | 9 | `kt --help` and per-command help, helpful errors (unknown skill suggests `kt skills`, bad CSV line numbers, missing files point to the fetch script), sensible defaults (committed models). -1: `kt recommend` only uses BKT; DKT is saved but has no CLI front end. |
| 10 | README | 10 | House skeleton, mermaid + blockquote claim, findings table at top, 5 real Input/Output samples, NOT-do section, 7 real problems hit, British spelling, one-line credit to HKUDS/DeepTutor. |
| 10 | Code quality | 9 | ruff clean, typed, numpy-only runtime, small modules. -1: `BKT._em` is a long function (forward, backward and M-step in one loop body). |
| 5 | Honesty | 5 | Every number traceable to a results file; the hypothesis that DKT beats everything was wrong on ASSIST09 and the README says so; limitations listed. |
| **100** | | **96** | |

## Done

- Loaders for ASSISTments 2009 (csv or zip) and KDD Cup 2010 (txt or zip) with three explicit
  row shapes (raw / expanded / collapsed) and counted cleaning (missing skills, exact
  duplicates, conflicting attempts, non-binary labels).
- From-scratch models: ItemMean baseline, BKT (EM, bounded, pooled fallback), PFA (Newton per
  skill), IRT 1PL/2PL (MAP fit, online EAP ability; leaky "fitted" mode for the leak test),
  DKT (numpy LSTM, hand-written backward pass, truncated BPTT, Adam, early stopping).
- Student split 80/15/5 (house policy for <10k units) + row split; horizons k = 1..10;
  bootstrap CIs; train/val/test reported for every model.
- Next-exercise policy (target 0.7, mastery threshold) with a simulator-free evaluation (band
  check, Leopard-style effort/score, coverage-matched against three-in-a-row).
- LLM arm: prompts, cached Ollama client, sampling, scoring, `scripts/run_models.sh` with
  RAM/GPU checks and `--dry-run`.

## Queued for the model run

- `scripts/run_models.sh` - qwen2.5:14b-instruct on `results/assist09/llm_jobs.jsonl`
  (300 ASSIST09 test attempts, 172 students). **300 model calls**, ~15 min at ~3 s/call.
  Writes `results/assist09/llm_answers.jsonl` and `results/assist09/llm_arm.json`; then
  `uv run kt report` adds the LLM table to `results/SUMMARY.md`. The README has no LLM number
  and needs one short paragraph once it exists.

## Known weaknesses remaining

- Algebra05 test split is 29 students (policy 80/15/5 on 574 students); mitigated by two extra
  seeds, not removed.
- DKT reads skills only; an item-aware DKT was not tried, so "IRT beats DKT on ASSIST09" is
  about the standard DKT.
- Mastery comparison: point estimates only, observational (ASSISTments stops assignments at
  three in a row).
- Hyper-parameters for the raw/expanded variants and extra seeds reuse the configuration chosen
  on the collapsed variant's validation set (each model still early-stops on its own data).

## Reproduce every result

```bash
unset VIRTUAL_ENV
uv sync
uv run pytest -q
uv run python demo.py
scripts/fetch_data.sh                    # 3 files, sha256-checked (slow hosts: parallel ranges)
uv run kt study --data data/raw/skill_builder_data_original.csv \
    --crosscheck data/raw/skill_builder_data_corrected_collapsed.csv        # ~31 min
uv run kt study --dataset algebra05 --data data/raw/algebra_2005_2006.zip \
    --variant expanded --variant collapsed                                  # ~73 min
uv run kt llm build --data data/raw/skill_builder_data_original.csv         # 300 frozen jobs
uv run kt report                                                            # results/SUMMARY.md
scripts/run_models.sh --dry-run                                             # model arm plan
uv run kt recommend --history examples/history.csv
```

Runs are deterministic: re-running the ASSIST09 study reproduced every JSON value except
wall-clock seconds.
