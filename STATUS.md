# STATUS

**READY-FOR-REVIEW** - the classical study (the headline) is complete on two real public
datasets and three student splits each. The LLM arm is built, tested with a fake client and
queued in `scripts/run_models.sh`; it is a side arm, not the headline.

## Self-score (honest, after a hostile pass)

| Points | Criterion | Score | Reason |
|---:|---|---:|---|
| 15 | Works from a clean clone | 15 | Fresh `git clone` of HEAD to a temp dir: `uv sync --offline`, `uv run pytest -q` (97 passed), `uv run python demo.py`, `uv run kt recommend --history examples/history.csv`, `ruff check` all succeed with no `data/` present. The packaged models are committed in `src/kt/pretrained/` (the only model location). Tests read no env vars and no data dirs. |
| 20 | Real data, real result | 20 | ASSISTments 2009 original release and KDD Cup Algebra 2005, sha256-pinned. Both studies were re-run this round with the committed code; every README number is in `results/*/*.json` / `results/SUMMARY.md`. Our collapse matches the publisher's collapsed release row for row. |
| 15 | Finding quality | 14 | Headline now reported as a three-split mean and range, with the concentration of repeated records per split and scores without those students; paired student-level bootstrap CIs per split; same-rows controls for duplicates, horizons and the leak; prior art (Xiong et al., Wilson et al., Khajah et al. 2016) cited and the work framed as a replication plus four additions. -1: the mastery comparison has no CI and the policy evaluation is observational (stated). |
| 15 | Correctness | 14 | No-peeking flip test on every model at two horizons, LSTM gradient check, EM recovery, reference BKT filter, brute-force AUC; regression tests for the tutor default (8/8 vs 0/6), the parser (1.5 rejected), the CLI error order and the quick-run model guard. -1: per-student ordering assumptions tested on fixtures only. |
| 10 | Usability | 10 | Default model is the per-skill BKT the study scores; `--help` everywhere; every bad input exits 2 with an `error:` line (unknown skill hint names the `--model` in use, target outside (0, 1) rejected, `--variant` checked after the data file); `--quick` never overwrites the packaged models. |
| 10 | README | 10 | House skeleton, prior-work section, three-split headline and concentration subsection, 5 re-captured Input/Output samples, NOT-do section, 10 real problems hit. |
| 10 | Code quality | 9 | ruff clean, typed, numpy-only runtime, one model location, no duplicated computation in `study.py`. -1: `BKT._em` is still one long function. |
| 5 | Honesty | 5 | Seed-0-only headline replaced by the three-split mean with the weaker splits shown; the skill-level IRT the tutor ships is scored and its limitation shown in the README; the raw-trained DKT's small loss vs the collapsed one (0.004-0.008 on every split) is stated rather than called "matching". |
| **100** | | **97** | |

## What changed this round (reviewer defects 1-11)

1. Tutor default is now BKT (per-skill, the model the study scores). The IRT option is
   documented as a skill-level 1PL with one shared ability; the study scores that exact model
   (`irt_skill_level`: test AUC 0.724 vs 0.771 item-level). Regression tests pin the
   8/8-Venn vs 0/6-Area ranking for both.
2. README blockquote, mermaid, findings rows 1-2 and section 1-2 report three-split means and
   ranges; new subsection "The raw headline rests on a handful of students" (per-seed share of
   rows, DKT with and without the owners, the test > val explanation). `kt report` generates
   both tables.
3. Prior work section: Xiong et al. 2016, Wilson et al. 2016, Khajah/Lindsey/Mozer 2016; the
   repo is framed as a replication plus horizons, leak isolation, calibration band and
   coverage-matched mastery.
4. HEAD is self-consistent: `src/kt/pretrained/`, every seed result, `llm_sample.json` and both
   datasets' fitted models are committed; verified by a fresh clone.
5. README test count, Layout (tutor.py, audit.py, synthetic.py, pretrained/), samples 2-5
   re-captured verbatim.
6. `exact_duplicate_rows` renamed `repeated_order_skill_rows` (Algebra05: `repeated_kc_in_step`);
   README says "repeated (order_id, skill) records".
7. `parse_probability`: only `%` or a whole number 2-100 is a percentage; `1.5` is rejected.
8. `--variant` error is `error:` + exit 2 after the data check; the unknown-skill hint names
   `--model`.
9. Algebra05 re-run with the current `run_dataset`; `algebra05_irt.json` now exists.
10. `kt study` writes models only to `src/kt/pretrained/` (`models/` removed; `--quick` saves
    nothing unless `--models-dir` is given). `study.py`'s `keep` alias and double `owners` gone.
11. README states the LLM's 30-attempt window vs full history, that the dry-run needs the
    dataset, and the per-student weighting of the sample (now measured:
    `eligible_test_correct_rate_per_student`).

Beyond the list: `--target` outside (0, 1) is rejected; the seed table has a mean column; the
raw-trained DKT scored once per attempt is reported on all three splits.

## Queued for the model run

- `scripts/run_models.sh` - qwen2.5:14b-instruct on the 300 frozen ASSIST09 test attempts
  (172 students). **300 model calls**, ~15 min at ~3 s/call. Writes
  `results/assist09/llm_answers.jsonl` and `llm_arm.json`; `uv run kt report` then adds the
  LLM table. The README needs one paragraph once it exists.

## Known weaknesses remaining

- Algebra05 test split is 29 students; mitigated by two extra seeds.
- DKT reads skills only; an item-aware DKT was not tried.
- Mastery comparison: point estimates only, observational.
- Hyper-parameters for raw/expanded and extra seeds reuse the seed-0 collapsed selection.
- The tutor cannot run the item-level IRT (a history has no problem ids).

## Reproduce every result

```bash
unset VIRTUAL_ENV
uv sync
uv run pytest -q
uv run python demo.py
scripts/fetch_data.sh                    # 3 files, sha256-checked
uv run kt study --data data/raw/skill_builder_data_original.csv \
    --crosscheck data/raw/skill_builder_data_corrected_collapsed.csv        # ~40 min
uv run kt llm build --data data/raw/skill_builder_data_original.csv         # 300 frozen jobs
uv run kt study --dataset algebra05 --data data/raw/algebra_2005_2006.zip \
    --variant expanded --variant collapsed                                  # ~75 min
uv run kt report                                                            # results/SUMMARY.md
scripts/run_models.sh --dry-run                                             # model arm plan
uv run kt recommend --history examples/history.csv
```
