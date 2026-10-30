# Uncertainty-Aware Validation Framework for Agentic Reasoning Patterns

Code, baseline telemetry, deterministic scoring pipelines, and statistical audit scripts accompanying the manuscript *"How Fragile Are Comparisons Between Agentic Patterns? A Validation Protocol with Measured Effect Sizes"*.

## 1. Experiment Overview

The benchmark suite evaluates four orchestration patterns running on a single frozen backbone (`google/gemma-4-12b-it` in `bfloat16` via `vllm` on NVIDIA A100-SXM4-40GB). All model calls across the four patterns use `temperature = 0.7`, explicit random seeds passed to `vllm`, and role-specific `max_tokens` ceilings (`1,400` tokens for drafting/research roles and `1,024` tokens for `reviewer` and `judge`; see `config/experiment.yaml` and `src/llm_client.py`):

- **P1 (`pattern=1`, Governed Workflow)**: `K = 1`, two sequential asymmetric roles (`drafter` followed by `formatter`, 2 LLM calls per item, preceded by a deterministic Python lexical sentence filter when context is non-empty), no revision loop, and no abstention (`src/patterns/p1_governed.py`).
- **P2 (`pattern=2`, Role-Separated Pipeline)**: `K = 1`, three asymmetric roles (`researcher`, `analyst`, `reviewer`); `researcher` extracts key facts at round 0, followed by an `analyst`-`reviewer` loop for up to `pipeline_max_revisions = 2` revision rounds (3 to 7 LLM calls per item) triggered by the reviewer's `VERDICT`, returning the last `analyst` output (`src/patterns/p2_role_pipeline.py`).
- **P3 (`pattern=3`, Reflexive Single Agent)**: `K = 1`, single role (`solo_reflexive`); drafts an initial answer at round 0 and runs self-critique (`VERDICT: approved` vs. `VERDICT: needs_revision`) and revision calls up to `pattern3_max_iterations = 3` (at most 2 critique rounds, 2 to 5 LLM calls per item), with no abstention (`src/patterns/p3_reflexive.py`).
- **P4 (`pattern=4`, Multi-Perspective Deliberation)**: `K = 3` symmetric analysts (`analyst_1`, `analyst_2`, `analyst_3`) at `temperature = 0.7` diversified via deterministic seed offsets `pattern4_agent_seed_offsets = [0, 4444, 7778]` from the run seed `r`, followed by a single `judge` call (at `temperature = 0.7`, seed `r`) that performs both adjudication and synthesis (`VERDICT: approved` followed by the synthesized answer, or `VERDICT: needs_revision` with feedback for up to `max_rounds = 2`, 4 to 8 LLM calls per item, where a final `needs_revision` is recorded as an operational abstention). Evaluated across three independent run seeds: `1234`, `2345`, and `3456` (`src/patterns/p4_deliberation.py`).

Evaluation covers 150 stratified items per benchmark across four benchmarks (**FinQA**, **LiveBench reasoning**, **PopQA**, and **IFEval**), yielding `24 x 150 = 3,600` per-item records consolidated in `data/audits/per_item_scores_v11.csv` (`4 patterns x 4 benchmarks x 150 = 2,400` records at `seed=1234`, plus `2 extra seeds x 4 benchmarks x 150 = 1,200` P4 records at `seeds 2345, 3456`).

## 2. Repository Layout

- `src/`: Core orchestration pattern implementations (`src/patterns/`), LLM client and runners (`src/llm_client.py`, `src/runner.py`, `src/runner_supplementary_p4.py`), uncertainty sensors and composite aggregation (`src/metrics/`), and deterministic benchmark scorers (`src/scoring/`).
- `config/experiment.yaml`: Frozen model, seed, iteration, and per-role token-ceiling configuration.
- `data/report_run_a100/`, `data/report_run_a100_seed2345/`, `data/report_run_a100_seed3456/`: Raw A100 baseline telemetry (`24` `.jsonl` files, 150 records each, plus `3` `manifest.json` provenance files).
- `data/audits/per_item_scores_v11.csv`: Canonical 3,600-row per-item evaluation matrix (`33` columns) joining accuracy, abstention flags, answer-span diagnostic categories, and runtime uncertainty sensors (`u_R`, `u_C`, `u_F`, `u_L`, `U_surface`, `U_composite`).
- `data/audits/*_v11_scores/` and `data/audits/audit_recomputed_uf_v11/`: Per-benchmark enriched scoring audits and answer-span recomputed `u_F` records for P4.
- `data/audits/statistical_results.json` and `data/audits/auroc_results_v11.json`: Machine-readable statistical test outputs and sensor AUROC audit tables.
- `data/finqa_labeling/`: Blinded 120-cell human adjudication grid (`30` FinQA queries x `4` patterns at `seed=1234`) and inter-annotator agreement artefacts.
- `scripts/`: Deterministic scoring, consolidation, and statistical analysis entry points.
- `tests/`: Unit and reproduction test suite.

## 3. Environment Setup

All offline scoring, statistical tests, and table reproduction scripts run on CPU (Linux or macOS) with Python 3.10+. Raw A100 LLM inference requires Linux with CUDA and `vllm>=0.6`.

```bash
python3 -m venv .venv
source .venv/bin/activate
# For offline scoring, statistical reproduction, and tests on CPU/macOS,
# install all dependencies except Linux/CUDA-only vllm:
grep -v '^vllm' requirements.txt | pip install -r /dev/stdin
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Note on deterministic IFEval verification: `requirements.txt` pins `langdetect==1.0.9` and `nltk==3.10.0`, and `src/scoring/ifeval_scorer.py` invokes `DetectorFactory.seed = 0` prior to constraint evaluation to eliminate stochastic language-detection drift.

Run the automated test suite:

```bash
.venv/bin/pytest -q tests/
```

## 4. Reproducing Empirical Analyses

Design specifications for the orchestration factor space, runtime uncertainty sensors, and benchmark configurations reside in `src/patterns/`, `src/metrics/`, and `config/experiment.yaml`. Every empirical analysis, paired statistical test, and diagnostic breakdown reported in the manuscript can be regenerated or verified directly from the tracked artefacts using the commands below.

| Empirical Analysis | Command / Script | Output / Verification Artefact |
|---|---|---|
| **FinQA scoring validation**: Blind human adjudication grid (`120` cells: `30` queries x `4` patterns) and alternative numeric extraction rules at `seed=1234` | `.venv/bin/python scripts/adjudicate_finqa_rule.py`, `.venv/bin/python scripts/compute_agreement.py`, and `.venv/bin/python scripts/analysis_auroc.py --input data/audits/per_item_scores_v11.csv --no-plots --save-audit-json data/audits/auroc_results_v11.json` | `data/audits/finqa_adjudication_report.json`, `data/finqa_labeling/agreement_details.csv`, and `finqa_alternative_parsers` in `data/audits/auroc_results_v11.json` |
| **IFEval scaffolding sensitivity**: P4 answer-span extraction vs. full-output evaluation across all 3 seeds | `.venv/bin/python scripts/score_ifeval_v11.py` | `data/audits/MULTIBENCHMARK_ACCURACY_REPORT_V11.md` and `data/audits/ifeval_v11_scores/ifeval_audit_all_900.csv` |
| **IFEval language-detection determinism audit**: Seeded (`DetectorFactory.seed = 0`) vs. unseeded `langdetect` constraint verification | Compare `data/audits/ifeval_official_harness_per_item.csv` against `data/audits/ifeval_official_harness_per_item_unseeded.csv` | `data/audits/ifeval_comparison_report.csv` |
| **Primary benchmark accuracy and vs-baseline paired tests**: Per-benchmark accuracy (`seed=1234` for P1 to P3; 3-seed mean and SD for P4), paired McNemar tests (FinQA, PopQA, IFEval), and paired Wilcoxon signed-rank tests (LiveBench) | `.venv/bin/python scripts/analysis_statistics.py --input data/audits/per_item_scores_v11.csv --output-dir data/audits` | `data/audits/MULTIBENCHMARK_ACCURACY_REPORT_V11.md`, `data/audits/finqa_v11_scores/FINQA_SCORING_REPORT_V11.md`, and `mcnemar`, `livebench_wilcoxon_primary`, `p4_between_seed_variance` in `data/audits/statistical_results.json` |
| **Secondary benchmark metrics**: PopQA naive substring and answer-slot matching; IFEval loose prompt-level, instruction-level, and P4 full-output accuracy | `.venv/bin/python scripts/score_popqa_v11.py` and `.venv/bin/python scripts/score_ifeval_v11.py` | `data/audits/MULTIBENCHMARK_ACCURACY_REPORT_V11.md`, `data/audits/popqa_v11_scores/popqa_audit_all_900.csv`, and `data/audits/ifeval_v11_scores/ifeval_audit_all_900.csv` |
| **Token consumption and wall-clock latency**: Per-record prompt/completion token totals (`tokens_prompt + tokens_completion`) and runtime (`latency_ms`) | Inspect `tokens_prompt`, `tokens_completion`, and `latency_ms` in `data/audits/per_item_scores_v11.csv` | `data/audits/per_item_scores_v11.csv` |
| **All-pairs orchestration comparisons**: Six-pair comparisons per benchmark at `seed=1234` with within-benchmark step-down Holm adjustment | `.venv/bin/python scripts/analysis_statistics.py --input data/audits/per_item_scores_v11.csv --output-dir data/audits` | `holm_stepdown_seed1234` in `data/audits/statistical_results.json` |
| **Answer-span factual-consistency (`u_F`) coverage**: P4 `u_F` defined rates, undefined-value breakdown (zero claims, judge parse errors, abstentions, empty spans), and raw-vs-span claim counts | `.venv/bin/python scripts/analysis_auroc.py --input data/audits/per_item_scores_v11.csv --no-plots --save-audit-json data/audits/auroc_results_v11.json` | `uf_coverage_p4` in `data/audits/auroc_results_v11.json` and `data/audits/audit_recomputed_uf_v11/recompute_uf_summary.json` |
| **Individual runtime sensor AUROC**: Error-detection AUROC for `u_R`, `u_C`, and `u_F`, LiveBench truncation sensitivity check, and P2 IFEval `u_R` top-20% review flagging | `.venv/bin/python scripts/analysis_auroc.py --input data/audits/per_item_scores_v11.csv --no-plots --save-audit-json data/audits/auroc_results_v11.json` | `sensor_auroc.pooled_sensor_auroc`, `sensor_auroc.livebench_uR_truncation_check`, `sensor_auroc.p2_ifeval_uR_flagging` in `data/audits/auroc_results_v11.json` |
| **FinQA answer-space agreement**: Cross-seed (P4) and cross-pattern (P1 to P3 at `seed=1234`) numerical agreement under `dropna=False` and `dropna=True` | `.venv/bin/python scripts/analysis_auroc.py --input data/audits/per_item_scores_v11.csv --no-plots --save-audit-json data/audits/auroc_results_v11.json` | `finqa_agreement` (`dropna_false` and `dropna_true`) in `data/audits/auroc_results_v11.json` |
| **Pre-registered silent-failure test**: Evaluating P4 `u_C` on P1 zero-surface-uncertainty errors (`U_surface = 0`), covering item-level flagging across all queries (Setup A) and P4 error detection within silent-failure items (Setup B) | `.venv/bin/python scripts/analysis_auroc.py --input data/audits/per_item_scores_v11.csv --no-plots --save-audit-json data/audits/auroc_results_v11.json` | `silent_failure_auroc` in `data/audits/auroc_results_v11.json` |
| **Multi-seed P4 sensor dispersion and global FDR**: Per-seed and pooled AUROC dispersion (`u_R`, `u_C`, `u_F`, `U_composite`) across seeds `1234`, `2345`, and `3456`, plus 32-test Benjamini-Hochberg FDR across active `(sensor, pattern, benchmark)` pairs | `.venv/bin/python scripts/analysis_auroc.py --input data/audits/per_item_scores_v11.csv --no-plots --save-audit-json data/audits/auroc_results_v11.json` | `sensor_auroc.p4_seed_dispersion` and `sensor_auroc.fdr_32_active_tests` in `data/audits/auroc_results_v11.json` |
| **Per-cell P4 `u_F` and composite `U(x)` distributions**: Coverage breakdown and descriptive statistics (`mean`, `SD`, `median`, `min..max`) across all `12` P4 `(benchmark, seed)` cells and subtotals | `.venv/bin/python scripts/generate_p4_appendix_tables.py` | Console Markdown tables and `data/audits/audit_recomputed_uf_v11/recompute_uf_summary.json` |
| **P4 verdict-line and answer-span categories**: Breakdown of judge verdict parsing categories (`approved`, `empty_approved`, `no_verdict`, `needs_revision`, `approved_fallback`) | Inspect `span_category` for `pattern == 4` in `data/audits/per_item_scores_v11.csv` | `data/audits/per_item_scores_v11.csv` |

## 5. Regenerating Per-Item Scores from Raw A100 Telemetry

Because the raw A100 outputs (`data/report_run_a100*/`) and canonical prompt mappings (`data/ifeval_mapping_150_prompts.jsonl`, `data/popqa_mapping_150_prompts.jsonl`, `data/finqa_references_cache.json`) are included in the repository, reviewers can re-run the deterministic scorers and rebuild `data/audits/per_item_scores_v11.csv` from scratch:

```bash
.venv/bin/python scripts/score_finqa_v11.py
.venv/bin/python scripts/score_ifeval_v11.py
.venv/bin/python scripts/score_livebench_v11.py
.venv/bin/python scripts/score_popqa_v11.py
.venv/bin/python scripts/build_per_item_scores_v11.py
```

Note that `score_livebench_v11.py` fetches the pinned `livebench/reasoning` reference split (`6fc6498a5dfba553f69f4413feabade1f1a2d384`) from Hugging Face Hub on first execution if not already cached locally.
