# Multi-Benchmark Automated Scoring Report (Phase 2 Task 5C)

- **Execution Date:** 2026-10-20
- **Scope:** Complete accuracy audit and record enrichment for PopQA, IFEval, and LiveBench across all 4 interaction patterns and 3 random seeds.
- **Total Evaluated Records:** 2,700 items (900 per benchmark across 6 execution cells).
- **Baseline Telemetry Integrity:** 100% of original GPU telemetry fields (u_L, u_F, u_C, u_R, u_P, U_surface, U_composite, tokens, latency) preserved in all generated records.
- **Unit Test Suite:** 15 / 15 unit tests passed (100% pass rate across test_score_popqa_v11.py, test_score_ifeval_v11.py, test_score_livebench_v11.py).

---

## 1. Consolidated Accuracy Summary by Pattern

| Benchmark | Metric | Pattern 1 (Governed Workflow) | Pattern 2 (Role-Separated Pipeline) | Pattern 3 (Reflexive Single Agent) | Pattern 4 (Multi-Perspective Deliberation) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **PopQA** | Exact Match (%) | 11.33% (17/150) | 16.67% (25/150) | 14.67% (22/150) | **18.67% ± 0.67%** |
| **IFEval** | Strict Prompt (%) | 2.00% (3/150) | 1.33% (2/150) | 2.00% (3/150) | **1.78% ± 0.38%** |
| **LiveBench** | Reasoning (%) | 69.33% | 73.33% | 80.58% | **78.28% ± 2.33%** |
| *FinQA (Task 5B)* | *Numerical Acc (%)* | *58.00% (87/150)* | *62.00% (93/150)* | *62.67% (94/150)* | ***75.78% ± 0.38%*** |

> **Methodological Note on Sample Asymmetry & Metric Decoupling:**
> Patterns 1 to 3 are evaluated on a single seed (`seed: 1234`, 150 records per pattern), whereas Pattern 4 (Multi-Perspective Deliberation) is evaluated across three independent seeds (`seeds: 1234, 2345, 3456`, 450 total records). Pattern 4 figures are reported as mean ± standard deviation across the three seeds. Accuracy metrics are strictly decoupled from semantic/factual uncertainty ($u_F$) and syntactic uncertainty ($u_L$), measuring ground-truth benchmark task performance.

---

## 2. Detailed Per-Cell Telemetry Breakdown

### PopQA (Exact Match on Wiki Aliases)
| Cell ID | Pattern | Seed | Total Records | Exact Matches | Abstentions | Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `popqa_p1_s1234` | P1 | 1234 | 150 | 17 | 0 | 11.33% |
| `popqa_p2_s1234` | P2 | 1234 | 150 | 25 | 0 | 16.67% |
| `popqa_p3_s1234` | P3 | 1234 | 150 | 22 | 0 | 14.67% |
| `popqa_p4_s1234` | P4 | 1234 | 150 | 28 | 0 | 18.67% |
| `popqa_p4_s2345` | P4 | 2345 | 150 | 29 | 0 | 19.33% |
| `popqa_p4_s3456` | P4 | 3456 | 150 | 27 | 0 | 18.00% |

### IFEval (Strict Instruction-Following)
| Cell ID | Pattern | Seed | Total Records | Strict Passes | Loose Passes | Abstentions | Strict Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ifeval_p1_s1234` | P1 | 1234 | 150 | 3 | 4 | 0 | 2.00% |
| `ifeval_p2_s1234` | P2 | 1234 | 150 | 2 | 2 | 0 | 1.33% |
| `ifeval_p3_s1234` | P3 | 1234 | 150 | 3 | 4 | 0 | 2.00% |
| `ifeval_p4_s1234` | P4 | 1234 | 150 | 3 | 3 | 2 | 2.00% |
| `ifeval_p4_s2345` | P4 | 2345 | 150 | 3 | 4 | 2 | 2.00% |
| `ifeval_p4_s3456` | P4 | 3456 | 150 | 2 | 2 | 4 | 1.33% |

### LiveBench (Reasoning Category Scorer)
| Cell ID | Pattern | Seed | Total Records | Equivalent Correct | Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `livebench_p1_s1234` | P1 | 1234 | 150 | 104.000 | 69.33% |
| `livebench_p2_s1234` | P2 | 1234 | 150 | 110.000 | 73.33% |
| `livebench_p3_s1234` | P3 | 1234 | 150 | 120.875 | 80.58% |
| `livebench_p4_s1234` | P4 | 1234 | 150 | 119.500 | 79.67% |
| `livebench_p4_s2345` | P4 | 2345 | 150 | 119.375 | 79.58% |
| `livebench_p4_s3456` | P4 | 3456 | 150 | 113.375 | 75.58% |

---

## 3. Operational Verification and Experimental Parity
- **Provenance:** Raw responses and telemetry generated on Colab Pro+ GPU A100 runs (`data/report_run_a100*`) using `google/gemma-4-12b-it` in bfloat16.
- **Frozen Baselines:** Original directories (`data/report_run_a100`, `data/report_run_a100_seed2345`, `data/report_run_a100_seed3456`) remain strictly frozen and unaltered.
- **Full Schema Parity:** Enriched JSONL and consolidated CSV files are written exclusively to `data/audits/<dataset>_v11_scores/`, with manifest metadata matching the FinQA PR #63 structure.
- **Unified Span Extraction:** Governed by `extract_answer_span` (Doc 08), isolating terminal responses after `VERDICT: approved`, treating `VERDICT: needs_revision` as operational abstention, and passing through unmoderated responses byte-identically.
