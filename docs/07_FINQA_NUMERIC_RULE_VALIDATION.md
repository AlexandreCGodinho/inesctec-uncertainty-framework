# FinQA Numeric-Answer Parsing Rule — Validation for Sign-off


## 1. Rule specification

`src/scoring/finqa_numeric_scorer.py` (`FinqaNumericScorer`)

1. **Extraction** — take the `**Final Answer:**` section if present, else the full
   answer. Select the last numeric token, preferring one carrying a unit word
   (`%`, `million`, `billion`, `thousand`). A trailing bare `1` after `to`/`:` is
   ignored (ratio "38.6 to 1" → 38.6).
2. **Normalization** — strip `$` and thousands separators. `%` → `/100`.
   `million`/`billion`/`thousand` → ×1e6/1e9/1e3 as base.
3. **Scale ambiguity (table units)** — gold `exe_ans` is in raw table units, which
   vary per table (raw / thousands / millions). When a unit word
   (`thousand`/`million`/`billion`) is present, candidates scale **bidirectionally**:
   `{base, base×1e-3, base×1e-6, base×1e+3, base×1e+6}`. This covers both
   directions of mismatch:
   - **Downward** (model says "33.458 million", table in raw units → 33458):
     `33.458 × 1e+3 = 33458` → match.
   - **Upward** (model says "40.264 million", table in raw units → 40264):
     `40.264 × 1e+3 = 40264` → match.
   Plain numbers with no unit word are NOT scaled (avoids false positives such
   as 1,056,000 → 1056).
4. **Comparison** — match if any candidate is within `rtol=0.005` (`math.isclose`,
   `atol=1e-6`) of `exe_ans`. This covers rounding (NKE 81.37 → 0.81372) and
   %-vs-decimal (81.37% → 0.8137).
5. **Yes/No questions** — when `exe_ans` is `"yes"`/`"no"`, exact case-insensitive
   token match on the answer (handles P4 `VERDICT: approved Yes`).

## 2. Validation — 10 known cases

| case | gold `exe_ans` | gold `answer` | model parsed | match | category |
|---|---|---|---|---|---|
| NKE/2009 | 0.81372 | 81% | 81.37% | 1.0 | % + rounding |
| ABC/2005 | 0.9 | 90% | 90.0% | 1.0 | % |
| MRO/2013 | 0.24744 | 24.7% | 24.7% | 1.0 | % + trailing year |
| AAL/2016 | 0.13202 | 13.2% | 13.2% | 1.0 | % rounding |
| TMUS/2017 | 82352.94118 | 82352.94 | 82,352.94 | 1.0 | plain number |
| MRO/2003 | 1056.0 | 1056 | 1,056 | 1.0 | thousand unit |
| AMT/2014 | 33458.0 | 63458* | 33.458M → 33458 | 1.0 | million in table-thousands |
| ETFC/2011 | 4.6 | 4.6 | 38.6 (to 1) | 0.0 | ratio tail (model wrong) |
| AMT/2003 | 0.33333 | 33.3% | 10,000 sqft | 0.0 | model omits the % |
| JKHY/2019 | "no" | no | no | 1.0 | yes/no |

Case → full `query_id` mapping: NKE/2009 → `NKE/2009/page_43.pdf-2`,
ABC/2005 → `ABC/2005/page_49.pdf-4`, MRO/2013 → `MRO/2013/page_19.pdf-4`,
AAL/2016 → `AAL/2016/page_8.pdf-1`, TMUS/2017 → `TMUS/2017/page_29.pdf-2`,
MRO/2003 → `MRO/2003/page_45.pdf-2`, AMT/2014 → `AMT/2014/page_149.pdf-2`,
ETFC/2011 → `ETFC/2011/page_144.pdf-2`, AMT/2003 → `AMT/2003/page_27.pdf-1`,
JKHY/2019 → `JKHY/2019/page_18.pdf-3`.

\* FinQA quirk: human-typed `answer` and program-executed `exe_ans` differ here.
Per Álvaro's ruling the gold is `exe_ans` (33458).

**8/10 match; the 2 mismatches are genuine model errors, not parser errors.**

## 3. Full-run summary (573 records, current telemetry)

| pattern | numeric accuracy |
|---|---|
| 1 | 80/150 = 0.533 |
| 2 | 73/150 = 0.487 |
| 3 | 73/150 = 0.487 |
| 4 | 79/123 = 0.642 |
| **total** | **305/573 = 0.532** |

Note: pattern 4's higher score correlates with the `VERDICT:` wrapper emitting a
clean final number — flagged for the report discussion, not a parser artifact.

## 4. Open items for sign-off

1. Confirm `rtol=0.005` tolerance.
2. Confirm scale-candidate set `{base, base×1e-3, base×1e-6, base×1e+3, base×1e+6}`
   (bidirectional, only when a unit word is present).
3. Confirm yes/no handled as exact token match.
4. Confirm using `exe_ans` (not `answer`) as gold even when they disagree
   (AMT/2014 33458 vs 63458).

## 5. Files

- `src/scoring/finqa_numeric_scorer.py`: scorer
- `tests/test_finqa_numeric_scorer.py`: 27 unit tests (all pass)
- `scripts/adjudicate_finqa_rule.py`: rule adjudication benchmark
- `data/audits/finqa_adjudication_report.json`: audit summary
- `data/audits/finqa_adjudication_details.csv`: per-item audit records
- `data/finqa_labeling/finqa_labeling_grid_adjudicated.csv`: 120 human-adjudicated ground truth labels

## 6. Phase 2 Task 5A Adjudication Benchmark (120 Hand-Labeled Cells)

To formally decide the extraction rule before executing the 900 scoring calls,
both candidate extraction rules were evaluated against the 120 blind hand-labeled
answers in `data/finqa_labeling/finqa_labeling_grid_adjudicated.csv` (30 queries
across 4 patterns, labeled independently by Fedi and Prof. Alexandre with gold
answers hidden):

| Extractor Rule | Overall Matches | Accuracy (%) | Pattern A (P1) | Pattern B (P2) | Pattern C (P4) | Pattern D (P3) |
|---|---|---|---|---|---|---|
| Rule 1 (Legacy Naive Extractor) | 58 / 120 | 48.33% | 15 / 30 (50.0%) | 14 / 30 (46.7%) | 18 / 30 (60.0%) | 11 / 30 (36.7%) |
| Rule 2 (Unified Adjudicated Extractor) | 117 / 120 | **97.50%** | **29 / 30 (96.7%)** | **29 / 30 (96.7%)** | **30 / 30 (100.0%)** | **29 / 30 (96.7%)** |

### Findings and Decision
1. **Rule 2 Selection**: Rule 2 resolves intermediate calculation tokens, masks date and time-horizon numbers, isolates P4 answer spans, and accurately extracts percentage, ratio, and boolean responses, achieving 97.50% fidelity against human consensus.
2. **Sign-off Readiness**: With 97.5% empirical alignment and 0 false extractions on true abstentions in P4, Rule 2 is officially chosen and frozen for the 900 FinQA execution calls in Task 5B.

### Known Limitations and Evaluation Timeline (Reviewer Recommendations)
1. **Chronological Sequence of Events:**
   - Blind hand-labeling of 120 answers independently by Fedi and Prof. Alexandre with official gold answers hidden.
   - Diagnosis of unit scaling from AMT/2014 and AWK/2012, formulating the bidirectional unit scaling rule.
   - Extraction rule specification frozen.
   - Adjudication benchmark executed on the 120-label sample, with gold answers hidden.

2. **Two Known Edge-Case Limitations (3 misses across 120 items):**
   - *Trailing text after conclusion:* In `AAL/2013/page_18.pdf-1` (Pattern B), the model provided 2.8% in its conclusion paragraph but added a trailing bullet point discussing 35.3% of operating expenses. The backward parser captured the trailing 35.3%.
   - *Contradictory numbers within a single answer:* In `AMT/2006/page_113.pdf-3` (Patterns A and D), the answer contained multiple contradictory numbers where a deterministic regex cannot arbitrate an epistemic abstention.

3. **Blinding Nuance Note:**
   - Because Pattern 4 models consistently output structural `VERDICT:` lines and reviewer commentary, the human labeling was strictly blind across P1, P2, and P3, while P4 was structurally recognizable.

