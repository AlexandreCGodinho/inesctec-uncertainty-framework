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

- `src/scoring/finqa_numeric_scorer.py` — scorer
- `tests/test_finqa_numeric_scorer.py` — 13 unit tests (all pass)
- `telemetry/a100/finqa_references.jsonl` — 1147 `query_id → exe_ans` from pinned
  `czyssrs/FinQA` `test.json` (commit `0f16e2867befa6840783e58be38c9efb9229d742`)
