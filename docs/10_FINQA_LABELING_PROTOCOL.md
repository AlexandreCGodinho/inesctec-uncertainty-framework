# FinQA Hand-Labeling Protocol

## Objective
Create a blind, per-pattern extraction accuracy benchmark for FinQA numeric answers. This determines which extraction rule (prose vs. bare number) is more faithful to the model's intent, **before** looking at automatic scores.

## Scope
- **30 query_ids** common to all 4 patterns (from T4 audit)
- **4 patterns**: P1 (Governed), P2 (Role Pipeline), P3 (Reflexive), P4 (Multi-Perspective)
- **Blind labeling**: Annotator sees answers labeled "Pattern A/B/C/D" (randomized), NOT P1/P2/P3/P4

## Selection Criteria (already satisfied)
- 30 query_ids present in all 4 patterns' FinQA outputs
- All 30 currently marked ABSTAIN in T4 kit due to mixed categories (P1-P3: `no_verdict`, P4: `approved`)
- Blindness verified: gold answers do not appear in the kit

## Labeling Instructions

### For each row (one query_id):

1. **Read all 4 answers** (Pattern A, B, C, D) — they are shuffled, you don't know which is which pattern
2. **Determine the numeric answer the model MEANT** as its final answer
   - Look for: bare numbers, percentages, dollar amounts, ratios, explicit "Final Answer:" lines
   - Ignore: reasoning text, commentary, judge deliberation (in P4)
3. **Enter in `intended_number`**:
   - The number as a clean string (e.g., `"15.7%"`, `"849 million"`, `"2.8%"`, `"19.9%"`, `"1000000"`)
   - If multiple numbers appear, pick the one that is clearly the FINAL answer
4. **Enter in `abstain`**: `true` or `false`
   - `true` if: no clear final number, ambiguous format, contradictory numbers, "I cannot determine"
   - `false` if: a clear final numeric answer is identifiable
5. **Enter in `abstain_reason`** (if abstain=true):
   - `"no number found"`
   - `"multiple conflicting numbers"`
   - `"answer is prose without clear numeric extraction"`
   - `"answer says cannot determine"`
   - `"format too ambiguous"`

### Important rules:
- **Do not look up the gold answer** — this is blind labeling
- **Do not try to guess which pattern is which** — the order is randomized
- **Be conservative** — if unsure, abstain and note why
- **P4 answers** often have "VERDICT: approved" + bare number — extract the number
- **P1-P3 answers** are prose — find the final numeric statement

## Output
- Completed grid with `intended_number`, `abstain`, `abstain_reason` filled
- Per-pattern extraction accuracy = (correct extractions) / (non-abstained)
- Per-pattern abstention rate

## Sign-off Required
Before labeling begins, **Álvaro + Alexandre must approve** this protocol and the blind grid.

## Files
- `data/finqa_labeling_grid.csv` — main grid for labeling (open in Excel/Sheets)
- `data/finqa_labeling_grid.jsonl` — programmatic version
- This protocol: `docs/10_FINQA_LABELING_PROTOCOL.md`
