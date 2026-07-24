# Answer Extraction Rule — PopQA (v3)

**Status:** Approved architecture (Álvaro, 23 Jul 2026). Corrections applied per his review.

## Scope

PopQA only. IFEval is scored on the full response (constraint-based). FinQA and LiveBench parse their own structured output. The extraction rule is deterministic and applied identically across all four patterns.

## Architectural Decision

Extraction lives inside `PopqaScorer.score()`, **not** in the orchestrator patterns. `state.final_answer` is never mutated, so `u_L`, `u_F`, `u_P` remain unaffected — same bug class as the Pattern 2 Reviewer suffix that was fixed earlier.

## Rule (`_extract_answer`)

Called once per record inside `PopqaScorer.score()`. Operates on lines of `final_answer`.

### Step 1 — Remove noise lines

Drop any line (case-insensitive) that matches:

- Starts with `VERDICT:`, `CRITIQUE:`, `##`, or is punctuation-only (`---`, `===`, `***`, etc.)
- Starts with conversational fillers: `i hope`, `let me know`, `feel free`, `if you have`, `please`, `thank`, `you're welcome`
- Is empty or whitespace-only

### Step 2 — Extract `FINAL ANSWER:` line

From the remaining lines, find those containing `final answer` (case-insensitive). If one or more exist, take the text **after the colon** on the **last** such line. If there is no text after the colon (answer is on the next line), fall through to Step 3 instead of returning the `FINAL ANSWER:` line verbatim.

### Step 3 — Fallback to last substantive line

If no `FINAL ANSWER:` line was found, return the **last remaining line** that contains at least one letter or digit.

### Step 4 — No substantive content

If after Step 1 no line survives with a letter or digit, return the raw `final_answer` text unchanged. The scorer will apply the official matching rule (see below) to that raw text, which will produce `match=0.0` for noise-only output.

## Official Matching Rule (applied after extraction)

After `_extract_answer` returns the candidate prediction, the `PopqaScorer` applies:

1. **Primary:** `_is_alias_match(ref, prediction)` — gold alias as substring, case-normalized, punctuation stripped. Returns 1.0 or 0.0. This drives `headline_bin`.
2. **Secondary:** `match_jaccard_word`, `match_jaccard_bigram`, `match_acronym` — unchanged from v1.

All four secondary scoring functions (`_jaccard_word_score`, `_jaccard_bigram_score`, `_acronym_score`) also operate on the **extracted** prediction.

## Diagnostic Column: `score_popqa_raw`

The scorer also computes `match` on the **unextracted** (raw) `final_answer` and emits it as `score_popqa_raw` in the scored JSONL. This survives into `per_item_scores.csv` via the merge in `generate_score_table.py`.

Purpose: enable a with/without-extraction comparison. If `score_popqa_raw == match` for every record in a run, one reviewer question ("does extraction change results?") is answered before it is asked.

## Test Cases (10)

All tests call `_extract_answer` as a pure function — no LLM calls.

| # | Input | Expected |
|---|-------|----------|
| 1 | `"FINAL ANSWER: Paris"` | `"Paris"` |
| 2 | `"Analysis...\nFINAL ANSWER: Paris\nFINAL ANSWER: London"` | `"London"` (last wins) |
| 3 | `"I hope this helps.\nParis"` | `"Paris"` (noise removed) |
| 4 | `"VERDICT: correct\nFINAL ANSWER: Einstein"` | `"Einstein"` (verdict removed) |
| 5 | `"I hope this helps.\nLet me know."` | raw text (noise only → scorer gives 0.0) |
| 6 | `"A\nB\nC"` | `"C"` (last substantive) |
| 7 | `"FINAL ANSWER: 42"` | `"42"` (numeric) |
| 8 | `""` | `""` (empty) |
| 9 | `"Just a random sentence"` | `"Just a random sentence"` (no FINAL ANSWER, no noise) |
| 10 | `"## Analysis\nThe capital is Paris."` | `"The capital is Paris."` (## removed) |
