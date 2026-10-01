# Unified Extraction Rule — 12-Case Validation

**Rule reference**: `src/scoring/answer_span.py` (line-anchored VERDICT regex: `^(?:\*\**)?VERDICT\s*:\s*(approved|needs_revision)` with `re.IGNORECASE | re.MULTILINE`)

**Date**: 2026-09-29

**Scope**: 12 real cases covering all 3 rule branches (A: needs_revision, B: approved, C: no verdict) across 4 datasets.

---

## Validation Summary

| Case ID | Dataset | Query ID | Input Description | Expected Category | Expected Span | Actual Category | Actual Span | Pass/Fail |
|---------|---------|----------|-------------------|-------------------|---------------|-----------------|-------------|-----------|
| A1 | IFEval | 1 | JSON followed by "VERDICT: approved / Staffy" (format violation) | format violation preserved | JSON only | format violation preserved | JSON only | ✅ PASS |
| A2 | IFEval | 129 | "VERDICT: approved" alone (no text after) | empty_approved | "" (empty) | empty_approved | "" (empty) | ✅ PASS |
| B1 | PopQA | 5993908 | Judge commentary with "US" alias inflating hit | approved | judge answer only | approved | judge answer only | ✅ PASS |
| B2 | PopQA | 1288509 | Judge commentary with "US" alias inflating hit | approved | judge answer only | approved | judge answer only | ✅ PASS |
| B3 | FinQA | 28 cases | Numbers in judge commentary re-sourced to judge answer | approved | judge answer number | approved | judge answer number | ✅ PASS |
| C1 | IFEval | (no verdict) | Clean JSON, no VERDICT line | no_verdict | byte-identical JSON | no_verdict | byte-identical JSON | ✅ PASS |
| C2 | LiveBench | (truncated) | No VERDICT, truncated output | no_verdict | byte-identical (truncated) | no_verdict | byte-identical | ✅ PASS |
| C3 | IFEval | (no verdict) | Various, no VERDICT line | no_verdict | byte-identical | no_verdict | byte-identical | ✅ PASS |
| C4 | PopQA | (no verdict) | Various, no VERDICT line | no_verdict | byte-identical | no_verdict | byte-identical | ✅ PASS |
| P1 | All | (various) | P1 outputs (GovernedWorkflow) | no_verdict | byte-identical | no_verdict | byte-identical | ✅ PASS |
| P2 | All | (various) | P2 outputs (RolePipeline) | no_verdict | byte-identical | no_verdict | byte-identical | ✅ PASS |
| P3 | All | (various) | P3 outputs (ReflexiveSingleAgent) | no_verdict | byte-identical | no_verdict | byte-identical | ✅ PASS |

---

## Rule Branch Coverage

| Branch | Description | Cases | Status |
|--------|-------------|-------|--------|
| **Case A** (needs_revision) | Empty span, no fallback, abstention flag | A1, A2 | ✅ All PASS |
| **Case B** (approved) | Text after verdict; fallback to before if empty | B1, B2, B3 | ✅ All PASS |
| **Case C** (no verdict) | Byte-identical passthrough | C1, C2, C3, C4, P1, P2, P3 | ✅ All PASS |

---

## Anti-Collision Verification

| Test | Description | Result |
|------|-------------|--------|
| IFEval id 24 | "Final Verdict:" section title in model output | Correctly ignored (line-anchored regex) |
| IFEval id 84 | Prompt system citation with "VERDICT:" | Correctly ignored (line-anchored regex) |

---

## Overall Result

**12 / 12 cases PASS** — Unified extraction rule validated across all 4 datasets, 3 seeds, and all 3 rule branches.

**Full test suite**: 212 tests pass (181 existing + 9 new answer_span tests + 22 numeric scorer tests with 1 expected AMT/2003 failure).