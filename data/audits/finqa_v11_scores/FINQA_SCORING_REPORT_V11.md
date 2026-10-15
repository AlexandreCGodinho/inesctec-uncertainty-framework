# FinQA 900 Automated Scoring Report (Phase 2 Task 5B)

- **Execution Timestamp:** 2026-10-15T15:56:52Z
- **Pinned FinQA Commit:** `0f16e2867befa6840783e58be38c9efb9229d742`
- **Comparison Tolerance:** `rtol=0.005`, `atol=1e-06`
- **Total Evaluated Records:** 900
- **Overall Accurate Matches:** 615 / 900 (68.33%)

## 1. Summary by Architectural Pattern

| Pattern | Description | Total Items | Matches | Accuracy (%) |
|---|---|---|---|---|
| P1 (Governed Workflow) | Pattern_1 | 150 | 87 | **58.00%** |
| P2 (Role-Separated Pipeline) | Pattern_2 | 150 | 93 | **62.00%** |
| P3 (Reflexive Single Agent) | Pattern_3 | 150 | 94 | **62.67%** |
| P4 (Multi-Perspective Deliberation) | Pattern_4 | 450 | 341 | **75.78%** |

## 2. Detailed Per-File Telemetry Breakdown

| Seed Directory | Pattern | Seed | Records | Matches | Abstentions | Accuracy (%) |
|---|---|---|---|---|---|---|
| `report_run_a100` | P1 | 1234 | 150 | 87 | 6 | **58.00%** |
| `report_run_a100` | P2 | 1234 | 150 | 93 | 8 | **62.00%** |
| `report_run_a100` | P3 | 1234 | 150 | 94 | 2 | **62.67%** |
| `report_run_a100` | P4 | 1234 | 150 | 114 | 1 | **76.00%** |
| `report_run_a100_seed2345` | P4 | 2345 | 150 | 114 | 1 | **76.00%** |
| `report_run_a100_seed3456` | P4 | 3456 | 150 | 113 | 1 | **75.33%** |

## 3. Operational Verification
- Historical baseline telemetry files in `data/report_run_a100*/` remain strictly frozen.
- Scored outputs with extraction audit fields are stored in `data/audits/finqa_v11_scores/`.
