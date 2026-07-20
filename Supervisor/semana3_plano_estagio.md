# Sprint Plan: Week 3
**INESC TEC Summer Research Internship 2026**

---

## 1. Objective: Accountability
Prove that the uncertainty metric $U(x)$ reliably predicts system failure. All analytical execution must finish by Monday, July 27.

**Reading Requirement:** Read `03_EVALUATION_METRICS_SPECIFICATION.md` (Doc 03) before starting. You must implement the instructions added to Section 6 and the update to Section 2.4. Terminology must align perfectly with Doc 03. Use the exact variable names and use the term *pattern*, never *architecture*.

**Prompt Freeze:** prompts are frozen as of commit <hash>. No prompt edits during Week 3. Any change invalidates the run and must be escalated.

**Methodology Warning (Prompt Tuning & Fairness):** 
1. Prompt optimization was strictly limited to the Pilot Phase.
2. Because Pattern 2 structurally uses more compute than Pattern 1 (see Doc 02, Sec 4.4), we must prove the comparison is fair regarding human effort. 
3. You must explicitly document in the Results Report that the time and effort spent tuning prompts was equalized across both patterns.

---

## 2. Assignments

### Block A: Telemetry, Metrics & Cost (Fedi Hassine)

*   **Task A1: Data Extraction & Data Join** 
    *   The runner records the sampled query IDs for all four datasets.
    *   Add dataset as an explicit field in every JSONL record; do not rely on the filename.
    *   Parse the raw JSONL telemetry files.
    *   Perform the data join of $U(x)$ with the Correctness score using the primary key `(query_id, pattern, seed)`.
    *   Record `termination_reason` (consensus vs. cap exhausted) as an explanatory covariate.
*   **Task A2: Risk-Coverage & AUROC Analysis** 
    *   Calculate AUROC to evaluate $U(x)$ as an incorrect answer detector.
    *   Generate Risk-Coverage graphs: sort queries by $U(x)$, progressively abstain from the most uncertain ones, and plot the accuracy increase.
    *   *Sub-task:* Report the failure detector with and without the $u_R$ component. Extract refinement loops as a binary signal (`refinement_triggered: Yes/No`).
*   **Task A3: Pareto Frontier (Cost Normalization)** 
    *   Plot the Pareto Frontier measuring *Accuracy vs. Tokens* and *Accuracy vs. Latency*.
*   **Task A4: Statistical Significance & Variance**
    *   Use Pattern 1 as the simplest-pattern baseline for the per-item score tables (do not use the ambiguous term "single-agent").
    *   Calculate confidence intervals for Patterns 1-3 dividing the tests: use McNemar's test for binary metrics (PopQA, FinQA, IFEval) and paired bootstrapping for numeric metrics (LiveBench).
    *   Pattern 4: paired bootstrap over items within each seed; between-seed variance reported separately (Doc 03, Section 6, item 6).
    *   Apply the **Benjamini-Hochberg FDR correction** for multiple comparisons across the pattern x dataset test family.

### Block B: Ground Truth Evaluation & Protocols (Pedro Marques & Fedi Hassine)

*   **Task B1: Official Evaluation Engine**
    *   Use the official evaluation harnesses provided by the dataset authors exclusively.
    *   **Pedro:** Run FinQA and LiveBench. Pin the release of LiveBench used.
        *   *LiveBench requirement:* Replace the silent-skip guard with a **fail-fast rule** (if a dataset returns zero records, the run must abort).
        *   *Manifest requirement:* Generate an end-of-run manifest containing expected vs. actual record counts per (dataset x pattern). Save it next to the JSONL files and assert it with a test.
        *   *Process rule:* Submit these LiveBench fixes via a dedicated Pull Request after the infrastructure revert PR is merged.
    *   **Fedi (Cross-Block Task):** Run IFEval and PopQA.
*   **Task B2: Per-Item Score Table**
    *   Produce a per-item score table storing the official metric value(s) of each record exactly as output by the harness (IFEval: all four variants; LiveBench: per-category scores; FinQA: execution and program accuracy; PopQA: alias match). Do not collapse to 0/1 at generation time.
    *   Add one binary headline column per dataset for paired tests: PopQA = alias match; FinQA = execution accuracy; IFEval = prompt-level strict. LiveBench has no natural binary: paired comparisons use the paired bootstrap on the numeric score.
    *   "Overall Accuracy (%)" remains forbidden (Doc 03, Section 6, item 3).
    *   Run Pattern 4 with 3 distinct run seeds (see key definition in Doc 03, Section 6, item 1).
*   **Task B3: Anonymous Qualitative Coding**
    *   Use an **Anonymous Coding** protocol for the qualitative analysis of failures.
    *   Sampling rule, fixed before reading any output: all incorrect items plus 15 randomly drawn correct items per (pattern x dataset), selected by script with a fixed RNG seed.
    *   The closed category list is written by Alexandre and frozen before coding starts.
    *   Anonymization mechanics: a script exports shuffled, label-free transcripts; the label map is kept by Alexandre and rejoined only after both coders finish.
    *   *Validation:* Alexandre will evaluate a sample of these coded logs to calculate Cohen's Kappa.

---

## 3. Checkpoint (Wednesday, July 22)
Review preliminary data joins. We will verify the `(query_id, pattern, seed)` matching across telemetry and grading, and validate the AUROC calculation logic.

---

## 4. Deadline & Handover
Analysis freeze: Thursday, July 23 (EOD); no new numbers after this point. Friday, July 24: write and deliver Results Report v1. Handover deliverables (mandatory; internships end July 31): frozen JSONL telemetry, per-item score tables, and scripts that regenerate every figure and table in the report from those two inputs.
