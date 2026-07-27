# 02. Experimental Design
**INESC TEC Summer Research Internship 2026**
Version 3.0 (supersedes Methodological_Baseline_v2.md; renamed for clarity: this document defines the fixed experimental conditions). Owns: base model selection, hardware and memory constraints, dataset selection, execution phases. For metric formulas see `03_EVALUATION_METRICS_SPECIFICATION.md`. For pattern descriptions see `01_PROJECT_DESCRIPTION.md`.

---

## 1. Purpose

The experiment compares four agentic orchestration patterns under identical conditions. This document locks the three experimental conditions that must be held constant so that differences in telemetry are attributable to orchestration alone:

1. the **base LLM** (Section 2 and 3),
2. the **hardware envelope** (Section 4),
3. the **datasets** (Section 5),

and defines the **execution phases** (Section 6).

---

## 2. Candidate Model Screening

State-of-the-art open-weights models were screened using public leaderboards: Arena.ai (LMSYS), LiveBench.ai, Artificial Analysis, Hugging Face Open LLM Leaderboard v2, Vellum.ai, and LLM-Stats.com.

### Table 1: Screening of Open-Weights Candidates

| Candidate Model | Configuration | Arena.ai (Elo) | LiveBench (% Unseen Bias) | HF Open LLM v2 (Avg) | BFCL Tool Acc. (%) | VRAM (Weights Only) |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Gemma 4 12B IT | 12B Dense | ~1215 | 58.4% | 73.5 | 83.2% | 24 GB (FP16) |
| Gemma 4 31B IT | 31B Dense | ~1265 | 62.4% | 79.8 | 87.2% | 62 GB (FP16) |
| Qwen 3.5 27B Instruct | 27B Dense | ~1255 | 61.8% | 78.5 | 88.1% | 54 GB (FP16) |
| Qwen 3.5 35B Instruct | 35B Dense | ~1272 | 63.2% | 81.1 | 89.4% | 70 GB (FP16) |
| Llama 3.3 70B Instruct | 70B Dense | ~1290 | 66.8% | 83.4 | 86.0% | 70 GB (FP8) |
| Llama 4 Scout | 109B MoE (10B active) | ~1295 | 68.2% | 84.1 | 88.5% | >109 GB |

Throughput figures from vendor leaderboards refer to H100 hardware and are therefore not reported here; they do not transfer to our execution environment (Section 4).

Models above ~30 GB deployable footprint are excluded by the hardware envelope (Section 4), leaving two finalists, evaluated in Section 3.

---

## 3. Base Model Selection

### 3.1 Decision Criterion (stated before the choice)

The study measures the differential effect of orchestration, not absolute task performance. Consequently:

1. **Telemetry fidelity dominates reasoning capacity.** u_P (the Plausibility Paradox term) is the central variable of the research question, and it requires unbiased log-probabilities. Any quantization that distorts the log-probability distribution damages the study's key measurement.
2. **A weaker base model is acceptable, and even useful.** More baseline errors produce more variance for the orchestration layers to act upon, increasing the statistical power of the between-pattern comparison.
3. **KV-cache headroom protects the deliberative pattern.** Pattern 4 holds concurrent conversational state; insufficient cache truncates context and artificially inflates u_C and u_R (see Section 4.2). Headroom is therefore a scientific requirement, not a convenience.

### 3.2 The Two Finalists

| Candidate | A100 Deployment | Quantization Impact on Telemetry | Profile |
| :--- | :--- | :--- | :--- |
| **Gemma 4 12B IT** | Native bfloat16, ~24 GB | None. Log-probability distributions fully preserved. | High-fidelity telemetry, lower reasoning capacity. |
| **Qwen 3.5 27B IT** | INT8, ~27 GB | Low but nonzero: minor syntax parsing variance and mild perplexity drift. | Better reasoning (relevant for FinQA), degraded u_P precision. |

### 3.3 Decision

Applying the criterion of 3.1, the selected base model is **Gemma 4 12B IT (bfloat16)**: it is the only finalist with zero quantization impact on u_P telemetry, and it leaves the largest KV-cache headroom (Section 4.1). Qwen's reasoning advantage optimizes the wrong objective for this study.

> Status: pending final sign-off at the coordination meeting. If Qwen is chosen instead, a mandatory calibration step must be added to Phase 1: compare INT8 vs bfloat16 log-probabilities on a held-out sample to quantify and correct the perplexity drift before any u_P figure is reported.

---

## 4. Hardware Envelope: Google Colab Pro, NVIDIA A100 80 GB

All execution is restricted to the **A100 (80 GB VRAM)** tier of Google Colab Pro. The H100 tier is excluded due to prohibitive credit consumption. All memory reasoning below is for 80 GB.

### 4.1 Memory Allocation Map (A100, 80 GB)

```
====================================================================================
[Gemma 4 12B bf16]  ████████████████████████ [~24 GB weights] ░░░░░░░░░░░░░░░░ [~16 GB KV-cache]
[Qwen 3.5 27B INT8] ███████████████████████████ [~27 GB weights] ░░░░░░░░░░░░░ [~13 GB KV-cache]
====================================================================================
```

Log-probabilities for u_P are extracted through **vLLM** (`logprobs=True`) running on the A100 instance.

### 4.2 KV-Cache as a Confounder, and Its Control

In Pattern 4 (Multi-Perspective Deliberation), concurrent agents maintain extensive conversational history. If the combined footprint of weights plus active cache exceeds GPU capacity, vLLM evicts or truncates context. Truncation changes agent answers for hardware reasons, inflating u_C and u_R with variance that has nothing to do with orchestration.

Mandatory controls, enforced by the orchestrator:

1. **Per-agent context ceiling** in Pattern 4 (value set during Phase 2 profiling; initial ceiling 8k tokens per agent).
2. **Truncation event logging**: every run records whether any truncation or cache eviction occurred.
3. **Exclusion rule**: runs with truncation events are flagged and excluded from the main analysis (reported separately in an appendix).

### 4.3 Ensemble Size for Pattern 4 (Locked Parameter: K = 3)

Pattern 4 executes with **exactly three symmetric agents** in every run, recorded in `config/experiment.yaml` (`pattern4_num_agents: 3`) and held constant across all datasets and phases. The methodological rationale (estimator stability of u_C, majority/dissent structure, and the symmetry requirement) is owned by `03_EVALUATION_METRICS_SPECIFICATION.md`, Section 2.3.1. This document locks the parameter and its operational consequences:

* **Serial execution.** The three agent generations within a round are independent and run **serially** on the single A100; the KV-cache is released between generations, so K = 3 does not require concurrent cache residency. Only the Judge's context grows with K, and it is bounded by the per-agent context ceiling of Section 4.2.
* **Cost impact.** Relative to a two-agent configuration, K = 3 adds roughly 50% inference volume to Pattern 4 in Phase 3. This is absorbed, if necessary, by reducing the per-dataset sample size, a trade accepted because a u_C built on a single pairwise distance (K = 2) would not support the study's main comparison.
* **Sensitivity check.** Phase 2 includes a small-sample K = 2 vs K = 3 comparison on FinQA, so the final report can state the choice was empirically examined.

### 4.4 Generation Imbalance (Pattern 1 vs Pattern 2)

We explicitly acknowledge a structural imbalance in compute volume between Pattern 1 (simplest-pattern baseline) and Pattern 2 (role-separated pipeline). By design, Pattern 1 executes 2 LLM generations per query, whereas Pattern 2 requires 3 generations to complete its workflow. This asymmetry is inherent to the architectural patterns being evaluated rather than a methodological oversight. Consequently, any performance differences between P1 and P2 must be interpreted with the understanding that they are not strictly compute-matched.

---

## 5. Dataset Selection

Four datasets are selected to give every U(x) component at least one dataset where it is the primary signal, while avoiding benchmark saturation and training-set contamination. Clinical datasets (e.g., MedQA) were considered and rejected: baseline floor effects would inject variance into u_C that is unrelated to orchestration. FinQA provides numerical multi-step reasoning without that failure mode.

### Table 2: Dataset-to-Variable Mapping

| Dataset | Domain | Primary Variable | Secondary Variable | Why this dataset stresses these variables |
| :--- | :--- | :---: | :---: | :--- |
| PopQA (long-tail) | Open-domain knowledge | u_P | u_F | Long-tail Wikidata entities push the model beyond its training distribution, producing fluent but hallucinated prose, exactly the u_P signature. |
| LiveBench | Logical inference | u_R | u_L | Contamination-free by construction; adversarial critique rounds reveal whether reasoning stays anchored or drifts. |
| FinQA | Numerical reasoning | u_C | u_R | Multi-step numeric problems make convergence between symmetric agents meaningful and errors objectively detectable. |
| IFEval | Constraint adherence | u_L | u_C | Non-cognitive, programmatically verifiable constraints isolate formatting failures from reasoning failures. |

### Table 3: Programmatic Handles and References

| Dataset | Citation | Initialization |
| :--- | :--- | :--- |
| PopQA | Mallen et al., 2023. When Not to Trust Language Models: Investigating Retrieval-Augmented Generation. | `load_dataset("akariasai/PopQA")` |
| IFEval | Zhou et al., 2023. Instruction-Following Evaluation for Large Language Models. | `load_dataset("google/IFEval")` |
| LiveBench | Abacus.AI, 2025. LiveBench: A Challenging, Contamination-Free LLM Benchmark. | via `inspect_evals` or livebench.ai source API |
| FinQA | Chen et al., 2021. FinQA: A Dataset of Numerical Reasoning over Financial Data. | `load_dataset("czyssrs/FinQA")` |

Sample sizes per dataset and per phase are fixed during Phase 1 and recorded in the repository (`config/experiment.yaml`), never in this document.

---

## 6. Execution Lifecycle

All phases run exclusively on the A100 environment with the base model of Section 3.

### Phase 1: Zero-Shot Ablation (Methodological Control)
* **Objective:** establish baseline values of all U(x) components with no orchestration at all.
* **Protocol:** the base LLM is evaluated in a zero-shot, single-prompt configuration across the four datasets. Subsequent variance is thereby attributable exclusively to the orchestration pattern. If Qwen is selected, the INT8 calibration of Section 3.3 happens here.

### Phase 2: Development, Syntactic Validation, and Debugging
* **Objective:** functional implementation of the four orchestrators (plain Python + Pydantic).
* **Protocol:** resolve state machine transition anomalies, structure the model-call wrappers, remediate JSON schema mismatches. Initial validation on IFEval and PopQA to establish pipeline integrity. KV-cache profiling for the Pattern 4 context ceiling (Section 4.2) and the K = 2 vs K = 3 sensitivity check (Section 4.3) happen here.

### Phase 3: Production Evaluation and Telemetry Extraction
* **Objective:** batch execution to extract final U(x) telemetry.
* **Protocol:** full-scale inference across all four datasets and all four patterns. vLLM extracts log-probabilities for u_P. Telemetry is synchronized to the validation repository. Analysis follows the comparison protocol of `03_EVALUATION_METRICS_SPECIFICATION.md`, Section 4.

## 7. Design Constants

The following parameters are designated as **Design Constants**. None of these may be changed without prior written approval from Alexandre and Álvaro:
1. **Base Model**: Gemma 4 12B IT (bfloat16)
2. **Inference Provider**: vLLM
3. **Hardware Envelope**: A100 (80 GB VRAM)
4. **Datasets and Sampled Query IDs**: N=150 queries per dataset (confirmed by A100 Smoke Run CU burn rate). (Budget math corrected: N=150 generates ~17,600 total calls considering P4x3 and u_F batching, totaling ~477 CUs, maintaining a safety margin against the 600 CUs limit).
5. **Prompts after the freeze**
6. **Iteration caps**
7. **Ensemble Size**: K = 3

## 8. Pilot Data Status

The pilot of 18-19/07 is formally designated as the **development split (Llama 3.1 8B via Groq)**. The corresponding files are committed and tracked under the directory `data/dev_split_groq_llama31/`.
Prompt tuning is allowed on pilot data only; the final report run uses prompts frozen at commit `<hash>` prior to launch, on a disjoint sampled query set.
Prompt-engineering effort is equalized across patterns and documented here.

## 9. Amendment 1 (27/07) - Sampling Protocol

A post-pilot audit revealed a structural bias in the initial sampling logic, which used a deterministic first-N slice (`[:sample_size]`). On datasets ordered by category (such as PopQA, whose first 150 items are all "occupation" relations), this produced an unrepresentative sample.

To correct this, the sampling protocol has been updated:
1. **Seeded Uniform Sampling:** The first-N slice was replaced with uniform random sampling without replacement, using a local `random.Random(SAMPLING_SEED)` instance to guarantee determinism without relying on the global random state.
2. **Configuration-driven Seed:** `SAMPLING_SEED` is defined exclusively in `experiment.yaml`, completely separated from the script logic.
3. **Contamination Control:** All `query_ids` previously exposed during the Groq pilot (the dev split) are strictly filtered out and excluded from the pool before sampling occurs. This ensures the N=150 report run is a completely disjoint and virgin sample.

## 10. Amendment 2 (27/07) - Conditional Prompt Framing

A pre-launch audit of the Gemma PopQA telemetry from the A100 smoke run revealed a prompt-framing artifact. Because PopQA lacks reference passages (`context` is empty), Gemma strictly obeyed the constraint to "Base your answer strictly on the provided context" and output refusal artifacts across all four patterns (e.g., *"The provided context does not contain any information regarding John Mayne's occupation"*).

To fix this without distorting datasets that do provide context (like FinQA), conditional prompt framing has been applied identically across all four orchestrator patterns:

**Non-Empty Context (e.g., FinQA, LiveBench)**
The original strict-grounding prompt remains unchanged. For example:
> *"Using the context below, draft an answer to the question. Base your answer strictly on the provided context. Do not add external knowledge."*

**Empty Context (e.g., PopQA, IFEval)**
The strict constraint and the `CONTEXT:` block are removed, replaced exactly by:
> *"Answer the request below using your own knowledge."*

This change was authorized jointly by AF and AG and was implemented across all four pattern modules before any report-run results were seen.
