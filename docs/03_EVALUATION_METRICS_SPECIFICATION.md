# 03. Evaluation Metrics Specification
**INESC TEC Summer Research Internship 2026**
Version 2.0 (supersedes EVALUATION_METRICS_SPECIFICATION.md v1). **This is the single canonical source for every metric definition.** Any table or description of metrics found elsewhere is informal and yields to this document. Owns: component definitions and formulas, variable dependency graph, activation matrix, comparison protocol, accuracy reporting protocol. For datasets and model see `02_EXPERIMENTAL_DESIGN.md`.

---

## 1. The Composite Score

The foundational paper (When to Intervene?) defines a Composite Score of Uncertainty:

$$U(x) = \sum_{k} w_k\,u_k(x)$$

All components are normalized to [0, 1], where 0 indicates absolute certainty and 1 indicates maximum uncertainty. The components, their measurement, and their applicability per pattern are defined below. Implementation is plain Python + Pydantic; the orchestrator reads the activation matrix (Section 3) at runtime.

---

## 2. Component Definitions (Canonical)

### 2.1 u_L (Linguistic Instability)
* **Concept:** instability of the generation process, observed as degraded capacity to follow rigid formatting instructions.
* **Practical adaptation:** extracting sub-token entropy across multiple samples exceeds the I/O budget of a multi-agent run; structural instability is used as a behavioral proxy.
* **Implementation:** programmatic verification of IFEval-style schema constraints.
* **Formula:**
  $$u_L = 1 - \frac{\text{Passed Schema Constraints}}{\text{Total Constraints Applied}}$$

### 2.2 u_F (Factual Support)
* **Concept:** factual deviation and hallucination relative to the reference context.
* **Implementation:** the response is decomposed into atomic claims; an internal LLM judge scores each claim against canonical references, with severity weights.
* **Formula:** each claim receives $s_j \in \{-1, 0, 0.5, 1\}$ and a severity weight $\alpha_j$; the weighted mean is $\bar{s} = \frac{\sum_j \alpha_j s_j}{\sum_j \alpha_j}$, and
  $$u_F = \frac{1 - \bar{s}}{2}$$
  All claims contradicted gives $u_F = 1$; all supported gives $u_F = 0$.

### 2.3 u_C (Inter-Agent Divergence)
* **Concept:** structural divergence among **parallel, symmetric agents answering the same query**. High divergence signals that the ensemble is uncertain.
* **Scope restriction (important):** u_C is only meaningful when agents are symmetric. In pipelines where agents produce artifacts of different kinds by design (a researcher's evidence dump vs a reviewer's verdict), embedding distance measures role difference, not disagreement, and is therefore not computed. See the activation matrix (Section 3).
* **Implementation:** sentence embeddings (e.g., `all-MiniLM-L6-v2`) over final answers; mean pairwise cosine distance across the K agents.
* **Formula:**
  $$u_C = 1 - \frac{1}{\binom{K}{2}} \sum_{i \neq j} \text{CosineSimilarity}(E(A_i), E(A_j))$$
* **Design change vs v1:** v1 blended divergence with mean factual support ($u_C = (D + (1-A))/2$). That coupling made u_C a second measurement of u_F and destroyed the independence of the components (see Section 5). In v2, u_C measures divergence only. The scientifically interesting case of "confident consensus on a wrong answer" is not lost: it is captured in the analysis as the interaction of low u_C with high u_F, reported explicitly in Phase 3 results.

#### 2.3.1 Ensemble Size and Symmetry (K = 3, Symmetric Agents)

Pattern 4 runs with exactly **K = 3 agents**, and the agents must be **symmetric**: identical instructions, with output diversity produced exclusively by sampling (distinct seeds). Both decisions follow from the validity of u_C itself.

**Why not K = 2.** The u_C formula averages cosine distance over $\binom{K}{2}$ pairs. With K = 2 that is a single pair, so u_C collapses to one raw pairwise distance: no averaging, maximal estimator variance, and no way to distinguish "one agent produced an outlier" from "the ensemble is genuinely uncertain". Since u_C is the primary variable on FinQA, running at K = 2 would rest that dataset's headline metric on the noisiest possible estimator. K = 2 also removes any majority structure: the Judge can only arbitrate a tie, and the analysis loses the qualitatively distinct signal of "two agents agree, one dissents".

**Why K = 3 (and not 4 or more).** K = 3 is the smallest ensemble that (a) averages over multiple pairs (three), stabilizing the u_C estimate, and (b) supports majority/dissent structure, which feeds directly into the silent-failure analysis of Section 4.3: the interesting question is whether a dissenting agent flags exactly the queries where surface metrics look confident but the answer is wrong. K = 4 would improve the estimate further (six pairs) but adds one full generation per round plus Judge-context growth, aggravating the KV-cache confounder controlled in `02_EXPERIMENTAL_DESIGN.md`, Section 4.2, and consuming Colab credits that are better spent on sample size. K = 3 is the inflection point where marginal validity gain stops justifying marginal cost. A K = 2 vs K = 3 sensitivity check on a small FinQA sample is scheduled in Phase 2 so the choice is reported as tested, not assumed.

**Why symmetry is mandatory.** Assigning adversarial stances (e.g., "Agent Pro" vs "Agent Con") *instructs* divergence: agents told to defend opposite positions produce distant embeddings regardless of the model's actual uncertainty. u_C would then measure compliance with the prompt design, not epistemic instability. This is the same failure mode that excludes u_C from the Role-Separated Pipeline (asymmetric artifacts, see 2.3), reintroduced through prompting instead of through the pattern's structure. Under symmetry, convergence genuinely means the model keeps arriving at the same answer, and divergence genuinely means uncertainty. Adversarial pressure, if desired, belongs in the Judge's critique rounds, not in the identity of the agents whose outputs feed u_C.

**Calibration note.** Pairwise cosine distances between sentence embeddings occupy, in practice, a narrow band well inside [0, 1]. Like u_R, u_C is therefore rescaled against the distribution observed in the Phase 1 calibration set before entering the composite.

### 2.4 u_R (Reasoning Stability)
* **Concept:** epistemic stability across critique rounds; large shifts in the reasoning trace indicate weak initial grounding.
* **Applicability:** requires at least two comparable rounds of the same reasoning artifact (self-critique iterations, or reviewer-triggered revisions).
* **Implementation:** embedding distance between the reasoning trace at round k and round k-1.
* **Formula:** for M traces and K rounds,
  $$u_R = \frac{1}{M(K-1)} \sum_{i=1}^{M} \sum_{k=2}^{K} \text{EuclideanDistance}(E(r_i^{(k)}), E(r_i^{(k-1)}))$$
  normalized to [0, 1] by the maximum observed distance in the Phase 1 calibration set.
* **Note (Consensus Effect):** In cases of immediate consensus, $u_R$ is zero, which structurally inflates the metric with zeros for easy queries. For analysis, the failure detector must be reported **with and without** the $u_R$ component. Furthermore, the existence of refinement loops must be extracted and treated as an independent binary signal (`refinement_triggered: Yes/No`).

### 2.5 u_P (Plausibility Paradox)
* **Concept:** the gap between fluency and factual fidelity: highly fluent but factually wrong output, the primary driver of automation bias.
* **Implementation:** log-probabilities extracted via vLLM (`logprobs=True`); fluency F as normalized inverse perplexity; fidelity T defined from u_F.
* **Formula:**
  $$u_P = \max(0,\; F - T), \qquad T = 1 - u_F$$
* **Declared dependency:** u_P is by construction a function of u_F. This is intentional: the paradox is *defined* as fluency exceeding fidelity. The dependency is declared here and accounted for in Section 5; it must never be presented as an independent measurement of factuality.

---

## 3. Applicability and the Activation Matrix

### 3.1 The Structural Problem

The four patterns do not all expose the same signals:

1. **Governed Workflow:** single linear pass; no critique rounds, no parallel agents. u_C and u_R do not exist.
2. **Role-Separated Pipeline:** sequential specialized roles with reviewer-triggered revisions. u_R exists (revision rounds are comparable artifacts). u_C does **not** exist: the agents are asymmetric, so embedding distance between their outputs is uninformative (see 2.3).
3. **Reflexive Single Agent:** self-critique rounds; u_R exists, u_C does not.
4. **Multi-Perspective Deliberation:** parallel symmetric agents plus rounds; all five components exist.

Assigning maximum uncertainty (u_C = 1) to a pattern that structurally cannot emit the signal would distort its score; dropping terms without reweighting would make the weights sum to less than 1.

### 3.2 The Activation Matrix

When a pattern lacks the structural capacity to generate a component, that component is deactivated and its weight redistributed among the active components.

| Agentic Pattern | w_L | w_F | w_P | w_R | w_C | Active Sensors |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1. Governed Workflow | 0.33 | 0.33 | 0.34 | 0.00 | 0.00 | Surface only |
| 2. Role-Separated Pipeline | 0.25 | 0.25 | 0.25 | 0.25 | 0.00 | Surface + revision drift |
| 3. Reflexive Single Agent | 0.25 | 0.25 | 0.25 | 0.25 | 0.00 | Surface + self-critique drift |
| 4. Multi-Perspective Delib. | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | Full spectrum |

The orchestrator reads this matrix at runtime (`config/activation_matrix.yaml`) to compute U(x) per pattern.

> **Change vs v1:** Pattern 2 (Role-Separated Pipeline) is no longer "full spectrum". Its w_C is zero, per the symmetry requirement of 2.3. Pattern numbering here follows `01_PROJECT_DESCRIPTION.md` Section 3.

---

## 4. Comparison Protocol (How Patterns Are Compared Fairly)

A full-spectrum U(x) of Pattern 4 is a sum over five components; a Pattern 1 score sums three. Comparing those two numbers directly compares different instruments, not different systems: a higher Pattern 4 score could simply reflect additional noisy terms. The protocol therefore operates on two levels.

### 4.1 Confirmatory Level: the Shared Surface Score

For between-pattern comparison, the primary endpoint is the surface score computed identically for all four patterns:

$$U_{\text{surface}}(x) = \tfrac{1}{3}\,u_L + \tfrac{1}{3}\,u_F + \tfrac{1}{3}\,u_P$$

Every claim of the form "pattern A is less uncertain than pattern B" must be made on U_surface, on identical queries.

### 4.2 Diagnostic Level: Pattern-Specific Sensors

u_R and u_C are reported for the patterns that expose them, as evidence about **where** uncertainty manifests, not as part of the compared score. The full U(x) with the activation matrix is reported per pattern as descriptive telemetry.

### 4.3 The Silent-Failure Hypothesis (Falsifiable Formulation)

The hypothesis of interest is that simple patterns appear confident because they lack mechanisms (consensus, critique) that would expose their uncertainty. This is tested, not assumed, as follows: identify queries where Pattern 1 shows low U_surface (apparent confidence) but a wrong final answer against the dataset ground truth; measure how often, on those same queries, Pattern 4's deep sensors (u_C, u_R) are elevated. If the deep sensors systematically flag the queries that surface metrics miss, that supports the silent-failure hypothesis. If they do not, the hypothesis is weakened. Both outcomes are publishable findings.

---

## 5. Variable Dependency Graph (Declared)

The five components are **not** mutually independent, and no document should describe them as orthogonal. The dependency structure, declared:

```
u_L   independent (programmatic constraints)
u_F   independent (LLM judge over atomic claims)
u_C   independent of u_F as of v2 (pure divergence)
u_R   independent (trace drift)
u_P   = f(u_F) by construction (the paradox definition)
```

Consequences for analysis:

1. u_F and u_P are never interpreted as two independent confirmations of factuality; effective weight of factual content in U(x) is w_F plus the u_F-mediated share of w_P.
2. All Phase 3 reports include the per-component breakdown, never only the composite.
3. Correlations between components are computed and reported in Phase 3 as a validity check of this declared structure.

---

## 6. Accuracy Reporting Specification

1) Primary output is a per-item score table, not percentages. One row per (query_id, pattern, seed), with: dataset, official metric value(s), termination reason, tokens, latency.
In the primary key, seed is the run seed. Pattern 4's K = 3 agent seeds are derived deterministically from the run seed (e.g., agent_seed_i = run_seed * 100 + i; the exact rule is Pedro's implementation choice, but it must be deterministic and documented) and never appear in the key. Pattern 4 runs with 3 run seeds; Patterns 1-3 with a single run seed.
All downstream analysis (pattern comparison, U(x) join, cost normalization) reads from this table. The harness adapter from the bridge task produces it. Correctness scores use accuracy semantics (1 = correct); uncertainty components use u-semantics (1 = maximally uncertain); the prefix identifies the family.

2) Accuracy is reported per (pattern x dataset), using each benchmark's official metric, named explicitly. PopQA: "alias-match accuracy". FinQA: "execution accuracy" (program accuracy reported separately). IFEval: all four official variants, with prompt-level strict as the headline. LiveBench: official per-category scores, benchmark version pinned and recorded. A generic "Accuracy (%)" label is not used anywhere.

3) No pooled number across datasets. If a summary is needed, use a macro-average with equal weight per dataset, labeled as such.
Item-level pooling is forbidden: it silently weights by dataset size.

4) Comparisons are paired. All patterns run on the same queries. Reference pattern: P1 (the simplest-pattern baseline; "single-agent" would be ambiguous, since P3 is also single-agent). The test depends on the data type of the headline column: for the binary headlines (PopQA alias match, FinQA execution accuracy, IFEval prompt-level strict), use McNemar's test, exact version if discordant pairs are few; for LiveBench's numeric per-category scores, use a paired bootstrap over per-item differences. In all cases report confidence intervals for the differences, not for individual values. Paired tests are what gives us power when differences are a few points.

5. Multiple comparisons: Benjamini-Hochberg FDR across the pattern x dataset test family, implemented with statsmodels.stats.multitest.multipletests(pvals, method='fdr_bh'). Consistent with our other work.

6) Variance: multiple seeds for Pattern 4 only (budget constraint; three full-run seeds would consume ~450 CUs), with between-seed variance reported separately. P1 to P3 get paired-bootstrap confidence intervals.

7. Scoring rules are frozen here, before results. Two distinct cases:
(a) No usable answer exists (timeout, crash, unparseable output): score = 0 (incorrect), flagged by the adapter (parse_error / timeout).
(b) Cap-exhausted Pattern 4 answers (termination_reason = cap): the emitted final_answer IS scored normally by the official harness. Cap exhaustion is recorded in termination_reason, used as an explanatory covariate, and accuracy is additionally reported stratified by termination_reason.
**Rationale**: forcing score = 0 on cap-exhausted items would make U(x) predict a label it helped construct (u_R and refinement_triggered are elevated exactly on those queries), circularly inflating AUROC and the risk-coverage curves. Scoring the emitted answer and stratifying preserves the failure signal without fabricating it. The adapter flags these categories; this spec, not the analyst, decides them.

8) Selective accuracy: accuracy as a function of coverage, using U(x) to abstain, plus AUROC of U(x) as a failure detector. This is the analysis that carries the FAccT claim; everything above exists to make it possible.
