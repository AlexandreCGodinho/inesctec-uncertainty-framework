# 04. Onboarding and Setup
**INESC TEC Summer Research Internship 2026**
Version 1.0. Owns: environment setup, installs, API access pattern, repository layout, starter template. Contains no scientific rationale: for the "why" of any choice see documents 01 to 03.

---

## 1. Accounts and Access (Day 1 Checklist)

- [ ] GitHub/GitLab account added to the project repository (ask your supervisor for the invite).
- [ ] Google account with **Colab Pro** access (licenses provided by CRACS; confirm with Alexandre that yours is active).
- [ ] Hugging Face account + access token (`Read` scope) for dataset and model downloads: https://huggingface.co/settings/tokens
- [ ] Accept the Gemma license on Hugging Face (required to download the base model weights).

Store your HF token in Colab via `Secrets` (key name: `HF_TOKEN`). Never paste tokens into notebooks or commit them.

## 2. Environment

All heavy execution happens on **Google Colab Pro with an A100 (40 GB) runtime**. Select it via `Runtime > Change runtime type > A100 GPU`. Do not use H100 instances: they burn the project's credits (see `02_EXPERIMENTAL_DESIGN.md`, Section 4, for the envelope).

Local development (writing and unit-testing orchestrator code without a GPU) is encouraged: everything except real inference runs on your laptop with Python 3.11+.

### 2.1 Installs (first cell of every Colab notebook)

```bash
pip install -q \
    "pydantic>=2.7" \
    "vllm>=0.6" \
    "litellm>=1.40" \
    "sentence-transformers>=3.0" \
    "datasets>=2.19" \
    "pandas" "numpy" "pyyaml" "tqdm" \
    "git+https://github.com/LiveBench/LiveBench.git@4bf3d6f4cb37fa8dc3967dd1b124fef5d4099635"
```

For local (non-GPU) development, omit `vllm` and use the mock client (Section 4.3).

## 3. How Model Calls Work (The API Pattern)

We do not call any external commercial API for the experiments. The base model runs **locally on the A100 via vLLM**, which exposes an OpenAI-compatible server inside the Colab instance:

```bash
# Cell: start the inference server (runs in background)
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-12b-it \
    --dtype bfloat16 \
    --max-logprobs 5 \
    --gpu-memory-utilization 0.5 \
    --port 8000 &
```

All orchestrator code talks to it through **LiteLLM**, so the calling code is identical whether the backend is the local vLLM server, a mock, or (for quick smoke tests only) a hosted model:

```python
from litellm import completion

response = completion(
    model="openai/google/gemma-4-12b-it",   # OpenAI-compatible route
    api_base="http://localhost:8000/v1",
    api_key="not-needed",
    messages=[{"role": "user", "content": "ping"}],
    logprobs=True, top_logprobs=5,           # required for u_P telemetry
    temperature=0.7, seed=1234,               # seeds are set per run config
)
```

Rule: **orchestrators never import vLLM or requests directly.** They receive a client object (Section 4.3). This keeps every pattern testable without a GPU.

## 4. Repository Layout and Starter Template

```
agentic-eval/
├── config/
│   ├── experiment.yaml           # sample sizes, seeds, model id, pattern4_num_agents
│   └── activation_matrix.yaml    # weights per pattern (from doc 03, Section 3)
├── docs/                         # official documentation and sprint plans
├── src/
│   ├── schemas.py                # Pydantic state models (shared, do not fork)
│   ├── llm_client.py             # LiteLLM wrapper + mock client
│   ├── metrics/                  # one module per u_k, coded against doc 03
│   ├── scoring/                  # dataset ground-truth scorers
│   ├── patterns/
│   │   ├── p1_governed.py        # Intern 1
│   │   ├── p2_role_pipeline.py   # Intern 1
│   │   ├── p3_reflexive.py       # Intern 2
│   │   └── p4_deliberation.py    # Intern 2
│   └── runner.py                 # dataset loop, telemetry writer
├── notebooks/                    # Colab entry points only (thin wrappers)
├── scripts/                      # data processing and utility scripts
└── telemetry/                    # JSONL outputs, one file per run (gitignored)
```

### 4.0 Experiment Configuration (`config/experiment.yaml`)

```yaml
model_id: google/gemma-4-12b-it     # locked in doc 02, Section 3
dtype: bfloat16
base_seed: 1234
pattern4_num_agents: 3               # locked in doc 02, Section 4.3; do not change
pattern4_agent_seeds: [1234, 5678, 9012]   # one seed per symmetric agent
pattern4_context_ceiling_tokens: 8000       # doc 02, Section 4.2; tuned in Phase 2
samples_per_dataset:                 # set during Phase 1
  popqa: null
  livebench: null
  finqa: null
  ifeval: null
```

Everything experimental lives here. If a number appears in code instead of in this file, that is a bug.

### 4.1 Shared State Schemas (`src/schemas.py`)

```python
"""Shared Pydantic state. Both interns code against these models.
Changing a field requires agreement from both interns + supervisor."""
from datetime import datetime, timezone
from enum import IntEnum
from pydantic import BaseModel, Field


class PatternID(IntEnum):
    GOVERNED_WORKFLOW = 1
    ROLE_SEPARATED_PIPELINE = 2
    REFLEXIVE_SINGLE_AGENT = 3
    MULTI_PERSPECTIVE_DELIBERATION = 4


class LLMCall(BaseModel):
    """One model invocation, fully audited."""
    agent_role: str                      # e.g. "researcher", "judge", "solo"
    round_index: int = 0                 # critique/revision round, 0-based
    prompt: str
    response_text: str
    logprobs: list[float] | None = None  # required in Phase 3 for u_P
    truncated: bool = False              # KV-cache control, doc 02 Section 4.2


class RunState(BaseModel):
    """The complete, replayable state of one query through one pattern."""
    run_id: str
    pattern: PatternID
    dataset: str                          # "popqa" | "livebench" | "finqa" | "ifeval"
    query_id: str
    query: str
    reference: str                        # ground truth / reference context
    calls: list[LLMCall] = Field(default_factory=list)
    final_answer: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def rounds(self) -> int:
        return 1 + max((c.round_index for c in self.calls), default=0)


class Telemetry(BaseModel):
    """Metric outputs for one run. None = component not active for this pattern."""
    run_id: str
    pattern: PatternID
    u_L: float | None = None
    u_F: float | None = None
    u_C: float | None = None
    u_R: float | None = None
    u_P: float | None = None
    U_surface: float | None = None        # doc 03, Section 4.1
    U_composite: float | None = None      # doc 03, Section 3.2 weights
    truncation_flag: bool = False
```

### 4.2 Orchestrator Interface (`src/patterns/base.py`)

```python
from abc import ABC, abstractmethod
from src.schemas import RunState
from src.llm_client import LLMClient


class Orchestrator(ABC):
    """Every pattern implements exactly this. The runner treats all four
    patterns identically; all pattern-specific logic lives inside run()."""

    def __init__(self, client: LLMClient):
        self.client = client

    @abstractmethod
    def run(self, state: RunState) -> RunState:
        """Consume a RunState with the query filled in; return it with
        `calls` populated and `final_answer` set. Must be deterministic
        given the client's seed. Must never mutate global state."""
```

### 4.3 LLM Client with Mock (`src/llm_client.py`)

```python
from litellm import completion
from src.schemas import LLMCall


class LLMClient:
    def __init__(self, api_base: str = "http://localhost:8000/v1",
                 model: str = "openai/google/gemma-4-12b-it",
                 seed: int = 1234):
        self.api_base, self.model, self.seed = api_base, model, seed

    def call(self, agent_role: str, prompt: str, round_index: int = 0,
             want_logprobs: bool = False) -> LLMCall:
        r = completion(
            model=self.model, api_base=self.api_base, api_key="not-needed",
            messages=[{"role": "user", "content": prompt}],
            logprobs=want_logprobs, top_logprobs=5 if want_logprobs else None,
            temperature=0.7, seed=self.seed,
        )
        text = r.choices[0].message.content
        lps = None
        if want_logprobs and r.choices[0].logprobs:
            lps = [t.logprob for t in r.choices[0].logprobs.content]
        return LLMCall(agent_role=agent_role, round_index=round_index,
                       prompt=prompt, response_text=text, logprobs=lps)


class MockLLMClient(LLMClient):
    """GPU-free client for local development and unit tests."""
    def __init__(self):  # no server needed
        self.seed = 0

    def call(self, agent_role, prompt, round_index=0, want_logprobs=False):
        return LLMCall(agent_role=agent_role, round_index=round_index,
                       prompt=prompt,
                       response_text=f"[mock:{agent_role}:r{round_index}]",
                       logprobs=[-0.1, -0.2] if want_logprobs else None)
```

### 4.4 Minimal Working Example (your first task)

```python
# notebooks/smoke_test.ipynb (works locally, no GPU)
from src.llm_client import MockLLMClient
from src.schemas import RunState, PatternID
from src.patterns.p1_governed import GovernedWorkflow

state = RunState(run_id="smoke-001", pattern=PatternID.GOVERNED_WORKFLOW,
                 dataset="ifeval", query_id="demo",
                 query="Answer in exactly two sentences: what is INESC TEC?",
                 reference="INESC TEC is a research institute in Porto.")

result = GovernedWorkflow(MockLLMClient()).run(state)
print(result.final_answer, f"({len(result.calls)} calls, {result.rounds} rounds)")
```

Week 1 goal for each intern: make this smoke test pass for your first pattern with the mock client, then swap in the real client on Colab and confirm one real inference with logprobs present.

## 5. Working Agreements

* Branches: `p1-governed`, `p2-pipeline`, `p3-reflexive`, `p4-deliberation`; merge to `main` by PR reviewed by the other intern (yes, cross-review: you are each other's first reviewer).
* `src/schemas.py` and `src/metrics/` are shared code: changes require a PR approved by both interns.
* Every run writes one JSONL line per query to `telemetry/`; never overwrite, never edit telemetry by hand.
* Seeds and sample sizes come from `config/experiment.yaml`, never hardcoded.
