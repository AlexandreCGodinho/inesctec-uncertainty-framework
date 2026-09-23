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
    latency_ms: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0


class RunState(BaseModel):
    """The complete, replayable state of one query through one pattern."""
    run_id: str
    pattern: PatternID
    dataset: str                          # "popqa" | "livebench" | "finqa" | "ifeval"
    query_id: str
    query: str
    reference: str                        # ground truth / reference context
    calls: list[LLMCall] = Field(default_factory=list)
    seed: int = 0
    model_name: str = ""
    provider: str = ""
    agent_seeds: list[int] | None = None
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
    judge_parse_error: bool = False
    query_id: str = ""
    dataset: str = ""                     # "popqa" | "livebench" | "finqa" | "ifeval"
    seed: int = 0
    agent_seeds: list[int] | None = None  # audits P4 sub-agent seeds
    final_answer: str | None = None
    termination_reason: str | None = None
    tokens_prompt: int = 0
    tokens_completion: int = 0
    latency_ms: float = 0.0
    model_name: str = ""
    provider: str = ""
