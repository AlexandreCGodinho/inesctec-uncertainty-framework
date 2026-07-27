import re
from pathlib import Path

import yaml

from src.patterns.base import Orchestrator
from src.schemas import RunState, LLMCall

_CONFIG_CACHE: dict | None = None


def _load_experiment_config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        cfg_path = Path(__file__).resolve().parents[2] / "config" / "experiment.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"experiment.yaml not found at {cfg_path}. "
                "Make sure config/experiment.yaml exists."
            )
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(
                f"experiment.yaml at {cfg_path} did not parse to a mapping "
                f"(got {type(cfg).__name__}); check the file is valid YAML."
            )
        _CONFIG_CACHE = cfg
    return _CONFIG_CACHE


RESEARCHER_PROMPT_WITH_CONTEXT = """You are a RESEARCHER agent. Extract key facts from the context relevant to the query.
Be precise, cite sources, do NOT add opinions.

CONTEXT:
{context}

QUERY: {query}

KEY FACTS:"""

RESEARCHER_PROMPT_NO_CONTEXT = """You are a RESEARCHER agent. Answer the request below using your own knowledge.

QUERY: {query}

KEY FACTS:"""

ANALYST_PROMPT_WITH_CONTEXT = """You are an ANALYST agent. Synthesize the research into a clear, structured answer.
Use logical reasoning, connect facts, identify gaps.

RESEARCH FINDINGS:
{research}

CONTEXT:
{context}

QUERY: {query}

YOUR ANALYSIS:"""

ANALYST_PROMPT_NO_CONTEXT = """You are an ANALYST agent. Answer the request below using your own knowledge.

RESEARCH FINDINGS:
{research}

QUERY: {query}

YOUR ANALYSIS:"""

ANALYST_REVISION_PROMPT_WITH_CONTEXT = """You are an ANALYST agent. Revise your previous analysis to address the reviewer's critique below.
Use logical reasoning, connect facts, identify gaps.

PREVIOUS ANALYSIS:
{previous_analysis}

REVIEWER CRITIQUE:
{reviewer_critique}

CONTEXT:
{context}

QUERY: {query}

REVISED ANALYSIS:"""

ANALYST_REVISION_PROMPT_NO_CONTEXT = """You are an ANALYST agent. Answer the request below using your own knowledge.

PREVIOUS ANALYSIS:
{previous_analysis}

REVIEWER CRITIQUE:
{reviewer_critique}

QUERY: {query}

REVISED ANALYSIS:"""

REVIEWER_PROMPT_WITH_CONTEXT = """You are a REVIEWER agent. Critique the analysis for factual accuracy.
Check: are all claims supported? Is anything missing? Is the reasoning sound?

ORIGINAL CONTEXT:
{context}

ANALYSIS TO REVIEW:
{analysis}

QUERY: {query}

VERDICT (approved/needs_revision/rejected):
CRITIQUE:"""

REVIEWER_PROMPT_NO_CONTEXT = """You are a REVIEWER agent. Answer the request below using your own knowledge.

ANALYSIS TO REVIEW:
{analysis}

QUERY: {query}

VERDICT (approved/needs_revision/rejected):
CRITIQUE:"""

_VERDICT_PATTERN = re.compile(
    r"VERDICT\s*(?:\([^)]*\))?\s*:\s*(approved|needs_revision|rejected)",
    re.IGNORECASE,
)


def _parse_verdict(response_text: str) -> str | None:
    match = _VERDICT_PATTERN.search(response_text)
    if match is None:
        return None
    return match.group(1).lower()


def _is_approved(response_text: str) -> bool:
    return _parse_verdict(response_text) == "approved"


def _find_reviewer_by_round(calls: list[LLMCall], round_index: int) -> LLMCall | None:
    for c in reversed(calls):
        if c.agent_role == "reviewer" and c.round_index == round_index:
            return c
    return None


def _find_analyst_by_round(calls: list[LLMCall], round_index: int) -> LLMCall | None:
    for call in calls:
        if call.agent_role == "analyst" and call.round_index == round_index:
            return call
    return None


class RoleSeparatedPipeline(Orchestrator):
    def __init__(self, client, max_revisions=None):
        super().__init__(client)
        if max_revisions is None:
            cfg = _load_experiment_config()
            max_revisions = cfg.get("pipeline_max_revisions", 2)
        if not isinstance(max_revisions, int) or isinstance(max_revisions, bool) or max_revisions < 0:
            raise ValueError(
                f"pipeline_max_revisions must be a non-negative integer, "
                f"got {max_revisions!r}."
            )
        self.max_revisions = max_revisions

    def run(self, state: RunState) -> RunState:
        context = state.reference

        if not context.strip():
            researcher_prompt = RESEARCHER_PROMPT_NO_CONTEXT.format(query=state.query)
        else:
            researcher_prompt = RESEARCHER_PROMPT_WITH_CONTEXT.format(context=context, query=state.query)

        researcher_call = self.client.call(
            agent_role="researcher",
            prompt=researcher_prompt,
            round_index=0,
        )
        state.calls.append(researcher_call)

        for current_round in range(self.max_revisions + 1):
            if current_round == 0:
                if not context.strip():
                    prompt = ANALYST_PROMPT_NO_CONTEXT.format(
                        research=researcher_call.response_text,
                        query=state.query,
                    )
                else:
                    prompt = ANALYST_PROMPT_WITH_CONTEXT.format(
                        research=researcher_call.response_text,
                        context=context,
                        query=state.query,
                    )
                analyst_call = self.client.call(
                    agent_role="analyst",
                    prompt=prompt,
                    round_index=0,
                    want_logprobs=True,
                )
            else:
                prev_reviewer = _find_reviewer_by_round(state.calls, current_round - 1)
                prev_analyst = _find_analyst_by_round(state.calls, current_round - 1)
                prev_analysis = prev_analyst.response_text if prev_analyst else ""
                critique = prev_reviewer.response_text if prev_reviewer else ""
                if not context.strip():
                    prompt = ANALYST_REVISION_PROMPT_NO_CONTEXT.format(
                        previous_analysis=prev_analysis,
                        reviewer_critique=critique,
                        query=state.query,
                    )
                else:
                    prompt = ANALYST_REVISION_PROMPT_WITH_CONTEXT.format(
                        previous_analysis=prev_analysis,
                        reviewer_critique=critique,
                        context=context,
                        query=state.query,
                    )
                analyst_call = self.client.call(
                    agent_role="analyst",
                    prompt=prompt,
                    round_index=current_round,
                    want_logprobs=True,
                )
            state.calls.append(analyst_call)

            if not context.strip():
                reviewer_prompt = REVIEWER_PROMPT_NO_CONTEXT.format(
                    analysis=analyst_call.response_text,
                    query=state.query,
                )
            else:
                reviewer_prompt = REVIEWER_PROMPT_WITH_CONTEXT.format(
                    context=context,
                    analysis=analyst_call.response_text,
                    query=state.query,
                )

            reviewer_call = self.client.call(
                agent_role="reviewer",
                prompt=reviewer_prompt,
                round_index=current_round,
            )
            state.calls.append(reviewer_call)

            if _is_approved(reviewer_call.response_text):
                break

        last_analyst = _find_analyst_by_round(state.calls, current_round)
        state.final_answer = last_analyst.response_text if last_analyst else ""
        return state
