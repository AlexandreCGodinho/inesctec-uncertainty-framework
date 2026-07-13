import re
from pathlib import Path

import yaml

from src.patterns.base import Orchestrator
from src.schemas import RunState, LLMCall


RESEARCHER_PROMPT = """You are a RESEARCHER agent. Extract key facts from the context relevant to the query.
Be precise, cite sources, do NOT add opinions.

CONTEXT:
{context}

QUERY: {query}

KEY FACTS:"""

ANALYST_PROMPT = """You are an ANALYST agent. Synthesize the research into a clear, structured answer.
Use logical reasoning, connect facts, identify gaps.

RESEARCH FINDINGS:
{research}

CONTEXT:
{context}

QUERY: {query}

YOUR ANALYSIS:"""

ANALYST_REVISION_PROMPT = """You are an ANALYST agent. Revise your previous analysis to address the reviewer's critique below.
Use logical reasoning, connect facts, identify gaps.

PREVIOUS ANALYSIS:
{previous_analysis}

REVIEWER CRITIQUE:
{reviewer_critique}

CONTEXT:
{context}

QUERY: {query}

REVISED ANALYSIS:"""

REVIEWER_PROMPT = """You are a REVIEWER agent. Critique the analysis for factual accuracy.
Check: are all claims supported? Is anything missing? Is the reasoning sound?

ORIGINAL CONTEXT:
{context}

ANALYSIS TO REVIEW:
{analysis}

QUERY: {query}

VERDICT (approved/needs_revision/rejected):
CRITIQUE:"""

_APPROVED_PATTERN = re.compile(r"\bapproved\b", re.IGNORECASE)
_NEGATION_PATTERN = re.compile(r"\b(not|un|non)[\s-]*approved\b", re.IGNORECASE)


def _is_approved(response_text: str) -> bool:
    if _NEGATION_PATTERN.search(response_text):
        return False
    return bool(_APPROVED_PATTERN.search(response_text))


def _find_reviewer_by_round(calls: list[LLMCall], round_index: int) -> LLMCall | None:
    for c in reversed(calls):
        if c.agent_role == "reviewer" and c.round_index == round_index:
            return c
    return None


class RoleSeparatedPipeline(Orchestrator):
    def __init__(self, client, max_revisions=None):
        super().__init__(client)
        if max_revisions is None:
            cfg_path = Path(__file__).resolve().parents[2] / "config" / "experiment.yaml"
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            max_revisions = cfg.get("pipeline_max_revisions", 2)
        self.max_revisions = max_revisions

    def run(self, state: RunState) -> RunState:
        context = state.reference

        researcher_call = self.client.call(
            agent_role="researcher",
            prompt=RESEARCHER_PROMPT.format(context=context, query=state.query),
            round_index=0,
        )
        state.calls.append(researcher_call)

        for current_round in range(self.max_revisions + 1):
            if current_round == 0:
                analyst_call = self.client.call(
                    agent_role="analyst",
                    prompt=ANALYST_PROMPT.format(
                        research=researcher_call.response_text,
                        context=context,
                        query=state.query,
                    ),
                    round_index=0,
                )
            else:
                prev_reviewer = _find_reviewer_by_round(state.calls, current_round - 1)
                prev_analysis = state.calls[-2].response_text
                critique = prev_reviewer.response_text if prev_reviewer else ""
                analyst_call = self.client.call(
                    agent_role="analyst",
                    prompt=ANALYST_REVISION_PROMPT.format(
                        previous_analysis=prev_analysis,
                        reviewer_critique=critique,
                        context=context,
                        query=state.query,
                    ),
                    round_index=current_round,
                )
            state.calls.append(analyst_call)

            reviewer_call = self.client.call(
                agent_role="reviewer",
                prompt=REVIEWER_PROMPT.format(
                    context=context,
                    analysis=analyst_call.response_text,
                    query=state.query,
                ),
                round_index=current_round,
            )
            state.calls.append(reviewer_call)

            if _is_approved(reviewer_call.response_text):
                break

        state.final_answer = (
            f"{state.calls[-2].response_text}\n\n"
            f"---\n[Reviewer: {state.calls[-1].response_text}]"
        )
        return state
