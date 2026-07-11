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

REVIEWER_PROMPT = """You are a REVIEWER agent. Critique the analysis for factual accuracy.
Check: are all claims supported? Is anything missing? Is the reasoning sound?

ORIGINAL CONTEXT:
{context}

ANALYSIS TO REVIEW:
{analysis}

QUERY: {query}

VERDICT (approved/needs_revision/rejected):
CRITIQUE:"""


class RoleSeparatedPipeline(Orchestrator):
    def run(self, state: RunState) -> RunState:
        context = state.reference

        researcher_call = self.client.call(
            agent_role="researcher",
            prompt=RESEARCHER_PROMPT.format(context=context, query=state.query),
            round_index=0,
        )
        state.calls.append(researcher_call)

        analyst_call = self.client.call(
            agent_role="analyst",
            prompt=ANALYST_PROMPT.format(
                research=researcher_call.response_text,
                context=context,
                query=state.query,
            ),
            round_index=0,
        )
        state.calls.append(analyst_call)

        reviewer_call = self.client.call(
            agent_role="reviewer",
            prompt=REVIEWER_PROMPT.format(
                context=context,
                analysis=analyst_call.response_text,
                query=state.query,
            ),
            round_index=0,
        )
        state.calls.append(reviewer_call)

        state.final_answer = (
            f"{analyst_call.response_text}\n\n"
            f"---\n[Reviewer: {reviewer_call.response_text}]"
        )
        return state
