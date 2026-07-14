import re

from src.patterns.base import Orchestrator
from src.schemas import RunState

# NOTE: This retrieval step is pure Python (no LLM call), so it does
# NOT produce an LLMCall and is NOT added to state.calls. Only the
# two downstream LLM calls (Draft, Format) are recorded in state.calls.


def _retrieve_relevant_context(context: str, query: str) -> str:
    """Deterministic lexical filter: keep sentences whose normalized words
    share at least one token with the query. Pure Python, no embeddings."""
    query_words = set(query.lower().split())

    sentences = re.split(r"(?<=[.!?])\s+", context.strip())
    if not sentences or not query_words:
        return context

    kept = []
    for sent in sentences:
        sent_lower = sent.lower()
        if any(w in sent_lower for w in query_words):
            kept.append(sent)

    return " ".join(kept) if kept else context


DRAFTER_PROMPT = """You are a research analyst. Using the context below, draft an answer to the question.
Base your answer strictly on the provided context. Do not add external knowledge.

CONTEXT:
{context}

QUESTION: {query}

DRAFT ANSWER:"""

FORMATTER_PROMPT = """You are an editor. Review and format the draft answer below into a clear, final answer.
Ensure the output is well-structured and ready for presentation.

DRAFT ANSWER:
{draft}

QUESTION: {query}

FINAL ANSWER:"""


class GovernedWorkflow(Orchestrator):
    def run(self, state: RunState) -> RunState:
        filtered_context = _retrieve_relevant_context(state.reference, state.query)

        draft_call = self.client.call(
            agent_role="drafter",
            prompt=DRAFTER_PROMPT.format(
                context=filtered_context,
                query=state.query,
            ),
            round_index=0,
        )
        state.calls.append(draft_call)

        format_call = self.client.call(
            agent_role="formatter",
            prompt=FORMATTER_PROMPT.format(
                draft=draft_call.response_text,
                query=state.query,
            ),
            round_index=0,
            want_logprobs=True,
        )
        state.calls.append(format_call)

        state.final_answer = format_call.response_text
        return state
