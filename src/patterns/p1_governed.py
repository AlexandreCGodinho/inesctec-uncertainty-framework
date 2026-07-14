import re
import string

from src.patterns.base import Orchestrator
from src.schemas import RunState

# NOTE: This retrieval step is pure Python (no LLM call), so it does
# NOT produce an LLMCall and is NOT added to state.calls. Only the
# two downstream LLM calls (Draft, Format) are recorded in state.calls.

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "but", "if", "do", "does", "did", "not", "it", "its", "so", "than",
}


def _extract_query_terms(query: str) -> set[str]:
    """Strip punctuation, lowercase, and remove stopwords from the query."""
    no_punct = query.translate(str.maketrans("", "", string.punctuation))
    words = no_punct.lower().split()
    return {w for w in words if w not in _STOPWORDS and w}


def _retrieve_relevant_context(context: str, query: str) -> str:
    query_terms = _extract_query_terms(query)
    if not query_terms:
        return context

    sentences = re.split(r"(?<=[.!?])\s+", context.strip())
    relevant = []
    for sentence in sentences:
        sentence_words = set(
            sentence.translate(str.maketrans("", "", string.punctuation)).lower().split()
        )
        if query_terms & sentence_words:
            relevant.append(sentence)

    return " ".join(relevant) if relevant else context


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
        )
        state.calls.append(format_call)

        state.final_answer = format_call.response_text
        return state
