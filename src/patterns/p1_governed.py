from src.patterns.base import Orchestrator
from src.schemas import RunState, LLMCall


GOVERNED_PROMPT = """You are a precise research assistant. Answer ONLY from the provided context.
If not in context, say "Not found in documents".

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


class GovernedWorkflow(Orchestrator):
    def run(self, state: RunState) -> RunState:
        prompt = GOVERNED_PROMPT.format(
            context=state.reference,
            query=state.query,
        )
        call = self.client.call(
            agent_role="solo",
            prompt=prompt,
            round_index=0,
            want_logprobs=False,
        )
        state.calls.append(call)
        state.final_answer = call.response_text
        return state
