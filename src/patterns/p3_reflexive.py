from src.patterns.base import Orchestrator
from src.schemas import RunState

class ReflexiveSingleAgent(Orchestrator):
    def run(self, state: RunState) -> RunState:
        # 1. Draft Generation
        draft_prompt = f"Answer the query based on the context.\nQuery: {state.query}\nContext: {state.reference}"
        draft_call = self.client.call(agent_role="solo_reflexive", prompt=draft_prompt, round_index=0)
        state.calls.append(draft_call)
        current_draft = draft_call.response_text

        # 2. Self-Critique Loop (max 2 critique rounds)
        max_critique_rounds = 2
        for r in range(1, max_critique_rounds + 1):
            critique_prompt = (
                f"Review the following draft for the query: '{state.query}' given the context: '{state.reference}'.\n"
                f"Draft: {current_draft}\n"
                f"Are there factual errors or omissions? Output exactly 'Validated' if it is perfect, "
                f"otherwise explain the errors and end with 'Errors Found'."
            )
            critique_call = self.client.call(agent_role="solo_reflexive", prompt=critique_prompt, round_index=r)
            state.calls.append(critique_call)
            
            if "Validated" in critique_call.response_text:
                break
                
            # If errors found, generate a new draft (same round index for the revision call)
            revision_prompt = (
                f"Revise the draft based on the critique.\n"
                f"Query: {state.query}\nContext: {state.reference}\n"
                f"Draft: {current_draft}\nCritique: {critique_call.response_text}"
            )
            revision_call = self.client.call(agent_role="solo_reflexive", prompt=revision_prompt, round_index=r)
            state.calls.append(revision_call)
            current_draft = revision_call.response_text

        state.final_answer = current_draft
        return state
