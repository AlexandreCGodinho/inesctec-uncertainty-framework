from src.patterns.base import Orchestrator
from src.schemas import RunState

class ReflexiveSingleAgent(Orchestrator):
    def run(self, state: RunState) -> RunState:
        # TODO: Implement Pattern 3 logic here
        
        # Example using the mock client
        prompt = f"Critique and answer: {state.query}\nContext: {state.reference}"
        call = self.client.call(agent_role="solo_reflexive", prompt=prompt, round_index=0)
        state.calls.append(call)
        
        state.final_answer = call.response_text
        return state
