from src.patterns.base import Orchestrator
from src.schemas import RunState

class MultiPerspectiveDeliberation(Orchestrator):
    def run(self, state: RunState) -> RunState:
        # TODO: Implement Pattern 4 logic here
        # Note: Must use exactly three symmetric agents (K=3) from config
        
        # Example
        call = self.client.call(agent_role="judge", prompt=state.query, round_index=0)
        state.calls.append(call)
        
        state.final_answer = call.response_text
        return state
