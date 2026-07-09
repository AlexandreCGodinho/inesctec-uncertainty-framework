import yaml
import os
from src.patterns.base import Orchestrator
from src.schemas import RunState

class MultiPerspectiveDeliberation(Orchestrator):
    def run(self, state: RunState) -> RunState:
        # Load seeds from config
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'experiment.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        seeds = config.get('pattern4_agent_seeds', [1234, 5678, 9012])
        num_agents = config.get('pattern4_num_agents', 3)
        
        # 1. Parallel Generation by Symmetric Agents
        analyst_responses = []
        original_seed = self.client.seed
        
        for i in range(num_agents):
            seed = seeds[i] if i < len(seeds) else original_seed + i
            self.client.seed = seed
            
            prompt = f"Analyze the query based on the context.\nQuery: {state.query}\nContext: {state.reference}"
            call = self.client.call(agent_role=f"analyst_{i+1}", prompt=prompt, round_index=0)
            state.calls.append(call)
            analyst_responses.append(call.response_text)
            
        # Restore original seed
        self.client.seed = original_seed
        
        # 2. Consensus Synthesis by Judge
        judge_prompt = (
            f"Synthesize the following analyses into a final consensus answer.\n"
            f"Query: {state.query}\nContext: {state.reference}\n\n"
        )
        for i, resp in enumerate(analyst_responses):
            judge_prompt += f"Analysis {i+1}:\n{resp}\n\n"
            
        judge_call = self.client.call(agent_role="judge", prompt=judge_prompt, round_index=1)
        state.calls.append(judge_call)
        
        state.final_answer = judge_call.response_text
        return state
