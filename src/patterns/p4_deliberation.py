import yaml
import os
from src.patterns.base import Orchestrator
from src.schemas import RunState
from src.llm_client import LLMClient

AGENT_PROMPT = (
    "You are Agent {agent_id}. Analyze the query based on the context.\n"
    "Query: {query}\n"
    "Context: {context}\n"
    "Provide a detailed, step-by-step logical reasoning to reach your conclusion."
)

class MultiPerspectiveDeliberation(Orchestrator):
    def __init__(self, client: LLMClient, num_agents=None):
        super().__init__(client)
        self.num_agents = num_agents

    def run(self, state: RunState) -> RunState:
        # Load config
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'experiment.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        seeds = config.get('pattern4_agent_seeds', [1234, 5678, 9012])
        num_agents = self.num_agents if self.num_agents is not None else config.get('pattern4_num_agents', 3)
        context_ceiling = config.get('pattern4_context_ceiling_tokens', 8000)
        
        # 1. Parallel Generation by Symmetric Agents
        analyst_responses = []
        original_seed = self.client.seed
        
        for i in range(num_agents):
            seed = seeds[i] if i < len(seeds) else original_seed + i
            self.client.seed = seed
            
            prompt = AGENT_PROMPT.format(agent_id=i, query=state.query, context=state.reference)
            call = self.client.call(agent_role=f"analyst_{i+1}", prompt=prompt, round_index=0)
            state.calls.append(call)
            analyst_responses.append(call.response_text)
            
        # Restore original seed
        self.client.seed = original_seed
        
        # 2. Consensus Synthesis by Judge
        judge_prompt = (
            f"You are the Judge. Synthesize the following analyses into a final consensus answer.\n"
            f"Query: {state.query}\nContext: {state.reference}\n\n"
        )
        for i, resp in enumerate(analyst_responses):
            judge_prompt += f"Analysis {i+1}:\n{resp}\n\n"
        judge_prompt += "Final decision: Provide a clear, definitive answer based on the analyses."
            
        judge_call = self.client.call(agent_role="judge", prompt=judge_prompt, round_index=1)
        state.calls.append(judge_call)
        
        state.final_answer = judge_call.response_text
        return state
