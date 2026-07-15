import yaml
import os
from src.patterns.base import Orchestrator
from src.schemas import RunState
from src.llm_client import LLMClient
from src.patterns.p2_role_pipeline import _is_approved

AGENT_PROMPT = (
    "You are Agent {agent_id}. Analyze the query based on the context.\n"
    "Query: {query}\n"
    "Context: {context}\n"
    "Provide a detailed, step-by-step logical reasoning to reach your conclusion."
)

AGENT_REVISION_PROMPT = (
    "You are Agent {agent_id}. Revise your analysis based on the Judge's feedback.\n"
    "Query: {query}\n"
    "Context: {context}\n\n"
    "Your Previous Analysis:\n{previous_analysis}\n\n"
    "Judge Feedback:\n{judge_feedback}\n\n"
    "Provide a refined logical reasoning to reach a better consensus."
)

JUDGE_PROMPT = (
    "You are the Judge. Evaluate the following analyses.\n"
    "Query: {query}\nContext: {context}\n\n"
    "{analyses}\n"
    "If the analyses reach a solid consensus or are accurate enough, output exactly 'VERDICT: approved' on a new line, followed by the final definitive answer.\n"
    "If they contradict each other or have errors, output 'VERDICT: needs_revision' and provide feedback for the analysts to improve."
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
        max_iters = config.get('pattern4_max_iterations', 2)
        
        original_seed = self.client.seed
        analyst_responses = [""] * num_agents
        judge_feedback = ""
        
        for r in range(max_iters + 1):
            # 1. Parallel Generation by Symmetric Agents
            for i in range(num_agents):
                seed = seeds[i] if i < len(seeds) else original_seed + i
                self.client.seed = seed
                
                if r == 0:
                    prompt = AGENT_PROMPT.format(agent_id=i, query=state.query, context=state.reference)
                else:
                    prompt = AGENT_REVISION_PROMPT.format(
                        agent_id=i, query=state.query, context=state.reference,
                        previous_analysis=analyst_responses[i],
                        judge_feedback=judge_feedback
                    )
                    
                call = self.client.call(agent_role=f"analyst_{i+1}", prompt=prompt, round_index=r, want_logprobs=True)
                state.calls.append(call)
                analyst_responses[i] = call.response_text
                
            # Restore original seed
            self.client.seed = original_seed
            
            # 2. Consensus Synthesis by Judge
            analyses_text = ""
            for i, resp in enumerate(analyst_responses):
                analyses_text += f"Analysis {i+1}:\n{resp}\n\n"
                
            prompt_judge = JUDGE_PROMPT.format(query=state.query, context=state.reference, analyses=analyses_text)
            judge_call = self.client.call(agent_role="judge", prompt=prompt_judge, round_index=r)
            state.calls.append(judge_call)
            
            if _is_approved(judge_call.response_text):
                break
            else:
                judge_feedback = judge_call.response_text
        
        state.final_answer = judge_call.response_text
        return state
