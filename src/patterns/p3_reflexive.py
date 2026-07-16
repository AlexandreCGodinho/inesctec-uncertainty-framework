import yaml
import os
from src.patterns.base import Orchestrator
from src.schemas import RunState
from src.llm_client import LLMClient
from src.patterns.p2_role_pipeline import _is_approved

class ReflexiveSingleAgent(Orchestrator):
    def __init__(self, client: LLMClient, max_iterations=None):
        super().__init__(client)
        self.max_iterations = max_iterations

    def run(self, state: RunState) -> RunState:
        # Load config for max_iterations fallback if None
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'experiment.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        max_iters = self.max_iterations if self.max_iterations is not None else config.get('pattern3_max_iterations', 3)

        # 1. Draft Generation
        draft_prompt = (
            f"You are an expert analyst. Answer the query strictly based on the context.\n"
            f"Query: {state.query}\n"
            f"Context: {state.reference}\n"
            f"Provide a clear, definitive draft."
        )
        draft_call = self.client.call(agent_role="solo_reflexive", prompt=draft_prompt, round_index=0)
        state.calls.append(draft_call)
        current_draft = draft_call.response_text

        # 2. Self-Critique Loop
        for r in range(1, max_iters):
            critique_prompt = (
                f"Review the following draft for the query: '{state.query}' given the context: '{state.reference}'.\n"
                f"Draft: {current_draft}\n\n"
                f"Critique the draft for any factual errors, omissions, or formatting issues.\n"
                f"If the draft is perfect and needs no changes, output exactly 'VERDICT: approved' on a new line.\n"
                f"Otherwise, output 'VERDICT: needs_revision' and explain what needs to be fixed."
            )
            critique_call = self.client.call(agent_role="solo_reflexive", prompt=critique_prompt, round_index=r)
            state.calls.append(critique_call)
            
            if _is_approved(critique_call.response_text):
                break
                
            # If errors found, generate a new draft
            revision_prompt = (
                f"Revise the draft based on the critique.\n"
                f"Query: {state.query}\nContext: {state.reference}\n\n"
                f"Original Draft:\n{current_draft}\n\n"
                f"Critique:\n{critique_call.response_text}\n\n"
                f"Provide the revised, final draft that addresses all critique points."
            )
            revision_call = self.client.call(agent_role="solo_reflexive", prompt=revision_prompt, round_index=r)
            state.calls.append(revision_call)
            current_draft = revision_call.response_text

        state.final_answer = current_draft
        return state
