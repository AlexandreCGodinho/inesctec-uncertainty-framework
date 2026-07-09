from litellm import completion
from src.schemas import LLMCall


class LLMClient:
    def __init__(self, api_base: str = "http://localhost:8000/v1",
                 model: str = "openai/google/gemma-4-12b-it",
                 seed: int = 1234):
        self.api_base, self.model, self.seed = api_base, model, seed

    def call(self, agent_role: str, prompt: str, round_index: int = 0,
             want_logprobs: bool = False) -> LLMCall:
        r = completion(
            model=self.model, api_base=self.api_base, api_key="not-needed",
            messages=[{"role": "user", "content": prompt}],
            logprobs=want_logprobs, top_logprobs=5 if want_logprobs else None,
            temperature=0.7, seed=self.seed,
        )
        text = r.choices[0].message.content
        lps = None
        if want_logprobs and r.choices[0].logprobs:
            lps = [t.logprob for t in r.choices[0].logprobs.content]
        return LLMCall(agent_role=agent_role, round_index=round_index,
                       prompt=prompt, response_text=text, logprobs=lps)


class MockLLMClient(LLMClient):
    """GPU-free client for local development and unit tests."""
    def __init__(self):  # no server needed
        self.seed = 0

    def call(self, agent_role, prompt, round_index=0, want_logprobs=False):
        return LLMCall(agent_role=agent_role, round_index=round_index,
                       prompt=prompt,
                       response_text=f"[mock:{agent_role}:r{round_index}]",
                       logprobs=[-0.1, -0.2] if want_logprobs else None)
