from __future__ import annotations
from pathlib import Path
from src.schemas import LLMCall

_CONFIG_CACHE = None


def _load_experiment_config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        import yaml
        path = Path(__file__).resolve().parents[1] / "config" / "experiment.yaml"
        with open(path) as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE


class LLMClient:
    LITELLM_PREFIX = "openai/"

    def __init__(self, api_base: str | None = None,
                 model: str | None = None,
                 seed: int | None = None):
        cfg = _load_experiment_config()
        db = "http://localhost:8000/v1"
        self.api_base = db if api_base is None else api_base
        raw_model = model if model is not None else cfg["model_id"]
        if not raw_model.startswith(self.LITELLM_PREFIX):
            raw_model = self.LITELLM_PREFIX + raw_model
        self.model = raw_model
        self.seed = cfg["base_seed"] if seed is None else seed
        self._completion = None

    @property
    def _completion_fn(self):
        if self._completion is None:
            from litellm import completion
            self._completion = completion
        return self._completion

    def call(self, agent_role: str, prompt: str, round_index: int = 0,
             want_logprobs: bool = False) -> LLMCall:
        r = self._completion_fn(
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
    def __init__(self):
        self.seed = 0

    def call(self, agent_role, prompt, round_index=0, want_logprobs=False):
        return LLMCall(agent_role=agent_role, round_index=round_index,
                       prompt=prompt,
                       response_text=f"[mock:{agent_role}:r{round_index}]",
                       logprobs=[-0.1, -0.2] if want_logprobs else None)
