from __future__ import annotations
import time
import random
import logging
from pathlib import Path
from src.schemas import LLMCall

logger = logging.getLogger(__name__)


class LLMConnectionError(RuntimeError):
    """Server unreachable after N retries."""

class LLMTimeoutError(RuntimeError):
    """Persistent timeout after N retries."""


_MAX_RETRIES = 3
_BASE_DELAY = 1.0
_MAX_DELAY = 10.0

_CONFIG_CACHE = None


def _load_experiment_config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        import yaml
        path = Path(__file__).resolve().parents[1] / "config" / "experiment.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"experiment.yaml not found at {path}. "
                "Make sure you're running from the repo root or that "
                "config/experiment.yaml exists."
            )
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"experiment.yaml at {path} did not parse to a dict."
            )
        for required_key in ("model_id", "base_seed"):
            if required_key not in data:
                raise KeyError(
                    f"experiment.yaml is missing required key '{required_key}'."
                )
        _CONFIG_CACHE = data
    return _CONFIG_CACHE


class LLMClient:
    LITELLM_PREFIX = "groq/"

    def __init__(self, api_base: str | None = None,
                 model: str | None = None,
                 seed: int | None = None):
        cfg = _load_experiment_config()
        
        import os
        from dotenv import load_dotenv
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        
        db = "https://api.groq.com/openai/v1"
        self.api_base = db if api_base is None else api_base
        
        # We will use Llama 3.1 8B on Groq because Gemma was decommissioned
        raw_model = "llama-3.1-8b-instant"
        if not raw_model.startswith(self.LITELLM_PREFIX):
            raw_model = self.LITELLM_PREFIX + raw_model
        self.model = raw_model
        
        self.seed = cfg["base_seed"] if seed is None else seed
        self._timeout = cfg.get("request_timeout_seconds", 60)
        self._completion = None

    @property
    def _completion_fn(self):
        if self._completion is None:
            from litellm import completion
            self._completion = completion
        return self._completion

    def _call_with_retry(self, **kwargs) -> object:
        from litellm.exceptions import (
            APIConnectionError,
            RateLimitError,
            ServiceUnavailableError,
            Timeout as LiteLLMTimeout,
        )
        retryable = (APIConnectionError, RateLimitError, ServiceUnavailableError, LiteLLMTimeout)

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._completion_fn(**kwargs)
            except retryable as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    delay = min(_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5), _MAX_DELAY)
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1, _MAX_RETRIES + 1, e, delay,
                    )
                    time.sleep(delay)
                else:
                    break
            except Exception as e:
                raise

        if isinstance(last_error, LiteLLMTimeout):
            raise LLMTimeoutError(
                f"LLM call timed out after {_MAX_RETRIES + 1} attempts. "
                f"Last error: {last_error}"
            ) from last_error
        raise LLMConnectionError(
            f"LLM call failed after {_MAX_RETRIES + 1} attempts. "
            f"Last error: {last_error}"
        ) from last_error

    def call(self, agent_role: str, prompt: str, round_index: int = 0,
             want_logprobs: bool = False,
             timeout_seconds: float | None = None) -> LLMCall:
        timeout = timeout_seconds if timeout_seconds is not None else self._timeout
        start = time.perf_counter()

        # Groq's Llama 3.1 does not support logprobs
        want_logprobs = False

        r = self._call_with_retry(
            model=self.model, api_base=self.api_base, api_key=self.api_key,
            messages=[{"role": "user", "content": prompt}],
            logprobs=want_logprobs, top_logprobs=5 if want_logprobs else None,
            temperature=0.7, seed=self.seed, timeout=timeout,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        text = r.choices[0].message.content or ""
        finish_reason = r.choices[0].finish_reason
        truncated = finish_reason == "length"

        lps = None
        if want_logprobs and getattr(r.choices[0], "logprobs", None):
            lps = [t.logprob for t in r.choices[0].logprobs.content]

        usage = getattr(r, "usage", None)
        tokens_prompt = usage.prompt_tokens if usage else 0
        tokens_completion = usage.completion_tokens if usage else 0
        
        # Throttle to stay safely under Groq's 6000 TPM (Tokens Per Minute) limit
        # The deliberation agent uses ~1500 tokens, so 4 requests per minute is the maximum safe limit.
        time.sleep(16)

        return LLMCall(
            agent_role=agent_role, round_index=round_index,
            prompt=prompt, response_text=text, logprobs=lps,
            truncated=truncated, latency_ms=round(elapsed_ms, 1),
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
        )


class MockLLMClient(LLMClient):
    def __init__(self):
        self.seed = 0

    def call(self, agent_role, prompt, round_index=0, want_logprobs=False,
             timeout_seconds=None):
        return LLMCall(agent_role=agent_role, round_index=round_index,
                       prompt=prompt,
                       response_text=f"[mock:{agent_role}:r{round_index}]",
                       logprobs=[-0.1, -0.2] if want_logprobs else None,
                       latency_ms=0.1, tokens_prompt=10, tokens_completion=5)


def check_server_available(api_base: str, timeout: float = 5.0) -> bool:
    try:
        from litellm import completion
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        model = "groq/llama-3.1-8b-instant"
        api_key = os.getenv("GROQ_API_KEY")
        
        r = completion(
            model=model, api_base="https://api.groq.com/openai/v1", api_key=api_key,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1, timeout=timeout,
        )
        return bool(r.choices)
    except Exception as e:
        logger.error(f"Groq API connection failed: {e}")
        return False
