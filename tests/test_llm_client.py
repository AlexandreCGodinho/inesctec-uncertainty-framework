import sys
import types

import pytest
from src.llm_client import (
    MockLLMClient,
    LLMClient,
    check_server_available,
    LLMTimeoutError,
    LLMConnectionError,
)


# ---------------------------------------------------------------------------
# Fake litellm module — installed once so _call_with_retry can import from
# litellm.exceptions without the real library.
# ---------------------------------------------------------------------------

def _install_fake_litellm():
    """Insert a fake ``litellm`` + ``litellm.exceptions`` into sys.modules.

    The mock ``completion()`` function is replaced per-test by setting
    ``client._completion`` (for LLMClient tests) or by assigning
    ``litellm.completion`` (for check_server_available tests).
    """
    if "litellm" in sys.modules:
        return  # already installed by a previous test

    litellm = types.ModuleType("litellm")
    exceptions = types.ModuleType("litellm.exceptions")

    class _APIError(Exception):
        def __init__(self, message, *args, **kwargs):
            super().__init__(message)

    class APIConnectionError(_APIError):
        pass
    class RateLimitError(_APIError):
        pass
    class ServiceUnavailableError(_APIError):
        pass
    class Timeout(_APIError):
        pass
    class BadRequestError(_APIError):
        pass

    exceptions.APIConnectionError = APIConnectionError
    exceptions.RateLimitError = RateLimitError
    exceptions.ServiceUnavailableError = ServiceUnavailableError
    exceptions.Timeout = Timeout
    exceptions.BadRequestError = BadRequestError

    litellm.exceptions = exceptions
    litellm.completion = None  # replaced per-test

    sys.modules["litellm"] = litellm
    sys.modules["litellm.exceptions"] = exceptions


_install_fake_litellm()

# Grab references to the fake exception classes so tests can raise them
import litellm  # noqa: E402 — our fake module
from litellm.exceptions import (  # noqa: E402
    APIConnectionError,
    BadRequestError,
    Timeout as LiteLLMTimeout,
)


# ---------------------------------------------------------------------------
# Existing MockLLMClient tests
# ---------------------------------------------------------------------------

def test_mock_client_returns_llm_call():
    client = MockLLMClient()
    result = client.call("solo", "Hello")
    assert result.agent_role == "solo"
    assert result.response_text == "[mock:solo:r0]"
    assert result.logprobs is None


def test_mock_client_with_logprobs():
    client = MockLLMClient()
    result = client.call("judge", "Test", want_logprobs=True)
    assert result.logprobs == [-0.1, -0.2]


def test_mock_client_tracking_fields():
    client = MockLLMClient()
    result = client.call("analyst", "Prompt")
    assert result.latency_ms == 0.1
    assert result.tokens_prompt == 10
    assert result.tokens_completion == 5


# ---------------------------------------------------------------------------
# LLMClient retry / timeout / tracking tests
# ---------------------------------------------------------------------------

class TestLLMRetryAndTimeout:
    """Set ``client._completion`` to bypass the lazy-import in ``_completion_fn``.

    ``_call_with_retry`` uses the fake ``litellm.exceptions`` installed by
    ``_install_fake_litellm``.
    """

    @staticmethod
    def _fake_usage(prompt=15, completion=3):
        return type("FakeUsage", (), {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
        })()

    @staticmethod
    def _fake_choice(content="hello", finish="stop", logprobs=None):
        return type("FakeChoice", (), {
            "message": type("FakeMsg", (), {"content": content})(),
            "finish_reason": finish,
            "logprobs": logprobs,
        })()

    @staticmethod
    def _fake_response(choices, usage=None):
        return type("FakeResponse", (), {
            "choices": choices,
            "usage": usage,
        })()

    def _make_client(self, mock_fn):
        client = LLMClient(api_base="http://fake:8000/v1", model="test-model", seed=0)
        client._completion = mock_fn
        return client

    def test_timeout_after_max_retries(self):
        call_count = [0]
        def mock_completion(**kwargs):
            call_count[0] += 1
            raise LiteLLMTimeout("simulated timeout", model="test", llm_provider="openai")

        client = self._make_client(mock_completion)
        with pytest.raises(LLMTimeoutError, match="timed out"):
            client.call(agent_role="test", prompt="ping")
        assert call_count[0] == 4

    def test_retry_succeeds_on_second_attempt(self):
        call_count = [0]
        def mock_completion(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise APIConnectionError("simulated", model="test", llm_provider="openai")
            return self._fake_response(
                choices=[self._fake_choice(content="success")],
                usage=self._fake_usage(),
            )

        client = self._make_client(mock_completion)
        result = client.call(agent_role="agent", prompt="hello")
        assert result.response_text == "success"
        assert call_count[0] == 2

    def test_non_retryable_exception_passthrough(self):
        call_count = [0]
        def mock_completion(**kwargs):
            call_count[0] += 1
            raise BadRequestError("bad request", model="test", llm_provider="openai")

        client = self._make_client(mock_completion)
        with pytest.raises(BadRequestError):
            client.call(agent_role="test", prompt="ping")
        assert call_count[0] == 1

    def test_latency_and_tokens_tracked(self):
        def mock_completion(**kwargs):
            return self._fake_response(
                choices=[self._fake_choice(content="tracked")],
                usage=self._fake_usage(prompt=20, completion=7),
            )

        client = self._make_client(mock_completion)
        result = client.call(agent_role="test", prompt="track me")
        assert result.latency_ms > 0
        assert result.tokens_prompt == 20
        assert result.tokens_completion == 7

    def test_timeout_override_per_call(self):
        captured_timeout = [None]
        def mock_completion(**kwargs):
            captured_timeout[0] = kwargs.get("timeout")
            return self._fake_response(
                choices=[self._fake_choice(content="ok")],
            )

        client = self._make_client(mock_completion)
        client.call(agent_role="test", prompt="ping", timeout_seconds=120)
        assert captured_timeout[0] == 120

    def test_truncated_flag_on_length_finish(self):
        def mock_completion(**kwargs):
            return self._fake_response(
                choices=[self._fake_choice(content="partial", finish="length")],
            )

        client = self._make_client(mock_completion)
        result = client.call(agent_role="test", prompt="long prompt")
        assert result.truncated is True

    def test_retry_connection_error_raises_connection_error(self):
        call_count = [0]
        def mock_completion(**kwargs):
            call_count[0] += 1
            raise APIConnectionError("server down", model="test", llm_provider="openai")

        client = self._make_client(mock_completion)
        with pytest.raises(LLMConnectionError, match="failed after"):
            client.call(agent_role="test", prompt="ping")
        assert call_count[0] == 4


# ---------------------------------------------------------------------------
# check_server_available tests — assign litellm.completion on the fake module
# ---------------------------------------------------------------------------

class TestCheckServerAvailable:

    @staticmethod
    def _fake_choice(content="ok"):
        return type("FakeChoice", (), {
            "message": type("FakeMsg", (), {"content": content})(),
        })()

    @staticmethod
    def _fake_response(choices):
        return type("FakeResponse", (), {"choices": choices})()

    def test_check_server_available_true(self):
        def mock_completion(**kwargs):
            return self._fake_response(choices=[self._fake_choice()])

        litellm.completion = mock_completion
        assert check_server_available("http://fake:8000/v1") is True

    def test_check_server_available_false(self):
        def mock_completion(**kwargs):
            raise RuntimeError("server down")

        litellm.completion = mock_completion
        assert check_server_available("http://fake:8000/v1") is False
