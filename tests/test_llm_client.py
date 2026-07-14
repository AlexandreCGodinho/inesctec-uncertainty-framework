from src.llm_client import MockLLMClient


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
