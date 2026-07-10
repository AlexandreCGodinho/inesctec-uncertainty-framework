from src.schemas import PatternID, LLMCall, RunState, Telemetry


def test_pattern_id_values():
    assert PatternID.GOVERNED_WORKFLOW == 1
    assert PatternID.ROLE_SEPARATED_PIPELINE == 2
    assert PatternID.REFLEXIVE_SINGLE_AGENT == 3
    assert PatternID.MULTI_PERSPECTIVE_DELIBERATION == 4


def test_llm_call_creation():
    call = LLMCall(agent_role="solo", prompt="Hello", response_text="Hi")
    assert call.agent_role == "solo"
    assert call.round_index == 0
    assert call.logprobs is None
    assert call.truncated is False


def test_run_state_rounds_property():
    state = RunState(
        run_id="test-001", pattern=PatternID.GOVERNED_WORKFLOW,
        dataset="ifeval", query_id="q1", query="test", reference="ref",
    )
    assert state.rounds == 1
    state.calls.append(LLMCall(agent_role="solo", prompt="", response_text="", round_index=2))
    assert state.rounds == 3


def test_telemetry_defaults():
    t = Telemetry(run_id="t1", pattern=PatternID.GOVERNED_WORKFLOW)
    assert t.u_L is None
    assert t.truncation_flag is False
