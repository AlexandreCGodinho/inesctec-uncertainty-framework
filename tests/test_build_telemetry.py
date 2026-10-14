import pytest

from src.schemas import RunState, PatternID, LLMCall
from src.llm_client import MockLLMClient
from src.patterns.p1_governed import GovernedWorkflow
from src.patterns.p2_role_pipeline import RoleSeparatedPipeline
from src.metrics.build_telemetry import build_telemetry


def _make_state(pattern):
    return RunState(
        run_id="test", pattern=pattern,
        dataset="ifeval", query_id="q1",
        query="What is AI?",
        reference="AI is artificial intelligence. It powers many applications. "
        "Machine learning is a subset of AI.",
    )


class TestBuildTelemetryPattern1:
    def test_u_C_is_none(self):
        result = GovernedWorkflow(MockLLMClient()).run(
            _make_state(PatternID.GOVERNED_WORKFLOW)
        )
        t = build_telemetry(result, MockLLMClient())
        assert t.u_C is None

    def test_u_R_is_none(self):
        result = GovernedWorkflow(MockLLMClient()).run(
            _make_state(PatternID.GOVERNED_WORKFLOW)
        )
        t = build_telemetry(result, MockLLMClient())
        assert t.u_R is None

    def test_u_L_is_not_none(self):
        result = GovernedWorkflow(MockLLMClient()).run(
            _make_state(PatternID.GOVERNED_WORKFLOW)
        )
        t = build_telemetry(result, MockLLMClient())
        assert t.u_L is not None

    def test_U_surface_is_not_none(self):
        result = GovernedWorkflow(MockLLMClient()).run(
            _make_state(PatternID.GOVERNED_WORKFLOW)
        )
        t = build_telemetry(result, MockLLMClient())
        assert t.U_surface is not None

    def test_U_composite_is_not_none(self):
        result = GovernedWorkflow(MockLLMClient()).run(
            _make_state(PatternID.GOVERNED_WORKFLOW)
        )
        t = build_telemetry(result, MockLLMClient())
        assert t.U_composite is not None


class TestBuildTelemetryPattern2:
    def test_u_C_is_none(self):
        result = RoleSeparatedPipeline(MockLLMClient(), max_revisions=1).run(
            _make_state(PatternID.ROLE_SEPARATED_PIPELINE)
        )
        t = build_telemetry(result, MockLLMClient())
        assert t.u_C is None

    def test_u_R_is_not_none_with_revisions(self):
        result = RoleSeparatedPipeline(MockLLMClient(), max_revisions=1).run(
            _make_state(PatternID.ROLE_SEPARATED_PIPELINE)
        )
        t = build_telemetry(result, MockLLMClient())
        assert t.u_R is not None

    def test_U_surface_is_not_none(self):
        result = RoleSeparatedPipeline(MockLLMClient(), max_revisions=1).run(
            _make_state(PatternID.ROLE_SEPARATED_PIPELINE)
        )
        t = build_telemetry(result, MockLLMClient())
        assert t.U_surface is not None

    def test_U_composite_is_not_none(self):
        result = RoleSeparatedPipeline(MockLLMClient(), max_revisions=1).run(
            _make_state(PatternID.ROLE_SEPARATED_PIPELINE)
        )
        t = build_telemetry(result, MockLLMClient())
        assert t.U_composite is not None


class TestBuildTelemetryTruncation:
    def test_truncation_flag_true_when_any_call_truncated(self):
        state = _make_state(PatternID.GOVERNED_WORKFLOW)
        state.calls.append(
            LLMCall(agent_role="test", prompt="", response_text="", truncated=True)
        )
        t = build_telemetry(state, MockLLMClient())
        assert t.truncation_flag is True

    def test_truncation_flag_false_when_no_truncation(self):
        result = GovernedWorkflow(MockLLMClient()).run(
            _make_state(PatternID.GOVERNED_WORKFLOW)
        )
        t = build_telemetry(result, MockLLMClient())
        assert t.truncation_flag is False


def test_u_P_value_for_pattern1():
    result = GovernedWorkflow(MockLLMClient()).run(
        _make_state(PatternID.GOVERNED_WORKFLOW)
    )
    t = build_telemetry(result, MockLLMClient())
    assert t.u_P is not None
    assert t.u_P == pytest.approx(0.4921716224507743)  # Valor agregado pós-Hard Fail


def test_u_P_with_none_logprobs():
    state = _make_state(PatternID.GOVERNED_WORKFLOW)
    state.calls.append(
        LLMCall(agent_role="formatter", prompt="", response_text="test",
                logprobs=None)
    )
    state.final_answer = "This is a substantive final answer sentence with enough characters."
    t = build_telemetry(state, MockLLMClient())
    assert t.u_P == 0.0


def test_u_R_is_zero_with_single_round():
    result = RoleSeparatedPipeline(MockLLMClient(), max_revisions=0).run(
        _make_state(PatternID.ROLE_SEPARATED_PIPELINE)
    )
    t = build_telemetry(result, MockLLMClient())
    assert t.u_R == 0.0


def test_pattern3_u_R_is_zero_on_immediate_approval():
    state = _make_state(PatternID.REFLEXIVE_SINGLE_AGENT)
    state.calls = [
        LLMCall(agent_role="solo_reflexive", round_index=0, prompt="",
                response_text="Initial perfect draft."),
        LLMCall(agent_role="solo_reflexive", round_index=1, prompt="",
                response_text="VERDICT: approved. No changes needed."),
    ]
    state.final_answer = "Initial perfect draft."
    t = build_telemetry(state, MockLLMClient())
    assert t.u_R == 0.0


def test_u_R_uses_analyst_traces_not_reviewer_or_researcher():
    from src.metrics.u_R import compute_u_R
    state = _make_state(PatternID.ROLE_SEPARATED_PIPELINE)
    state.calls.append(LLMCall(
        agent_role="researcher", round_index=0, prompt="",
        response_text="RESEARCHER_FACTS_123 unique tokens xyz"))
    state.calls.append(LLMCall(
        agent_role="analyst", round_index=0, prompt="",
        response_text="ANALYST_DRAFT_ALPHA"))
    state.calls.append(LLMCall(
        agent_role="reviewer", round_index=0, prompt="",
        response_text="VERDICT: needs_revision REVIEWER_CRITIQUE_BETA xyz"))
    state.calls.append(LLMCall(
        agent_role="analyst", round_index=1, prompt="",
        response_text="ANALYST_REVISED_GAMMA"))
    state.calls.append(LLMCall(
        agent_role="reviewer", round_index=1, prompt="",
        response_text="VERDICT: approved REVIEWER_FINAL_DELTA xyz"))
    state.final_answer = "ANALYST_REVISED_GAMMA"
    t = build_telemetry(state, MockLLMClient())
    expected = compute_u_R(["ANALYST_DRAFT_ALPHA", "ANALYST_REVISED_GAMMA"])
    assert t.u_R == pytest.approx(expected)


def test_truncation_flag_true_from_early_call():
    state = _make_state(PatternID.GOVERNED_WORKFLOW)
    state.calls.append(
        LLMCall(agent_role="drafter", prompt="", response_text="draft",
                truncated=True)
    )
    state.calls.append(
        LLMCall(agent_role="formatter", prompt="", response_text="final",
                truncated=False)
    )
    state.final_answer = "final"
    t = build_telemetry(state, MockLLMClient())
    assert t.truncation_flag is True


# Pattern 3 tests — synthetic RunState (orchestrator not yet merged)
# Call structure mirrors p3_reflexive.py: round 0 = draft; round r >= 1 =
# critique first, revision second (if not approved).

def test_pattern3_logprobs_from_last_round():
    state = _make_state(PatternID.REFLEXIVE_SINGLE_AGENT)
    state.calls = [
        LLMCall(agent_role="solo_reflexive", round_index=0, prompt="",
                response_text="draft", logprobs=[-0.5]),
        LLMCall(agent_role="solo_reflexive", round_index=1, prompt="",
                response_text="VERDICT: needs_revision", logprobs=[-0.9]),
        LLMCall(agent_role="solo_reflexive", round_index=1, prompt="",
                response_text="revised", logprobs=[-0.2]),
    ]
    state.final_answer = "This is a revised final answer sentence with enough characters."
    t = build_telemetry(state, MockLLMClient())
    assert t.u_P is not None


def test_pattern3_u_R_computed_with_multiple_rounds():
    state = _make_state(PatternID.REFLEXIVE_SINGLE_AGENT)
    state.calls = [
        LLMCall(agent_role="solo_reflexive", round_index=0, prompt="",
                response_text="Paris is the capital."),
        LLMCall(agent_role="solo_reflexive", round_index=1, prompt="",
                response_text="VERDICT: needs_revision. Add country name."),
        LLMCall(agent_role="solo_reflexive", round_index=1, prompt="",
                response_text="Paris is the capital of France."),
    ]
    state.final_answer = "Paris is the capital of France."
    t = build_telemetry(state, MockLLMClient())
    assert t.u_R is not None


# Pattern 4 tests — synthetic RunState (orchestrator not yet merged)
# Call structure mirrors p4_deliberation.py: K analysts at round 0, judge at round 1.

def test_pattern4_u_C_computed_from_symmetric_agents():
    """Multi-round: u_C must use only the LAST round's analyst answers."""
    from src.metrics.u_C import compute_u_C
    state = _make_state(PatternID.MULTI_PERSPECTIVE_DELIBERATION)
    # Round 0: analysts give initial answers, Judge says needs_revision
    state.calls = [
        LLMCall(agent_role="analyst_1", round_index=0, prompt="",
                response_text="Answer A0"),
        LLMCall(agent_role="analyst_2", round_index=0, prompt="",
                response_text="Answer B0"),
        LLMCall(agent_role="analyst_3", round_index=0, prompt="",
                response_text="Answer C0"),
        LLMCall(agent_role="judge", round_index=0, prompt="",
                response_text="VERDICT: needs_revision. More detail needed."),
        # Round 1: analysts revise, Judge approves
        LLMCall(agent_role="analyst_1", round_index=1, prompt="",
                response_text="Answer A1"),
        LLMCall(agent_role="analyst_2", round_index=1, prompt="",
                response_text="Answer B1"),
        LLMCall(agent_role="analyst_3", round_index=1, prompt="",
                response_text="Answer C1"),
        LLMCall(agent_role="judge", round_index=1, prompt="",
                response_text="VERDICT: approved. Consensus reached."),
    ]
    state.final_answer = "Consensus reached."
    t = build_telemetry(state, MockLLMClient())
    assert t.u_C is not None
    expected = compute_u_C(["Answer A1", "Answer B1", "Answer C1"])
    assert t.u_C == pytest.approx(expected)


def test_pattern4_u_P_uses_last_round_analyst_logprobs():
    """u_P for P4 averages logprobs from the last round's analysts."""
    from src.metrics.u_P import compute_u_P
    from src.metrics.u_F import compute_u_F
    state = _make_state(PatternID.MULTI_PERSPECTIVE_DELIBERATION)
    state.calls = [
        LLMCall(agent_role="analyst_1", round_index=0, prompt="",
                response_text="A", logprobs=[-0.5, -0.3]),
        LLMCall(agent_role="analyst_2", round_index=0, prompt="",
                response_text="B", logprobs=[-0.1, -0.2]),
        LLMCall(agent_role="judge", round_index=0, prompt="",
                response_text="VERDICT: approved."),
    ]
    state.final_answer = "This is an approved final consensus answer with many words."
    t = build_telemetry(state, MockLLMClient())
    assert t.u_P is not None
    averaged = [sum(vals) / len(vals) for vals in zip(*[[-0.5, -0.3], [-0.1, -0.2]])]
    expected = compute_u_P(averaged, compute_u_F("This is an approved final consensus answer with many words.", state.reference, MockLLMClient()))
    assert t.u_P == pytest.approx(expected)


def test_pattern4_u_R_on_immediate_consensus():
    """Single round (immediate approval) → u_R == 0.0 per Option B."""
    state = _make_state(PatternID.MULTI_PERSPECTIVE_DELIBERATION)
    state.calls = [
        LLMCall(agent_role="analyst_1", round_index=0, prompt="",
                response_text="Ans1"),
        LLMCall(agent_role="analyst_2", round_index=0, prompt="",
                response_text="Ans2"),
        LLMCall(agent_role="analyst_3", round_index=0, prompt="",
                response_text="Ans3"),
        LLMCall(agent_role="judge", round_index=0, prompt="",
                response_text="VERDICT: approved."),
    ]
    state.final_answer = "approved"
    t = build_telemetry(state, MockLLMClient())
    assert t.u_R == 0.0


def test_pattern4_u_R_on_multi_round_refinement():
    """Multiple Judge calls across rounds → u_R computed from judge traces."""
    from src.metrics.u_R import compute_u_R
    state = _make_state(PatternID.MULTI_PERSPECTIVE_DELIBERATION)
    state.calls = [
        LLMCall(agent_role="analyst_1", round_index=0, prompt="",
                response_text="Ans1 r0"),
        LLMCall(agent_role="analyst_2", round_index=0, prompt="",
                response_text="Ans2 r0"),
        LLMCall(agent_role="judge", round_index=0, prompt="",
                response_text="VERDICT: needs_revision. X is missing."),
        LLMCall(agent_role="analyst_1", round_index=1, prompt="",
                response_text="Ans1 r1"),
        LLMCall(agent_role="analyst_2", round_index=1, prompt="",
                response_text="Ans2 r1"),
        LLMCall(agent_role="judge", round_index=1, prompt="",
                response_text="VERDICT: approved. All good."),
    ]
    state.final_answer = "All good."
    t = build_telemetry(state, MockLLMClient())
    assert t.u_R is not None
    assert t.u_R > 0.0
    expected = compute_u_R([
        "VERDICT: needs_revision. X is missing.",
        "VERDICT: approved. All good.",
    ])
    assert t.u_R == pytest.approx(expected)
