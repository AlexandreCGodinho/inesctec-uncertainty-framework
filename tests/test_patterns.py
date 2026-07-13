import pytest

from src.schemas import RunState, PatternID, LLMCall
from src.llm_client import MockLLMClient
from src.patterns.p1_governed import GovernedWorkflow, _retrieve_relevant_context
from src.patterns.p2_role_pipeline import RoleSeparatedPipeline, _is_approved, _parse_verdict
from src.patterns.p3_reflexive import ReflexiveSingleAgent
from src.patterns.p4_deliberation import MultiPerspectiveDeliberation


def _make_state(pattern):
    return RunState(
        run_id="test", pattern=pattern,
        dataset="ifeval", query_id="q1",
        query="What is AI?",
        reference="AI is artificial intelligence. It powers many applications. "
        "Machine learning is a subset of AI.",
    )


class TestGovernedWorkflow:
    def test_final_answer_is_set(self):
        result = GovernedWorkflow(MockLLMClient()).run(
            _make_state(PatternID.GOVERNED_WORKFLOW)
        )
        assert result.final_answer is not None
        assert len(result.final_answer) > 0

    def test_two_llm_calls(self):
        result = GovernedWorkflow(MockLLMClient()).run(
            _make_state(PatternID.GOVERNED_WORKFLOW)
        )
        assert len(result.calls) == 2

    def test_correct_agent_roles_in_order(self):
        result = GovernedWorkflow(MockLLMClient()).run(
            _make_state(PatternID.GOVERNED_WORKFLOW)
        )
        roles = [c.agent_role for c in result.calls]
        assert roles == ["drafter", "formatter"]

    def test_rounds_is_one(self):
        result = GovernedWorkflow(MockLLMClient()).run(
            _make_state(PatternID.GOVERNED_WORKFLOW)
        )
        assert result.rounds == 1


class TestRetrieveRelevantContext:
    def test_filters_irrelevant_sentences(self):
        context = (
            "AI is artificial intelligence. "
            "Venus is the hottest planet. "
            "Machine learning is a subset of AI. "
            "Saturn has rings made of ice."
        )
        result = _retrieve_relevant_context(context, "artificial intelligence machine learning")
        assert "Venus" not in result
        assert "Saturn" not in result
        assert "artificial intelligence" in result

    def test_keeps_all_when_query_words_are_in_every_sentence(self):
        context = "AI is here. AI is everywhere. AI helps us."
        result = _retrieve_relevant_context(context, "AI")
        assert len(result) > 0
        assert "AI is here" in result
        assert "AI is everywhere" in result

    def test_returns_original_context_when_no_query_words(self):
        context = "The sky is blue."
        result = _retrieve_relevant_context(context, "AI")
        assert result == context

    def test_deterministic(self):
        ctx = "First sentence about AI. Second about ML. Third about data."
        r1 = _retrieve_relevant_context(ctx, "ML")
        r2 = _retrieve_relevant_context(ctx, "ML")
        assert r1 == r2


class TestRoleSeparatedPipeline:
    def test_linear_pass_with_zero_revisions(self):
        result = RoleSeparatedPipeline(MockLLMClient(), max_revisions=0).run(
            _make_state(PatternID.ROLE_SEPARATED_PIPELINE)
        )
        assert len(result.calls) == 3
        roles = [c.agent_role for c in result.calls]
        assert roles == ["researcher", "analyst", "reviewer"]
        assert result.rounds == 1
        assert result.final_answer is not None

    def test_revision_loop_until_max_revisions(self):
        max_r = 2
        result = RoleSeparatedPipeline(MockLLMClient(), max_revisions=max_r).run(
            _make_state(PatternID.ROLE_SEPARATED_PIPELINE)
        )
        expected_calls = 1 + (max_r + 1) * 2
        assert len(result.calls) == expected_calls
        assert result.rounds == max_r + 1
        assert result.final_answer is not None

    def test_approved_verdict_stops_early(self):
        class MockApprovingReviewer(MockLLMClient):
            def call(self, agent_role, prompt, round_index=0, want_logprobs=False):
                if agent_role == "reviewer":
                    return LLMCall(
                        agent_role="reviewer", round_index=round_index,
                        prompt=prompt,
                        response_text="VERDICT: approved. All good.",
                    )
                return super().call(agent_role, prompt, round_index, want_logprobs)

        result = RoleSeparatedPipeline(MockApprovingReviewer(), max_revisions=5).run(
            _make_state(PatternID.ROLE_SEPARATED_PIPELINE)
        )
        assert len(result.calls) == 3
        assert result.rounds == 1

    def test_rounds_property_increments_correctly(self):
        result = RoleSeparatedPipeline(MockLLMClient(), max_revisions=1).run(
            _make_state(PatternID.ROLE_SEPARATED_PIPELINE)
        )
        assert result.rounds == 2

    def test_critique_is_passed_forward(self):
        class MockCritiquingReviewer(MockLLMClient):
            def call(self, agent_role, prompt, round_index=0, want_logprobs=False):
                if agent_role == "reviewer":
                    return LLMCall(
                        agent_role="reviewer", round_index=round_index,
                        prompt=prompt,
                        response_text="VERDICT: needs_revision\nCRITIQUE: MISSING_CITATION_MARKER. Revise.",
                    )
                return super().call(agent_role, prompt, round_index, want_logprobs)

        result = RoleSeparatedPipeline(MockCritiquingReviewer(), max_revisions=1).run(
            _make_state(PatternID.ROLE_SEPARATED_PIPELINE)
        )
        assert "MISSING_CITATION_MARKER" in result.calls[3].prompt


class TestMaxRevisionsValidation:
    def test_negative_max_revisions_raises(self):
        with pytest.raises(ValueError, match="non-negative integer"):
            RoleSeparatedPipeline(MockLLMClient(), max_revisions=-1)

    def test_non_int_max_revisions_raises(self):
        with pytest.raises(ValueError, match="non-negative integer"):
            RoleSeparatedPipeline(MockLLMClient(), max_revisions="two")

    def test_missing_config_file_raises_filenotfounderror(self, monkeypatch):
        import pathlib
        original_exists = pathlib.Path.exists
        def mock_exists(path_obj):
            if "experiment.yaml" in str(path_obj):
                return False
            return original_exists(path_obj)
        monkeypatch.setattr("pathlib.Path.exists", mock_exists)
        with pytest.raises(FileNotFoundError, match="experiment.yaml not found"):
            RoleSeparatedPipeline(MockLLMClient())


class TestIsApproved:
    def test_approved_verdict(self):
        assert _is_approved("VERDICT: approved") is True

    def test_not_approved(self):
        assert _is_approved("VERDICT: not approved") is False

    def test_unapproved(self):
        assert _is_approved("VERDICT: unapproved, missing sources") is False

    def test_needs_revision(self):
        assert _is_approved("VERDICT: needs_revision") is False

    def test_verdict_mentioned_in_critique_text_not_misread(self):
        text = (
            "VERDICT: needs_revision\n"
            "CRITIQUE: This analysis will be approved once the citation "
            "for the GDP figure is added."
        )
        assert _is_approved(text) is False

    def test_verdict_parsed_correctly_when_approved(self):
        text = "VERDICT: approved\nCRITIQUE: All claims are well supported."
        assert _is_approved(text) is True

    def test_malformed_verdict_defaults_to_not_approved(self):
        text = "This looks fine to me."
        assert _is_approved(text) is False


def test_p3_reflexive():
    result = ReflexiveSingleAgent(MockLLMClient(), max_iterations=2).run(
        _make_state(PatternID.REFLEXIVE_SINGLE_AGENT)
    )
    assert len(result.calls) > 1
    assert result.final_answer is not None


def test_p4_deliberation():
    result = MultiPerspectiveDeliberation(MockLLMClient(), num_agents=3).run(
        _make_state(PatternID.MULTI_PERSPECTIVE_DELIBERATION)
    )
    analyst_calls = [c for c in result.calls if c.agent_role.startswith("analyst_")]
    assert len(analyst_calls) == 3
    judge_calls = [c for c in result.calls if c.agent_role == "judge"]
    assert len(judge_calls) == 1


def test_p4_agents_are_symmetric():
    from src.patterns.p4_deliberation import AGENT_PROMPT
    p1 = AGENT_PROMPT.format(agent_id=0, context="ctx", query="q")
    p2 = AGENT_PROMPT.format(agent_id=1, context="ctx", query="q")
    p3 = AGENT_PROMPT.format(agent_id=2, context="ctx", query="q")
    base1 = p1.replace("Agent 0", "Agent {id}")
    base2 = p2.replace("Agent 1", "Agent {id}")
    base3 = p3.replace("Agent 2", "Agent {id}")
    assert base1 == base2 == base3, "Agent prompts must differ only by ID, not by stance"
