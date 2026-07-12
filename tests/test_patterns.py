from src.schemas import RunState, PatternID
from src.llm_client import MockLLMClient
from src.patterns.p1_governed import GovernedWorkflow, _retrieve_relevant_context
from src.patterns.p2_role_pipeline import RoleSeparatedPipeline
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
    def test_filters_correctly_with_stopwords_and_punctuation(self):
        context = (
            "Paris is the capital of France. "
            "The Eiffel Tower is a famous landmark. "
            "Bananas are a good source of potassium."
        )
        result = _retrieve_relevant_context(context, "What is AI?")
        assert result == context or result == ""

    def test_filters_out_stopword_only_matches(self):
        context = (
            "Paris is the capital of France. "
            "Bananas are a good source of potassium."
        )
        result = _retrieve_relevant_context(context, "What is the capital?")
        assert "Paris" in result
        assert "Bananas" not in result

    def test_word_boundary_matching_avoids_substring_false_positives(self):
        context = (
            "The main office is located downtown. "
            "AI research is advancing rapidly."
        )
        result = _retrieve_relevant_context(context, "Tell me about AI?")
        assert "AI research" in result
        assert "main office" not in result

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


def test_p2_role_pipeline():
    result = RoleSeparatedPipeline(MockLLMClient()).run(
        _make_state(PatternID.ROLE_SEPARATED_PIPELINE)
    )
    assert len(result.calls) == 3
    roles = [c.agent_role for c in result.calls]
    assert roles == ["researcher", "analyst", "reviewer"]


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
