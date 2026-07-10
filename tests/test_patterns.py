from src.schemas import RunState, PatternID
from src.llm_client import MockLLMClient
from src.patterns import (
    GovernedWorkflow,
    RoleSeparatedPipeline,
    ReflexiveSingleAgent,
    MultiPerspectiveDeliberation,
)


def _make_state(pattern):
    return RunState(
        run_id="test", pattern=pattern,
        dataset="ifeval", query_id="q1",
        query="What is AI?", reference="AI is artificial intelligence.",
    )


def test_p1_governed():
    result = GovernedWorkflow(MockLLMClient()).run(_make_state(PatternID.GOVERNED_WORKFLOW))
    assert len(result.calls) == 1
    assert result.final_answer is not None


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
