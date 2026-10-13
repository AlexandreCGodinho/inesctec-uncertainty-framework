import pytest

from src.metrics import (
    compute_u_L,
    compute_u_C,
    compute_u_R,
    compute_u_P,
    compute_composite,
    compute_surface,
)


def test_u_L_no_constraints():
    assert compute_u_L("Any answer") == 0.0


def test_u_L_with_constraints():
    constraints = [{"type": "min_length", "value": 10}]
    assert compute_u_L("Short", constraints) > 0.0
    assert compute_u_L("Long enough answer here", constraints) == 0.0


def test_u_L_finqa_and_popqa_return_none():
    assert compute_u_L("The result is 42", dataset="finqa") is None
    assert compute_u_L("Paris", dataset="popqa") is None


def test_u_L_livebench_slot_compliance():
    # Triple stars slot
    q_stars = "Answer format: ***X*** where X is answer"
    assert compute_u_L("Reasoning...\n***Paris***", dataset="livebench", query=q_stars) == 0.0
    assert compute_u_L("Reasoning with no stars Paris", dataset="livebench", query=q_stars) == 1.0

    # False positive test for triple stars in intermediate reasoning
    ans_stars_reasoning_only = "We check ***case 1*** in reasoning.\nThe answer is London."
    assert compute_u_L(ans_stars_reasoning_only, dataset="livebench", query=q_stars) == 1.0
    ans_stars_both = "We check ***case 1*** in reasoning.\n***London***"
    assert compute_u_L(ans_stars_both, dataset="livebench", query=q_stars) == 0.0

    # Solution tag slot
    q_sol = "Output in the format: <solution>A, B</solution>"
    assert compute_u_L("Here is <solution>A, B</solution>", dataset="livebench", query=q_sol) == 0.0
    assert compute_u_L("Just A, B without tags", dataset="livebench", query=q_sol) == 1.0

    # Bold slot
    q_bold = "put your answer in **bold**"
    assert compute_u_L("The answer is **42**", dataset="livebench", query=q_bold) == 0.0
    assert compute_u_L("The answer is 42", dataset="livebench", query=q_bold) == 1.0

    # False positive test for bold in intermediate reasoning
    ans_bold_reasoning_only = "We consider **case 1** where x = 1.\nStep 2: 1 + 1 = 2.\nTherefore, the result is 2."
    assert compute_u_L(ans_bold_reasoning_only, dataset="livebench", query=q_bold) == 1.0
    ans_bold_both = "We consider **case 1** where x = 1.\nStep 2: 1 + 1 = 2.\nTherefore, the result is **2**."
    assert compute_u_L(ans_bold_both, dataset="livebench", query=q_bold) == 0.0

    # Orchestrator verdict preamble with header label and terminal answer
    ans_verdict = "VERDICT: approved\nReasoning...\n**Conclusion:**\n**42**"
    assert compute_u_L(ans_verdict, dataset="livebench", query=q_bold) == 0.0

    # Constraints applied strictly to isolated answer slot
    c_paris = [{"type": "contains", "value": "Paris"}]
    c_london = [{"type": "contains", "value": "London"}]
    # Slot has Paris -> passes Paris constraint, fails London constraint
    assert compute_u_L("Reasoning with London...\n***Paris***", dataset="livebench", constraints=c_paris) == 0.0
    assert compute_u_L("Reasoning with London...\n***Paris***", dataset="livebench", constraints=c_london) == 1.0
    # Text contains Paris in reasoning but terminal slot is London -> fails Paris constraint
    assert compute_u_L("Reasoning with Paris in reasoning...\n***London***", dataset="livebench", constraints=c_paris) == 1.0


def test_u_L_ifeval_extracted_span():
    # Prompt asks to start with 'Hello'
    query = "Write a greeting starting with 'Hello'."
    # Final answer with orchestration preamble and VERDICT line
    final_ans = "Preamble\nVERDICT: approved\nHello world!"
    # Should evaluate on extracted span 'Hello world!' and pass (u_L = 0.0)
    assert compute_u_L(final_ans, dataset="ifeval", query=query) == 0.0

    # Failing case where extracted span does not start with Hello
    final_ans_fail = "Preamble\nVERDICT: approved\nGood morning!"
    assert compute_u_L(final_ans_fail, dataset="ifeval", query=query) > 0.0


def test_u_C_single_answer():
    assert compute_u_C(["only one"]) is None


def test_u_C_two_identical():
    u = compute_u_C(["Hello world", "Hello world"])
    assert u is not None
    assert 0.0 <= u <= 1.0


def test_u_R_single_trace():
    assert compute_u_R(["only one"]) is None


def test_u_P_no_logprobs():
    assert compute_u_P(None, 0.5) == 0.0


def test_u_P_empty_logprobs():
    assert compute_u_P([], 0.3) == 0.0


def test_u_P_low_fluency():
    u = compute_u_P([-10.0, -12.0], u_F=0.1)
    assert u >= 0.0


def test_surface_all_none():
    assert compute_surface(None, None, None) is None


def test_surface_partial():
    s = compute_surface(0.1, 0.2, None)
    assert abs(s - 0.15) < 1e-10


# ─── Composite Score & Strict Abstention Tests ────────────────────

def test_composite_pattern1_governed():
    # P1 active: w_L: 0.50, w_F: 0.50
    # Both active present:
    c = compute_composite(pattern_id=1, u_L=0.2, u_F=0.4, u_P=0.9, u_R=None, u_C=None, dataset="ifeval")
    assert c == pytest.approx(0.50 * 0.2 + 0.50 * 0.4)

    # u_P value does not affect result because w_P = 0.00
    c_no_up = compute_composite(pattern_id=1, u_L=0.2, u_F=0.4, u_P=None, u_R=None, u_C=None, dataset="ifeval")
    assert c_no_up == pytest.approx(c)


def test_composite_strict_abstention_on_missing_active_sensor():
    # P1: w_F is active; if u_F is None (e.g. JudgeParseError), strict abstention returns None
    c = compute_composite(pattern_id=1, u_L=0.2, u_F=None, u_P=0.1, u_R=None, u_C=None, dataset="ifeval")
    assert c is None

    # P2: w_R is active; if u_R is None, strict abstention returns None
    c2 = compute_composite(pattern_id=2, u_L=0.2, u_F=0.3, u_P=None, u_R=None, u_C=None, dataset="ifeval")
    assert c2 is None

    # P4: w_C is active; if u_C is None, strict abstention returns None
    c4 = compute_composite(pattern_id=4, u_L=0.2, u_F=0.2, u_P=None, u_R=0.0, u_C=None, dataset="ifeval")
    assert c4 is None


def test_composite_renormalization_for_finqa_and_popqa():
    # On FinQA and PopQA, u_L is N/A by design
    # P1: without w_L, only w_F remains, renormalizing to 1.00
    c_finqa = compute_composite(pattern_id=1, u_L=None, u_F=0.35, u_P=None, u_R=None, u_C=None, dataset="finqa")
    assert c_finqa == pytest.approx(0.35)

    c_popqa = compute_composite(pattern_id=1, u_L=None, u_F=0.10, u_P=None, u_R=None, u_C=None, dataset="popqa")
    assert c_popqa == pytest.approx(0.10)

    # P2: active without w_L: w_F (0.333), w_R (0.334) -> ~50/50 split
    c_p2 = compute_composite(pattern_id=2, u_L=None, u_F=0.2, u_P=None, u_R=0.4, u_C=None, dataset="finqa")
    assert c_p2 == pytest.approx(0.30, abs=1e-3)
