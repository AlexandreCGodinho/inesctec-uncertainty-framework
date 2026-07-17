import re

from src.schemas import RunState, Telemetry, PatternID, LLMCall
from src.metrics.u_L import compute_u_L
from src.metrics.u_F import compute_u_F
from src.metrics.u_C import compute_u_C
from src.metrics.u_R import compute_u_R
from src.metrics.u_P import compute_u_P
from src.metrics.composite import compute_composite, compute_surface


def _find_final_logprobs(state: RunState) -> list[float] | None:
    if state.pattern == PatternID.GOVERNED_WORKFLOW:
        for c in reversed(state.calls):
            if c.agent_role == "formatter":
                return c.logprobs
        return None
    if state.pattern == PatternID.ROLE_SEPARATED_PIPELINE:
        for c in reversed(state.calls):
            if c.agent_role == "analyst":
                return c.logprobs
        return None
    if state.pattern == PatternID.REFLEXIVE_SINGLE_AGENT:
        artifacts = _extract_reflexive_artifact_calls(state)
        if not artifacts:
            return None
        return artifacts[-1].logprobs
    return None


def _extract_analyst_traces(state: RunState) -> list[str]:
    analyst_calls = [c for c in state.calls if c.agent_role == "analyst"]
    analyst_calls.sort(key=lambda c: c.round_index)
    return [c.response_text for c in analyst_calls]


def _extract_reflexive_artifact_calls(state: RunState) -> list[LLMCall]:
    """Pattern 3: return draft + revision calls (excluding critiques),
    in round order. At round 0 the single call is the draft; at each
    round >= 1 the first call is the critique and the second (if present)
    is the revision. A round with only one call at index >= 1 is an
    approved critique with no revision — skipped."""
    solo_calls = [c for c in state.calls if c.agent_role == "solo_reflexive"]
    rounds: dict[int, list[LLMCall]] = {}
    for c in solo_calls:
        rounds.setdefault(c.round_index, []).append(c)
    artifacts: list[LLMCall] = []
    for round_idx in sorted(rounds):
        calls_at_round = rounds[round_idx]
        if round_idx == 0:
            artifacts.append(calls_at_round[0])
        elif len(calls_at_round) >= 2:
            artifacts.append(calls_at_round[1])
    return artifacts


def _extract_reflexive_traces(state: RunState) -> list[str]:
    return [c.response_text for c in _extract_reflexive_artifact_calls(state)]


_ANALYST_AGENT_PATTERN = re.compile(r"^analyst_(\d+)$")


def _extract_analyst_agent_answers(state: RunState) -> list[str]:
    """Pattern 4: collect each symmetric agent's answer from the LAST round only,
    sorted by agent index for determinism."""
    agent_entries: list[tuple[int, int, str]] = []
    for c in state.calls:
        match = _ANALYST_AGENT_PATTERN.match(c.agent_role)
        if match:
            agent_entries.append((c.round_index, int(match.group(1)), c.response_text))
    if not agent_entries:
        return []
    max_round = max(rc for rc, _, _ in agent_entries)
    last_round = [(idx, txt) for rc, idx, txt in agent_entries if rc == max_round]
    last_round.sort(key=lambda pair: pair[0])
    return [txt for _, txt in last_round]


def _extract_judge_traces(state: RunState) -> list[str]:
    """Pattern 4: collect Judge response texts across all rounds,
    sorted by round_index for determinism."""
    judge_calls = [c for c in state.calls if c.agent_role == "judge"]
    judge_calls.sort(key=lambda c: c.round_index)
    return [c.response_text for c in judge_calls]


def _extract_last_round_analyst_logprobs(state: RunState) -> list[list[float]]:
    """Pattern 4: collect logprobs from each analyst_N at the last round.
    Returns empty list if no analysts carry logprobs."""
    entries: list[tuple[int, list[float]]] = []
    for c in state.calls:
        match = _ANALYST_AGENT_PATTERN.match(c.agent_role)
        if match and c.logprobs is not None:
            entries.append((c.round_index, c.logprobs))
    if not entries:
        return []
    max_round = max(rc for rc, _ in entries)
    return [lp for rc, lp in entries if rc == max_round]


def build_telemetry(state: RunState) -> Telemetry:
    u_L = compute_u_L(state.final_answer or "")
    u_F = compute_u_F(state.final_answer or "", state.reference)

    pattern_id = state.pattern.value

    if state.pattern == PatternID.MULTI_PERSPECTIVE_DELIBERATION:
        agent_answers = _extract_analyst_agent_answers(state)
        u_C = compute_u_C(agent_answers) if len(agent_answers) >= 2 else None
    else:
        u_C = None  # P1/P2/P3 lack symmetric parallel agents (Doc 03 §2.3)

    if state.pattern == PatternID.ROLE_SEPARATED_PIPELINE:
        traces = _extract_analyst_traces(state)
        u_R = compute_u_R(traces) if len(traces) >= 2 else None
    elif state.pattern == PatternID.REFLEXIVE_SINGLE_AGENT:
        traces = _extract_reflexive_traces(state)
        u_R = compute_u_R(traces) if len(traces) >= 2 else None
    elif state.pattern == PatternID.MULTI_PERSPECTIVE_DELIBERATION:
        judge_traces = _extract_judge_traces(state)
        if len(judge_traces) == 1:
            u_R = 0.0  # Option B: immediate consensus (Alexandre)
        elif len(judge_traces) >= 2:
            u_R = compute_u_R(judge_traces)
        else:
            u_R = None
    else:
        u_R = None

    if state.pattern == PatternID.MULTI_PERSPECTIVE_DELIBERATION:
        analyst_logprobs = _extract_last_round_analyst_logprobs(state)
        if analyst_logprobs:
            logprobs = [sum(vals) / len(vals) for vals in zip(*analyst_logprobs)]
        else:
            logprobs = None
    else:
        logprobs = _find_final_logprobs(state)
    u_P = compute_u_P(logprobs, u_F)

    U_surface = compute_surface(u_L, u_F, u_P)
    U_composite = compute_composite(pattern_id, u_L, u_F, u_P, u_R, u_C)

    termination_reason = "truncated" if any(c.truncated for c in state.calls) else "completed"

    return Telemetry(
        run_id=state.run_id,
        pattern=state.pattern,
        query_id=state.query_id,
        seed=state.seed,
        final_answer=state.final_answer,
        termination_reason=termination_reason,
        tokens_prompt=sum(c.tokens_prompt for c in state.calls),
        tokens_completion=sum(c.tokens_completion for c in state.calls),
        latency_ms=round(sum(c.latency_ms for c in state.calls), 1),
        u_L=u_L,
        u_F=u_F,
        u_C=u_C,
        u_R=u_R,
        u_P=u_P,
        U_surface=U_surface,
        U_composite=U_composite,
        truncation_flag=any(c.truncated for c in state.calls),
    )
