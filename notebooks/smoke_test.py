import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm_client import MockLLMClient
from src.schemas import RunState, PatternID
from src.patterns.p3_reflexive import ReflexiveSingleAgent
from src.patterns.p4_deliberation import MultiPerspectiveDeliberation

state_p3 = RunState(run_id="smoke-001", pattern=PatternID.REFLEXIVE_SINGLE_AGENT,
                 dataset="ifeval", query_id="demo",
                 query="Answer in exactly two sentences: what is INESC TEC?",
                 reference="INESC TEC is a research institute in Porto.")

result_p3 = ReflexiveSingleAgent(MockLLMClient()).run(state_p3)
print("--- Pattern 3 (Reflexive) ---")
print(result_p3.final_answer)
print(f"({len(result_p3.calls)} calls, {result_p3.rounds} rounds)")

state_p4 = RunState(run_id="smoke-002", pattern=PatternID.MULTI_PERSPECTIVE_DELIBERATION,
                 dataset="ifeval", query_id="demo",
                 query="Answer in exactly two sentences: what is INESC TEC?",
                 reference="INESC TEC is a research institute in Porto.")

result_p4 = MultiPerspectiveDeliberation(MockLLMClient()).run(state_p4)
print("\n--- Pattern 4 (Deliberation) ---")
print(result_p4.final_answer)
print(f"({len(result_p4.calls)} calls, {result_p4.rounds} rounds)")
