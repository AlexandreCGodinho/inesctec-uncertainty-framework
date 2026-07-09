import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm_client import MockLLMClient
from src.schemas import RunState, PatternID
from src.patterns.p3_reflexive import ReflexiveSingleAgent

state = RunState(run_id="smoke-001", pattern=PatternID.REFLEXIVE_SINGLE_AGENT,
                 dataset="ifeval", query_id="demo",
                 query="Answer in exactly two sentences: what is INESC TEC?",
                 reference="INESC TEC is a research institute in Porto.")

result = ReflexiveSingleAgent(MockLLMClient()).run(state)
print(result.final_answer, f"({len(result.calls)} calls, {result.rounds} rounds)")
