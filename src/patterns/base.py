from abc import ABC, abstractmethod
from src.schemas import RunState
from src.llm_client import LLMClient


class Orchestrator(ABC):
    """Every pattern implements exactly this. The runner treats all four
    patterns identically; all pattern-specific logic lives inside run()."""

    def __init__(self, client: LLMClient):
        self.client = client

    @abstractmethod
    def run(self, state: RunState) -> RunState:
        """Consume a RunState with the query filled in; return it with
        `calls` populated and `final_answer` set. Must be deterministic
        given the client's seed. Must never mutate global state."""
