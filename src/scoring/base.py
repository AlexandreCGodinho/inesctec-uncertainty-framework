from abc import ABC, abstractmethod
import json

class Scorer(ABC):
    @abstractmethod
    def score(self, record: dict, reference: dict) -> dict:
        """
        Evaluate the telemetry record against the reference data.
        :param record: A dictionary representing the parsed JSONL Telemetry record.
                       Contains fields like 'final_answer', 'query_id', etc.
        :param reference: Ground-truth reference data.
        :return: A dictionary of metric scores (e.g., {'execution_accuracy': 1.0}).
        """
        pass
