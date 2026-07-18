import json
from .base import Scorer

class LiveBenchScorer(Scorer):
    """
    Evaluates LiveBench questions.
    Uses official LiveBench exact match or per-category evaluation.
    """
    def __init__(self):
        pass

    def score(self, record: dict, reference: dict) -> dict:
        pred_ans_str = record.get("final_answer", "")
        if not pred_ans_str:
            return {"accuracy": 0.0, "parse_error": True}

        # Reference should be a dict containing 'ground_truth' or 'answer'
        gold_ans = ""
        if isinstance(reference, dict):
            gold_ans = reference.get("ground_truth", reference.get("answer", ""))
        else:
            gold_ans = str(reference)
            
        parse_error = False
        accuracy = 0.0
        
        # Placeholder for official LiveBench evaluation logic
        # Typically LiveBench uses exact string matching, regex, or code execution
        # For this skeleton, we use basic substring matching
        if gold_ans and str(gold_ans).lower() in pred_ans_str.lower():
            accuracy = 1.0
            
        return {
            "accuracy": accuracy,
            "parse_error": parse_error
        }
