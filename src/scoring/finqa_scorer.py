import math
import json
import re
from .base import Scorer

class FinQAScorer(Scorer):
    """
    Evaluates FinQA programs.
    Expects final_answer to contain either the numerical answer or a program.
    Expects reference to contain the gold execution answer and optionally the gold program.
    """
    def __init__(self, atol=1e-4, rtol=1e-4):
        self.atol = atol
        self.rtol = rtol

    def _extract_number(self, text: str) -> float:
        # Extract the last number found in the string
        matches = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
        if not matches:
            raise ValueError(f"No number found in text: {text}")
        return float(matches[-1])

    def score(self, record: dict, reference: dict) -> dict:
        pred_ans_str = record.get("final_answer", "")
        if not pred_ans_str:
            return {"execution_accuracy": 0.0, "program_accuracy": 0.0, "parse_error": True}

        # The reference should be a dict with at least 'exe_ans' or similar. 
        # For this prototype, we assume it's just the gold number string, or a dict.
        gold_ans_val = None
        gold_prog_str = None
        if isinstance(reference, dict):
            gold_ans_val = reference.get("exe_ans")
            if gold_ans_val is None:
                gold_ans_val = reference.get("answer")
            gold_prog_str = reference.get("program")
        else:
            try:
                gold_ans_val = float(reference)
            except (ValueError, TypeError):
                pass
        
        # 1. Execution Accuracy
        exe_acc = 0.0
        parse_error = False
        try:
            pred_val = self._extract_number(pred_ans_str)
            if gold_ans_val is not None:
                gold_val = float(gold_ans_val)
                if math.isclose(pred_val, gold_val, rel_tol=self.rtol, abs_tol=self.atol):
                    exe_acc = 1.0
        except ValueError:
            parse_error = True
            
        # 2. Program Accuracy (Symbolic exact match fallback for now)
        prog_acc = 0.0
        if gold_prog_str and gold_prog_str in pred_ans_str:
            prog_acc = 1.0
            
        return {
            "execution_accuracy": exe_acc,
            "program_accuracy": prog_acc,
            "parse_error": parse_error
        }
