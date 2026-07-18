import json
import math
from typing import Dict, Any
from .base import Scorer
from .finqa_eval.evaluate import eval_program, equal_program, program_tokenization

class FinQAScorer(Scorer):
    """
    Evaluates FinQA programs using the official czyssrs/FinQA executor.
    Expects final_answer to contain a parsed program (e.g. list of strings).
    Expects reference to contain the gold execution answer and table.
    """
    def __init__(self, atol=1e-4, rtol=1e-4):
        self.atol = atol
        self.rtol = rtol

    def score(self, record: dict, reference: dict) -> dict:
        pred_ans = record.get("final_answer", "")
        if not pred_ans:
            return {"execution_accuracy": 0.0, "program_accuracy": 0.0, "parse_error": True}

        # Reference should contain 'table', 'exe_ans', and 'program'
        table = reference.get("table", [])
        gold_res = reference.get("exe_ans", reference.get("answer"))
        gold_prog = reference.get("program", "")
        
        parse_error = False
        exe_acc = 0.0
        prog_acc = 0.0
        
        # In a real scenario, the final_answer would be parsed into a program list.
        # We assume pred_ans is either a string or already a list.
        if isinstance(pred_ans, str):
            try:
                # Basic tokenization if it's a string like "add(1, 2)"
                pred_prog = program_tokenization(pred_ans)
            except Exception:
                parse_error = True
                pred_prog = []
        else:
            pred_prog = pred_ans
            
        gold_prog_list = []
        if isinstance(gold_prog, str):
            gold_prog_list = program_tokenization(gold_prog)
        elif isinstance(gold_prog, list):
            gold_prog_list = gold_prog
            
        # Evaluate Execution
        try:
            if not parse_error and pred_prog:
                invalid_flag, exe_res = eval_program(pred_prog, table)
                if invalid_flag == 1 or exe_res == "n/a":
                    parse_error = True
                elif invalid_flag == 0 and exe_res != "n/a" and gold_res is not None:
                    # Compare using tolerance
                    if math.isclose(float(exe_res), float(gold_res), rel_tol=self.rtol, abs_tol=self.atol):
                        exe_acc = 1.0
        except Exception:
            parse_error = True
            
        # Evaluate Program Symbolic Matching
        try:
            if not parse_error and pred_prog and gold_prog_list:
                if equal_program(gold_prog_list, pred_prog):
                    prog_acc = 1.0
        except Exception:
            pass # Program matching failed
            
        return {
            "execution_accuracy": exe_acc,
            "program_accuracy": prog_acc,
            "parse_error": parse_error
        }

