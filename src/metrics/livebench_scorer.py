import json
import warnings
from .base import Scorer

class LiveBenchScorer(Scorer):
    """
    Evaluates LiveBench questions.
    Uses official LiveBench exact match or per-category evaluation.
    """
    def __init__(self):
        try:
            import livebench.common as lb
        except ImportError:
            raise ImportError("LiveBench package is missing. Please run `pip install git+https://github.com/LiveBench/LiveBench.git@4bf3d6f4cb37fa8dc3967dd1b124fef5d4099635`.")

        self.lb = lb
        # Load all LiveBench questions into a dict for quick lookup
        # This is necessary because telemetry only contains a subset of fields.
        categories, _ = lb.get_categories_tasks()
        self.questions_map = {}
        for cat_name, cat_dataset in categories.items():
            # lb.load_questions filters by the default releases
            questions = lb.load_questions(cat_dataset, livebench_releases=lb.LIVE_BENCH_RELEASES, task_name=None)
            for q in questions:
                self.questions_map[str(q["question_id"])] = q

    def score(self, record: dict, reference: dict) -> dict:
        from livebench.gen_ground_truth_judgment import play_a_match_gt

        pred_ans_str = record.get("final_answer", "")
        if not pred_ans_str:
            return {"accuracy": 0.0, "parse_error": True}

        q_id = str(record.get("query_id", ""))
        if q_id not in self.questions_map:
            raise ValueError(f"Question ID {q_id} not found in LiveBench benchmark.")
            
        full_question = self.questions_map[q_id]
        
        # LiveBench evaluation expects this specific structure
        answer_dict = {
            "choices": [{"turns": [pred_ans_str]}]
        }
        
        match = self.lb.MatchSingle(
            question=full_question,
            model="agent",
            answer=answer_dict
        )
        
        try:
            result = play_a_match_gt(match)
        except Exception as e:
            # Handle uncaught exceptions in LiveBench code
            return {"accuracy": 0.0, "parse_error": True, "error_msg": str(e)}
            
        accuracy = float(result.get("score", 0.0))
        eval_status = result.get("eval_status", "")
        
        return {
            "accuracy": accuracy,
            "category": result.get("category", ""),
            "parse_error": eval_status == "eval_error"
        }
