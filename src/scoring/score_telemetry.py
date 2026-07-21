import json
import argparse
from pathlib import Path

from .finqa_scorer import FinQAScorer
from .livebench_scorer import LiveBenchScorer
from .popqa_scorer import PopqaScorer
from .ifeval_scorer import IfevalScorer

def main():
    parser = argparse.ArgumentParser(description="Score JSONL telemetry outputs using official harnesses.")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL telemetry file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL file with scores")
    parser.add_argument("--dataset", type=str, required=True, choices=["finqa", "livebench", "popqa", "ifeval"])
    parser.add_argument("--reference", type=str, required=True, help="Path to reference JSONL ground truths")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    ref_path = Path(args.reference)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    # Load references into memory (keyed by query_id)
    references = {}
    if ref_path.exists():
        with open(ref_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                ref_obj = json.loads(line)
                q_id = ref_obj.get("query_id", ref_obj.get("id"))
                if q_id is not None:
                    references[str(q_id)] = ref_obj
                    
    # Initialize appropriate scorer
    if args.dataset == "finqa":
        scorer = FinQAScorer()
    elif args.dataset == "livebench":
        scorer = LiveBenchScorer()
    elif args.dataset == "popqa":
        scorer = PopqaScorer()
    elif args.dataset == "ifeval":
        scorer = IfevalScorer()
    else:
        raise NotImplementedError(f"Scorer for {args.dataset} is assigned to another team member or not implemented yet.")
        
    results = []
    with open(input_path, "r") as f_in:
        for line in f_in:
            if not line.strip(): continue
            record = json.loads(line)
            
            q_id = str(record.get("query_id", ""))
            ref_data = references.get(q_id, {})
            
            try:
                score_dict = scorer.score(record, ref_data)
            except Exception as e:
                score_dict = {"parse_error": True, "error_msg": str(e)}
                
            out_record = {
                "query_id": q_id,
                "pattern": record.get("pattern"),
                "seed": record.get("seed"),
                "scores": score_dict
            }
            results.append(out_record)
            
    with open(output_path, "w") as f_out:
        for res in results:
            f_out.write(json.dumps(res) + "\n")
            
    print(f"Scored {len(results)} records. Output written to {output_path}")

if __name__ == "__main__":
    main()
