import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Generate per-item score table.")
    parser.add_argument("--raw-telemetry-dir", type=str, default="telemetry", help="Directory with raw JSONL telemetry files.")
    parser.add_argument("--scored-telemetry-dir", type=str, default="telemetry/scored", help="Directory with scored JSONL files.")
    parser.add_argument("--output", type=str, default="telemetry/per_item_scores.csv", help="Path to output CSV.")
    parser.add_argument("--dev", action="store_true", help="Run in dev mode, suffixing output with _DEV.")
    args = parser.parse_args()

    out_path_str = args.output
    if args.dev and not out_path_str.endswith("_DEV.csv"):
        out_path_str = out_path_str.replace(".csv", "_DEV.csv")
    args.output = out_path_str

    raw_dir = Path(args.raw_telemetry_dir)
    scored_dir = Path(args.scored_telemetry_dir)

    if not raw_dir.exists():
        print(f"Directory {raw_dir} does not exist.")
        return
        
    if not scored_dir.exists():
        scored_dir.mkdir(parents=True, exist_ok=True)
        print(f"Warning: Scored telemetry dir {scored_dir} did not exist. Created it, but there's nothing to merge.")

    # Key: (query_id, pattern, seed)
    # Value: dict of combined data
    merged_data = defaultdict(dict)

    # 1. Read raw telemetry
    for raw_file in raw_dir.glob("*.jsonl"):
        with open(raw_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                
                q_id = str(record.get("query_id", ""))
                pattern = str(record.get("pattern", ""))
                seed = str(record.get("seed", ""))
                key = (q_id, pattern, seed)
                
                # Use dataset from record if available, otherwise infer from filename
                dataset_val = record.get("dataset", "") or raw_file.stem.split("_")[0]
                
                merged_data[key].update({
                    "query_id": q_id,
                    "pattern": pattern,
                    "seed": seed,
                    "dataset": dataset_val,
                    "termination_reason": record.get("termination_reason", ""),
                    "tokens_prompt": record.get("tokens_prompt", 0),
                    "tokens_completion": record.get("tokens_completion", 0),
                    "latency_ms": record.get("latency_ms", 0.0),
                    "u_L": record.get("u_L"),
                    "u_F": record.get("u_F"),
                    "u_C": record.get("u_C"),
                    "u_R": record.get("u_R"),
                    "u_P": record.get("u_P"),
                    "U_surface": record.get("U_surface"),
                    "U_composite": record.get("U_composite"),
                    "truncation_flag": record.get("truncation_flag", False),
                    "final_answer": record.get("final_answer", ""),
                    "model_name": record.get("model_name", ""),
                    "provider": record.get("provider", ""),
                })

    # 2. Read scored telemetry
    for scored_file in scored_dir.glob("*.jsonl"):
        with open(scored_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                
                q_id = str(record.get("query_id", ""))
                pattern = str(record.get("pattern", ""))
                seed = str(record.get("seed", ""))
                key = (q_id, pattern, seed)
                
                scores = record.get("scores", {})
                
                # Combine scores into merged data
                # We add everything inside scores to the record
                merged_data[key].update(scores)
                
    # 3. Define the headline binary column logic
    def get_headline_bin(dataset: str, record: dict) -> str:
        if record.get("parse_error"):
            return "0"
            
        dataset = dataset.lower()
        if dataset == "popqa":
            return str(int(record.get("match", 0)))
        elif dataset == "finqa":
            return str(int(record.get("execution_accuracy", 0)))
        elif dataset == "ifeval":
            return str(int(record.get("strict_prompt", 0)))
        elif dataset == "livebench":
            return ""
        return ""

    # 4. Gather all unique headers
    all_headers = set(["query_id", "pattern", "seed", "dataset", "termination_reason", "tokens_completion", "latency_ms", "headline_bin"])
    for record in merged_data.values():
        all_headers.update(record.keys())
        
    # Sort headers to have a predictable order
    ordered_headers = ["query_id", "pattern", "seed", "dataset", "headline_bin", "termination_reason", "tokens_completion", "latency_ms", "score_popqa_raw"]
    remaining_headers = sorted(list(all_headers - set(ordered_headers)))
    final_headers = ordered_headers + remaining_headers

    # 5. Write CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=final_headers)
        writer.writeheader()
        for key, record in merged_data.items():
            record["headline_bin"] = get_headline_bin(record.get("dataset", ""), record)
            # Ensure all headers exist in the dict to avoid DictWriter errors
            for h in final_headers:
                if h not in record:
                    record[h] = ""
            writer.writerow(record)
            
    print(f"Generated score table with {len(merged_data)} records at {out_path}")

if __name__ == "__main__":
    main()
