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
    args = parser.parse_args()

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
        # Ignore manifest and scored files
        if raw_file.name == "manifest.json" or raw_file.parent == scored_dir:
            continue
            
        with open(raw_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                
                q_id = str(record.get("query_id", ""))
                pattern = str(record.get("pattern", ""))
                seed = str(record.get("seed", ""))
                key = (q_id, pattern, seed)
                
                # We need to know which dataset it is from the filename or record
                # Try to infer dataset from filename if not explicitly present in a standard way
                dataset_inferred = raw_file.stem.split("_")[0] 
                
                merged_data[key].update({
                    "query_id": q_id,
                    "pattern": pattern,
                    "seed": seed,
                    "dataset": dataset_inferred,
                    "termination_reason": record.get("termination_reason", ""),
                    "tokens_completion": record.get("tokens_completion", 0),
                    "latency_ms": record.get("latency_ms", 0.0),
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
            return str(int(record.get("alias_match", 0)))
        elif dataset == "finqa":
            return str(int(record.get("execution_accuracy", 0)))
        elif dataset == "ifeval":
            # Using prompt-level strict as headline per spec
            return str(int(record.get("prompt_level_strict", 0)))
        elif dataset == "livebench":
            # LiveBench has no natural binary, we leave it blank in headline_bin
            return ""
        return ""

    # 4. Gather all unique headers
    all_headers = set(["query_id", "pattern", "seed", "dataset", "termination_reason", "tokens_completion", "latency_ms", "headline_bin"])
    for record in merged_data.values():
        all_headers.update(record.keys())
        
    # Sort headers to have a predictable order
    ordered_headers = ["query_id", "pattern", "seed", "dataset", "headline_bin", "termination_reason", "tokens_completion", "latency_ms"]
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
