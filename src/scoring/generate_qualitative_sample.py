import json
import csv
import argparse
import random
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Generate anonymized qualitative coding sample.")
    parser.add_argument("--score-table", type=str, default="telemetry/per_item_scores.csv", help="Path to input score CSV.")
    parser.add_argument("--raw-telemetry-dir", type=str, default="telemetry", help="Directory with raw JSONL telemetry files.")
    parser.add_argument("--output-transcript", type=str, default="telemetry/qualitative_coding_transcript.jsonl", help="Output transcript JSONL.")
    parser.add_argument("--output-labels", type=str, default="telemetry/qualitative_coding_labels.csv", help="Output label map CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed for sampling.")
    args = parser.parse_args()

    random.seed(args.seed)

    score_path = Path(args.score_table)
    if not score_path.exists():
        print(f"Error: Score table {score_path} not found.")
        return

    # Load score table
    incorrect_items = []
    correct_items_by_group = defaultdict(list)
    
    with open(score_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_id = row.get("query_id")
            pattern = row.get("pattern")
            seed = row.get("seed")
            dataset = row.get("dataset")
            headline = row.get("headline_bin")
            
            # Skip if dataset is livebench (no natural binary headline, handled manually or excluded from this automated rule)
            if dataset.lower() == "livebench":
                # Fallback: Treat accuracy == 1.0 as correct, else incorrect
                accuracy_val = float(row.get("accuracy", 0.0))
                is_correct = (accuracy_val >= 1.0)
            else:
                is_correct = (headline == "1")

            item_key = (q_id, pattern, seed, dataset)
            
            if not is_correct:
                incorrect_items.append(item_key)
            else:
                group_key = (dataset, pattern)
                correct_items_by_group[group_key].append(item_key)

    # Sample 15 correct items per group
    sampled_correct = []
    for group_key, items in correct_items_by_group.items():
        if len(items) <= 15:
            sampled_correct.extend(items)
        else:
            sampled_correct.extend(random.sample(items, 15))

    final_sample_keys = set(incorrect_items + sampled_correct)
    
    # Load original texts from raw telemetry
    raw_dir = Path(args.raw_telemetry_dir)
    telemetry_data = {}
    for raw_file in raw_dir.glob("*.jsonl"):
        if raw_file.name == "manifest.json" or "scored" in raw_file.parts:
            continue
            
        with open(raw_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                q_id = str(record.get("query_id", ""))
                pattern = str(record.get("pattern", ""))
                seed = str(record.get("seed", ""))
                dataset = raw_file.stem.split("_")[0]
                
                key = (q_id, pattern, seed, dataset)
                if key in final_sample_keys:
                    telemetry_data[key] = record

    # Shuffle and Anonymize
    final_sample_list = list(telemetry_data.items())
    random.shuffle(final_sample_list)
    
    transcripts = []
    labels = []
    
    for i, (key, record) in enumerate(final_sample_list):
        anon_id = f"item_{i:04d}"
        q_id, pattern, seed, dataset = key
        
        # Build anonymized transcript
        transcript = {
            "anon_id": anon_id,
            "dataset": dataset,
            "query": record.get("query"),
            "reference": record.get("reference"),
            "final_answer": record.get("final_answer"),
            "calls": []
        }
        
        # Scrub agent roles and pattern identities from calls if needed
        for call in record.get("calls", []):
            transcript["calls"].append({
                "prompt": call.get("prompt"),
                "response": call.get("response_text")
            })
            
        transcripts.append(transcript)
        
        # Keep labels
        labels.append({
            "anon_id": anon_id,
            "dataset": dataset,
            "query_id": q_id,
            "pattern": pattern,
            "seed": seed
        })
        
    # Write transcript
    out_transcript = Path(args.output_transcript)
    out_transcript.parent.mkdir(parents=True, exist_ok=True)
    with open(out_transcript, "w", encoding="utf-8") as f:
        for t in transcripts:
            f.write(json.dumps(t) + "\n")
            
    # Write labels
    out_labels = Path(args.output_labels)
    with open(out_labels, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["anon_id", "dataset", "query_id", "pattern", "seed"])
        writer.writeheader()
        writer.writerows(labels)
        
    print(f"Generated qualitative sample with {len(transcripts)} items.")
    print(f"Transcript written to {out_transcript}")
    print(f"Label map written to {out_labels}")

if __name__ == "__main__":
    main()
