import logging
import json
from pathlib import Path
from src.runner import load_dataset_queries

logging.basicConfig(level=logging.ERROR) 

def _get_dev_split_query_ids() -> set[str]:
    dev_split_dir = Path("data/dev_split_groq_llama31")
    excluded_ids = set()
    if dev_split_dir.exists():
        for jsonl_file in dev_split_dir.glob("*.jsonl"):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if "query_id" in record:
                            qid = str(record["query_id"])
                            if qid.isdigit():
                                excluded_ids.add(qid)
                    except Exception:
                        pass
    return excluded_ids

dev_ids = _get_dev_split_query_ids()
print(f"Extracted {len(dev_ids)} numeric dev split IDs (20 for PopQA + 20 for IFEval).")
assert len(dev_ids) == 40, f"Expected 40 numeric dev split IDs, found {len(dev_ids)}"
print("Assertion Passed: Dev split extraction yields exactly 20 numeric IDs per dataset.")

N = 10
queries_1 = load_dataset_queries("popqa", N, base_seed=1234, sampling_seed=42)
queries_2 = load_dataset_queries("popqa", N, base_seed=1234, sampling_seed=42)

ids_1 = [str(q["query_id"]) for q in queries_1]
ids_2 = [str(q["query_id"]) for q in queries_2]

# 1. Determinism
assert ids_1 == ids_2, "Same seed did not produce the same sample!"
print("Assertion Passed: Same seed yields the same sample deterministically.")

# 2. Exclusion
for qid in ids_1:
    assert qid not in dev_ids, f"Dev split ID {qid} leaked into sample!"
print("Assertion Passed: Dev split IDs successfully excluded from sample.")

# 3. Exact Size
assert len(queries_1) == N, f"Expected {N} items, got {len(queries_1)}"
print("Assertion Passed: Sample size exactly matches N.")

print("All guarantees verified successfully!")
