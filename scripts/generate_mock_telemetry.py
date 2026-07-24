"""Generate mock telemetry JSONL files for pipeline testing.
Produces files in telemetry/ matching runner.py output format."""
import json
from pathlib import Path

MOCK_DATA = {
    "popqa": [
        {"query_id": "p1", "final_answer": "Paris",       "u": 0.1, "correct": True},
        {"query_id": "p2", "final_answer": "London",      "u": 0.3, "correct": False},
        {"query_id": "p3", "final_answer": "Berlin",      "u": 0.5, "correct": False},
        {"query_id": "p4", "final_answer": "Madrid",      "u": 0.0, "correct": True},
        {"query_id": "p5", "final_answer": "Rome",        "u": 0.2, "correct": True},
        {"query_id": "p6", "final_answer": "Lisbon",      "u": 0.8, "correct": False},
        {"query_id": "p7", "final_answer": "Vienna",      "u": 0.9, "correct": False},
        {"query_id": "p8", "final_answer": "Athens",      "u": 0.1, "correct": True},
    ],
    "ifeval": [
        {"query_id": "i1", "final_answer": "Hello World!!",        "u": 0.0, "correct": True},
        {"query_id": "i2", "final_answer": "Hi",                   "u": 0.2, "correct": False},
        {"query_id": "i3", "final_answer": "Yes indeed sir",       "u": 0.1, "correct": True},
        {"query_id": "i4", "final_answer": "No",                   "u": 0.4, "correct": False},
        {"query_id": "i5", "final_answer": "A B C D E F G H I J",  "u": 0.0, "correct": True},
        {"query_id": "i6", "final_answer": "X",                    "u": 0.6, "correct": False},
    ],
}

MOCK_REFERENCES = {
    "popqa": [
        {"query_id": "p1", "answer": "Paris"},
        {"query_id": "p2", "answer": "Paris"},
        {"query_id": "p3", "answer": "Berlin"},
        {"query_id": "p4", "answer": "Madrid"},
        {"query_id": "p5", "answer": "Rome"},
        {"query_id": "p6", "answer": "Lisbon"},
        {"query_id": "p7", "answer": "Vienna"},
        {"query_id": "p8", "answer": "Athens"},
    ],
    "ifeval": [
        {"query_id": "i1", "prompt": "Generate a response with at least 10 characters"},
        {"query_id": "i2", "prompt": "Generate a response with at least 10 characters"},
        {"query_id": "i3", "prompt": "Generate a response with at least 5 characters"},
        {"query_id": "i4", "prompt": "Generate a response with at least 3 characters"},
        {"query_id": "i5", "prompt": "Generate a response with at least 10 characters"},
        {"query_id": "i6", "prompt": "Generate a response with at least 3 characters"},
    ],
}

def main():
    out_dir = Path("telemetry")
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset, records in MOCK_DATA.items():
        for pattern_id in [1, 2, 3, 4]:
            filename = out_dir / f"{dataset}_pattern{pattern_id}.jsonl"
            with open(filename, "w", encoding="utf-8") as f:
                for rec in records:
                    base_u = rec["u"]
                    u_L_val = base_u * 0.5
                    u_F_val = base_u * 0.3
                    u_C_val = base_u if pattern_id == 4 else None
                    u_R_val = base_u * 0.2 if pattern_id != 1 else None
                    u_P_val = base_u * 0.2
                    U_surface = round((u_L_val + u_F_val + u_P_val) / 3, 4)

                    if pattern_id == 1:
                        U_composite = round(0.33 * u_L_val + 0.33 * u_F_val + 0.34 * u_P_val, 4)
                    elif pattern_id in (2, 3):
                        U_composite = round(0.25 * u_L_val + 0.25 * u_F_val + 0.25 * u_P_val + 0.25 * u_R_val, 4)
                    else:
                        U_composite = round(0.20 * u_L_val + 0.20 * u_F_val + 0.20 * u_P_val + 0.20 * u_R_val + 0.20 * u_C_val, 4)

                    telemetry = {
                        "run_id": f"{rec['query_id']}_pattern{pattern_id}",
                        "pattern": pattern_id,
                        "query_id": rec["query_id"],
                        "dataset": dataset,
                        "seed": 1234,
                        "model_name": "google/gemma-4-12b-it",
                        "provider": "vllm",
                        "final_answer": rec["final_answer"],
                        "termination_reason": "completed",
                        "tokens_prompt": 50,
                        "tokens_completion": 20,
                        "latency_ms": 1500.0,
                        "u_L": u_L_val,
                        "u_F": u_F_val,
                        "u_C": u_C_val,
                        "u_R": u_R_val,
                        "u_P": u_P_val,
                        "U_surface": U_surface,
                        "U_composite": U_composite,
                        "truncation_flag": False,
                    }
                    f.write(json.dumps(telemetry) + "\n")
            print(f"Wrote {filename} ({len(records)} records)")

    # Write reference files
    ref_dir = Path("telemetry/references")
    ref_dir.mkdir(parents=True, exist_ok=True)
    for dataset, refs in MOCK_REFERENCES.items():
        ref_path = ref_dir / f"{dataset}_references.jsonl"
        with open(ref_path, "w", encoding="utf-8") as f:
            for ref in refs:
                f.write(json.dumps(ref) + "\n")
        print(f"Wrote {ref_path}")

if __name__ == "__main__":
    main()
