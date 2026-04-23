import csv
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HW_CONFIG_PATH = PROJECT_ROOT / "configs" / "hardware_config.yaml"


def load_interleaved_params():
    if HW_CONFIG_PATH.exists():
        with open(HW_CONFIG_PATH) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    return {
        "num_users": int(raw.get("interleaved_num_users", 3)),
        "turns_per_user": int(raw.get("interleaved_turns_per_user", 24)),
        "tokens_per_turn": int(raw.get("interleaved_tokens_per_turn", raw.get("chunk_size_tokens", 256))),
        "max_history_chunks": int(raw.get("interleaved_max_history_chunks_per_user", 12)),
    }

def generate_interleaved_trace(
    output_path,
    num_users=3,
    turns_per_user=24,
    tokens_per_turn=256,
    max_history_chunks=12,
):
    trace_rows = []
    
    # Track the chunks belonging to each user
    user_chunks = {i: [] for i in range(num_users)}
    
    query_id = 1
    next_chunk_id = 0
    
    for turn in range(turns_per_user):
        for user_idx in range(num_users):
            # Each user's turn adds 1 new chunk to their context (simulating a turn)
            # and they need all previous chunks plus the new one.
            
            # Create a new chunk for this turn
            new_chunk_id = next_chunk_id
            next_chunk_id += 1
            
            # Keep only recent history chunks so query chunk counts stay realistic.
            shared_ids = list(user_chunks[user_idx])[-max(0, max_history_chunks - 1):]
            unique_ids = [new_chunk_id]
            all_chunks = shared_ids + unique_ids
            
            # Update user's chunk history
            user_chunks[user_idx].append(new_chunk_id)
            
            # Approximate input tokens from configured chunk-equivalent turn size.
            input_tokens = len(all_chunks) * tokens_per_turn
            
            trace_rows.append({
                "query_id": query_id,
                "query_text": f"User {user_idx} Question {turn}",
                "input_tokens": input_tokens,
                "chunk_ids_needed": json.dumps(all_chunks),
                "shared_chunk_ids": json.dumps(shared_ids),
                "unique_chunk_ids": json.dumps(unique_ids),
                "num_chunks": len(all_chunks),
            })
            query_id += 1
            
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trace_rows[0].keys())
        writer.writeheader()
        writer.writerows(trace_rows)
        
    print(f"Interleaved trace generated at {output_path} with {len(trace_rows)} queries.")

if __name__ == "__main__":
    params = load_interleaved_params()
    generate_interleaved_trace(
        "data/traces_interleaved.csv",
        num_users=params["num_users"],
        turns_per_user=params["turns_per_user"],
        tokens_per_turn=params["tokens_per_turn"],
        max_history_chunks=params["max_history_chunks"],
    )
