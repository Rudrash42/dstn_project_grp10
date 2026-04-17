import csv
import json

def generate_interleaved_trace(output_path, num_users=3, turns_per_user=33):
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
            
            # The needed chunks for this user is their previous chunks + the new one
            shared_ids = list(user_chunks[user_idx])
            unique_ids = [new_chunk_id]
            all_chunks = shared_ids + unique_ids
            
            # Update user's chunk history
            user_chunks[user_idx].append(new_chunk_id)
            
            # Approximate input tokens (say 200 tokens per chunk)
            input_tokens = len(all_chunks) * 200
            
            trace_rows.append({
                "query_id": query_id,
                "query_text": f"Synthetic interleaved workload | user={user_idx} turn={turn}",
                "embedding_text": f"Synthetic interleaved workload | user={user_idx} turn={turn}",
                "focus_text": f"User {user_idx} Question {turn}",
                "context_profile": "synthetic_interleaved_stress_test",
                "prompt_text": f"User {user_idx} Question {turn}",
                "prompt_preview": f"User {user_idx} Question {turn}",
                "input_tokens": input_tokens,
                "chunk_ids_needed": json.dumps(all_chunks),
                "shared_chunk_ids": json.dumps(shared_ids),
                "unique_chunk_ids": json.dumps(unique_ids),
                "num_chunks": len(all_chunks),
                "chunk_id_source": "synthetic",
            })
            query_id += 1
            
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trace_rows[0].keys())
        writer.writeheader()
        writer.writerows(trace_rows)
        
    print(f"Interleaved trace generated at {output_path} with {len(trace_rows)} queries.")

if __name__ == "__main__":
    generate_interleaved_trace("data/traces_interleaved.csv", num_users=3, turns_per_user=33)
