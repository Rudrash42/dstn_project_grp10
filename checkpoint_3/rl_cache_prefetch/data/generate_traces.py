#!/usr/bin/env python3
"""
Generate synthetic trace datasets for RL training.
Creates 4 CSV files (one per workload) + a TTFT lookup JSON.

Each trace models chunk-level access patterns derived from
the real experiment data in checkpoint_2/results/.

Usage:
    python data/generate_traces.py
"""

import json
import math
import csv
import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — derived from real experiments
# ═══════════════════════════════════════════════════════════════

CHUNK_SIZE_TOKENS = 256
NUM_QUERIES = 50

# Output paths (relative to this script's directory)
DATA_DIR = Path(__file__).resolve().parent

# ── Real TTFT data from experiment_results_4.csv ──────────────
# Shared Prefix: ~163 tokens, Cold TTFT ~100ms, Warm ~57ms
# RAG Shared Doc: ~1750 tokens, Cold TTFT ~238ms, Warm ~70ms
# No Context:     ~8 tokens, Cold TTFT ~60ms (always cold)
# Multi-Turn:     65→1418 tokens, Cold TTFT ~61ms, Warm grows with length

TTFT_LOOKUP = {
    "shared_prefix": {
        "cold_ttft_ms": 100.0,
        "warm_ttft_ms": 57.0,
        "avg_input_tokens": 166,
        "shared_prefix_tokens": 160,  # ~1 chunk shared
    },
    "rag": {
        "cold_ttft_ms": 238.0,
        "warm_ttft_ms": 70.0,
        "avg_input_tokens": 1750,
        "shared_doc_tokens": 1700,  # ~7 chunks shared
    },
    "nocontext": {
        "cold_ttft_ms": 60.0,
        "warm_ttft_ms": 60.0,  # no sharing, always cold
        "avg_input_tokens": 8,
    },
    "multiturn": {
        "cold_ttft_ms": 61.0,
        "warm_ttft_ms_base": 61.0,
        "base_tokens": 65,
        "tokens_per_turn": 27,  # ~27 tokens added per turn
    },
}

# ── Query texts (from run_experiments.py) ─────────────────────

PREFIX_QUESTIONS = [
    "What exercises help with lower back pain?",
    "Is applying ice effective for reducing swelling?",
    "Define correct sitting posture for office workers.",
    "How long should a rotator cuff tear rehabilitation programme last?",
    "What is the recommended rest period after an acute ankle sprain?",
    "Which stretches are most effective for tight hamstrings?",
    "How should a patient progress from non-weight-bearing to full weight-bearing?",
    "What is the role of proprioception training after ACL reconstruction?",
    "How many sets and reps are recommended for quadriceps post surgery?",
    "What are the signs that a patient is overtraining during rehabilitation?",
    "How effective is dry needling for myofascial pain syndrome?",
    "What is the difference between active and passive physiotherapy?",
    "When is it safe to return to sport after a hamstring strain?",
    "How does ultrasound therapy aid soft tissue healing?",
    "What are the best exercises for strengthening the hip abductors?",
    "How should breathing be coordinated during core stability exercises?",
    "What is the McKenzie method and when is it indicated?",
    "How do wearable sensors improve rehabilitation outcomes?",
    "What is the recommended frequency of physiotherapy for chronic neck pain?",
    "How does foam rolling affect muscle recovery?",
    "What are the early mobilisation protocols after total knee replacement?",
    "How should a patient warm up before starting rehabilitation exercises?",
    "What is the evidence for kinesiology taping in shoulder impingement?",
    "How does sleep quality affect musculoskeletal recovery?",
    "What exercises are contraindicated after lumbar discectomy?",
    "How is gait analysis used in rehabilitation planning?",
    "What is the role of hydrotherapy in post-surgical rehabilitation?",
    "How long does it take to recover from a grade 2 ligament sprain?",
    "What are the benefits of eccentric training for tendinopathy?",
    "How should rehabilitation differ for elderly patients with hip fractures?",
    "What is the Oswestry Disability Index used for?",
    "How can computer vision detect incorrect squat form?",
    "What are the clinical criteria for diagnosing patellofemoral pain?",
    "How does chronic pain affect rehabilitation adherence?",
    "What is the recommended load progression for Achilles tendinopathy?",
    "How effective is TENS therapy for post-operative pain management?",
    "What is the difference between isometric and isotonic exercises?",
    "How should rehabilitation be modified for diabetic neuropathy patients?",
    "What are the red flags in low back pain requiring immediate referral?",
    "How does obesity affect joint loading during rehabilitation?",
    "What is neuromuscular electrical stimulation and when is it used?",
    "How can a patient self-monitor exercise intensity at home?",
    "What are the stages of tissue healing and how do they guide treatment?",
    "How effective is spinal manipulation for non-specific low back pain?",
    "What is the role of the transverse abdominis in lumbar stability?",
    "How does stress and anxiety impact musculoskeletal pain perception?",
    "What is the minimal detectable change for the Visual Analogue Scale?",
    "How should rehabilitation be adapted for osteoporosis patients?",
    "What are the best outcome measures for shoulder rehabilitation?",
    "How does dehydration affect muscle performance during exercise?",
]

RAG_QUESTIONS = [
    "What are the core vision and mission statements of Tongaat Hulett?",
    "Summarize the geographic footprint of Tongaat Hulett's operations.",
    "What was the total volume of sugar produced in the 2021 financial year?",
    "How many people were employed at the peak of the milling season?",
    "What percentage of Board members were non-executive and independent?",
    "Explain the Manufactured and Financial capitals in the business model.",
    "What were the key focus areas for 2022 to enable operational excellence?",
    "Summarize the impact of COVID-19 on operations and workforce in 2021.",
    "How much was invested in COVID-19 avoidance and treatment in 2021?",
    "Describe the Social Ethics Health and Safety Committee responsibilities.",
    "What is the energy intensity reduction target by year 2025?",
    "List the market-leading brands for sugar and animal feeds.",
    "What was the total SED expenditure in 2021?",
    "Detail the change in scope 1 and scope 2 carbon emissions 2020-2021.",
    "How does Tongaat Hulett define and manage its Intellectual Capital?",
    "What were the primary environmental efficiency investments in 2021?",
    "Summarize the 2021 corporate reputation survey findings.",
    "What are the water efficiency improvement targets by 2025?",
    "Explain the relationship with small-scale growers including volumes.",
    "What was the LTIFR for 2021?",
    "Describe the approach to human rights and child labor in supply chain.",
    "What were total hazardous and non-hazardous waste volumes in 2021?",
    "List the third-party certifications held such as ISO 45001.",
    "How does the company align with UN Sustainable Development Goals?",
    "What was the total revenue for the 2021 financial year?",
    "Explain the significance of the Sugar Industry Masterplan.",
    "Detail the training and development spend for employees in 2021.",
    "What are the primary climate change risks identified?",
    "Summarize the Success Management programme for performance.",
    "Who provides independent external assurance for the ESG report?",
    "What is the stated purpose of Implats in the 2023 ESG report?",
    "Name the managed operations in the Implats 2023 ESG report scope.",
    "What was the LTIFR for the Implats Group in 2023?",
    "Detail the total value distributed to stakeholders in 2023.",
    "What percentage of the Implats Board identifies as female?",
    "Summarize the CEO's statement on safety and zero harm.",
    "What are the three pillars of the Implats ESG framework?",
    "How much was invested in SED and community projects in 2023?",
    "Explain the double materiality principle in Implats ESG reports.",
    "What are the primary metals produced and their uses?",
    "Describe the 35MW solar PV project progress at Zimplats.",
    "What was the water recycling and reuse rate achieved in 2023?",
    "Detail the We Care programme support for employee families.",
    "What were key employee outcomes in wages and benefits in 2023?",
    "Explain the PS3 strategy and sustainability alignment.",
    "What is the carbon emissions reduction target by 2030?",
    "Describe SO2 abatement technology installation at Zimplats.",
    "What external ESG ratings were received from MSCI and S&P Global?",
    "How does Implats manage air quality and reduce emissions?",
    "Summarize the RBPlat acquisition significance for Western Limb.",
]

NO_CONTEXT_QUESTIONS = [
    "Who is the founder of Microsoft?",
    "What is the chemical symbol for sodium?",
    "How many players are there in a baseball team?",
    "What year did World War II end?",
    "What is the tallest species of tree?",
    "Who wrote The Divine Comedy?",
    "What is the hardest rock type?",
    "How many hearts does a squid have?",
    "What is the main ingredient in guacamole?",
    "Who developed the polio vaccine?",
    "What is the longest-running Broadway show?",
    "What is the square of 25?",
    "Who was the first woman to win a Nobel Prize?",
    "What is the currency of Brazil?",
    "What gas do plants absorb during photosynthesis?",
    "Who directed the movie Jaws?",
    "What is the largest internal organ in the human body?",
    "How many elements are in the periodic table?",
    "What is the freezing point of mercury in Celsius?",
    "Who painted The School of Athens?",
    "What is the smallest unit of life?",
    "How many time zones are there in Russia?",
    "What is the main language spoken in Argentina?",
    "Who invented the diesel engine?",
    "What is the diameter of Earth in kilometers?",
    "What is the rarest blood type?",
    "Who composed The Magic Flute?",
    "What is the powerhouse of a computer?",
    "How many sides does a dodecagon have?",
    "What is the largest species of penguin?",
    "Who discovered the planet Neptune?",
    "What is the boiling point of nitrogen in Celsius?",
    "What is the fastest bird in a dive?",
    "Who wrote The Brothers Karamazov?",
    "What is the largest artery in the human body?",
    "How many keys are on a standard computer keyboard?",
    "What is the smallest country by population?",
    "Who invented the telescope?",
    "What is the main component of natural gas?",
    "How many amendments are in the U.S. Constitution?",
    "What is the deepest lake in the world?",
    "Who was the first emperor of China?",
    "What is the currency of South Africa?",
    "What is the longest bone in the arm?",
    "Who discovered the circulation of blood?",
    "What is the primary ingredient in hummus?",
    "How many moons does Mars have?",
    "What is the largest species of cat?",
    "Who wrote The Picture of Dorian Gray?",
    "What is the chemical formula for methane?",
]

MULTITURN_QUESTIONS = [
    "Can you explain how the RehabQuest pose tracking works?",
    "What is the calibration procedure at the start of each session?",
    "How does T-pose calibration normalise body proportions?",
    "What are the hardware requirements for real-time pose tracking?",
    "How does MediaPipe detect the 33 pose landmarks?",
    "What is the role of cosine similarity in joint angle computation?",
    "How are joint angles computed in three dimensions?",
    "How is the system validated against Vicon motion capture?",
    "What accuracy metrics are used to evaluate performance?",
    "How does camera distance affect landmark detection accuracy?",
    "What GPU specification is needed for real-time processing?",
    "How does the system handle occlusion of body parts?",
    "What frame rate achieves clinically acceptable motion tracking?",
    "How are left and right side landmarks differentiated?",
    "What happens if T-pose calibration is performed incorrectly?",
    "How does the system account for varying patient heights?",
    "Can pose tracking work with a standard RGB webcam?",
    "How are the 33 landmarks mapped to anatomical joints?",
    "What filtering is applied to raw landmark coordinates?",
    "How does the system detect and reject outlier frames?",
    "How is shoulder flexion angle calculated from landmarks?",
    "How is knee extension range of motion extracted?",
    "What is the typical latency from movement to feedback?",
    "How does lighting affect landmark detection confidence?",
    "What confidence threshold accepts or rejects a landmark?",
    "How does the system track spinal alignment during exercises?",
    "How are exercise repetitions counted from angle time series?",
    "What ML model underlies the MediaPipe pose estimator?",
    "How was the MediaPipe model trained and what datasets used?",
    "Can the system distinguish correct from compensatory movement?",
    "How is data from multiple sessions stored and compared?",
    "What data format exports session results for clinicians?",
    "How does the system perform with limb prosthetics?",
    "What are failure modes of cosine similarity angle computation?",
    "How is the world coordinate frame defined relative to camera?",
    "How handle patients who cannot perform the initial T-pose?",
    "What is the mean absolute error vs Vicon gold standard?",
    "How does clothing affect landmark detection accuracy?",
    "Can multiple cameras improve tracking accuracy?",
    "How are upper and lower extremity exercises treated differently?",
    "What happens when patient moves out of camera frame?",
    "How is the skeleton model re-initialised after tracking loss?",
    "How are hip joint angles computed and which landmarks used?",
    "What is the difference between 2D and 3D landmarks?",
    "How does the system calculate left-right symmetry scores?",
    "What network architecture is used for landmark regression?",
    "How are progress reports generated from session data?",
    "Can the system operate offline without internet?",
    "How is patient privacy protected for video and landmark data?",
    "What future improvements are planned for pose tracking?",
]


# ═══════════════════════════════════════════════════════════════
# TRACE GENERATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def generate_prefix_trace(output_path: Path):
    """
    Shared Prefix workload: all queries share a common prefix (~1 chunk),
    each query adds a small unique segment (~0-1 chunks).
    """
    shared_prefix_chunks = math.ceil(160 / CHUNK_SIZE_TOKENS)  # 1 chunk
    global_chunk_counter = shared_prefix_chunks  # unique chunks start after shared

    rows = []
    for i in range(NUM_QUERIES):
        question = PREFIX_QUESTIONS[i % len(PREFIX_QUESTIONS)]
        # Each question adds ~5-15 tokens beyond the prefix → 0 extra chunks
        # (they fit within the last prefix chunk or form 1 new chunk)
        unique_chunk_count = 1 if (i % 3 == 0) else 0
        unique_ids = list(range(global_chunk_counter,
                                global_chunk_counter + unique_chunk_count))
        global_chunk_counter += unique_chunk_count

        shared_ids = list(range(shared_prefix_chunks))
        all_chunks = shared_ids + unique_ids
        input_tokens = 160 + len(question.split()) * 2  # rough estimate

        rows.append({
            "query_id": i + 1,
            "query_text": question,
            "input_tokens": input_tokens,
            "chunk_ids_needed": json.dumps(all_chunks),
            "shared_chunk_ids": json.dumps(shared_ids),
            "unique_chunk_ids": json.dumps(unique_ids),
            "num_chunks": len(all_chunks),
        })

    _write_csv(output_path, rows)
    print(f"  [prefix]    {len(rows)} queries, {shared_prefix_chunks} shared chunks → {output_path.name}")


def generate_rag_trace(output_path: Path):
    """
    RAG workload: all queries share a long document (~7 chunks),
    each query adds 1 unique chunk for the question itself.
    """
    shared_doc_chunks = math.ceil(1700 / CHUNK_SIZE_TOKENS)  # 7 chunks
    global_chunk_counter = shared_doc_chunks

    rows = []
    for i in range(NUM_QUERIES):
        question = RAG_QUESTIONS[i % len(RAG_QUESTIONS)]
        unique_ids = [global_chunk_counter]
        global_chunk_counter += 1

        shared_ids = list(range(shared_doc_chunks))
        all_chunks = shared_ids + unique_ids

        rows.append({
            "query_id": i + 1,
            "query_text": question,
            "input_tokens": 1750 + np.random.randint(-20, 20),
            "chunk_ids_needed": json.dumps(all_chunks),
            "shared_chunk_ids": json.dumps(shared_ids),
            "unique_chunk_ids": json.dumps(unique_ids),
            "num_chunks": len(all_chunks),
        })

    _write_csv(output_path, rows)
    print(f"  [rag]       {len(rows)} queries, {shared_doc_chunks} shared chunks → {output_path.name}")


def generate_nocontext_trace(output_path: Path):
    """
    No-Context workload: every query is independent, no chunk sharing.
    """
    global_chunk_counter = 0

    rows = []
    for i in range(NUM_QUERIES):
        question = NO_CONTEXT_QUESTIONS[i % len(NO_CONTEXT_QUESTIONS)]
        chunk_id = global_chunk_counter
        global_chunk_counter += 1

        rows.append({
            "query_id": i + 1,
            "query_text": question,
            "input_tokens": max(5, len(question.split()) + np.random.randint(-2, 3)),
            "chunk_ids_needed": json.dumps([chunk_id]),
            "shared_chunk_ids": json.dumps([]),
            "unique_chunk_ids": json.dumps([chunk_id]),
            "num_chunks": 1,
        })

    _write_csv(output_path, rows)
    print(f"  [nocontext] {len(rows)} queries, 0 shared chunks → {output_path.name}")


def generate_multiturn_trace(output_path: Path):
    """
    Multi-Turn Chat: each turn shares all previous turns' chunks,
    plus adds 1 new chunk for the new question + response.
    """
    rows = []
    all_accumulated_chunks = []

    for i in range(NUM_QUERIES):
        question = MULTITURN_QUESTIONS[i % len(MULTITURN_QUESTIONS)]
        new_chunk_id = i  # each turn adds exactly one new chunk
        all_accumulated_chunks.append(new_chunk_id)

        shared_ids = all_accumulated_chunks[:-1]  # all previous
        unique_ids = [new_chunk_id]
        input_tokens = 65 + i * 27  # grows linearly

        rows.append({
            "query_id": i + 1,
            "query_text": question,
            "input_tokens": input_tokens,
            "chunk_ids_needed": json.dumps(list(all_accumulated_chunks)),
            "shared_chunk_ids": json.dumps(shared_ids),
            "unique_chunk_ids": json.dumps(unique_ids),
            "num_chunks": len(all_accumulated_chunks),
        })

    _write_csv(output_path, rows)
    print(f"  [multiturn] {len(rows)} queries, growing shared set → {output_path.name}")


def _write_csv(path: Path, rows: list):
    """Write a list of dicts to a CSV file."""
    if not rows:
        return
    fieldnames = rows[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    print("\n[generate_traces] Creating synthetic trace datasets...")

    generate_prefix_trace(DATA_DIR / "traces_prefix.csv")
    generate_rag_trace(DATA_DIR / "traces_rag.csv")
    generate_nocontext_trace(DATA_DIR / "traces_nocontext.csv")
    generate_multiturn_trace(DATA_DIR / "traces_multiturn.csv")

    # Save TTFT lookup
    ttft_path = DATA_DIR / "ttft_lookup.json"
    with open(ttft_path, "w") as f:
        json.dump(TTFT_LOOKUP, f, indent=2)
    print(f"  [ttft]      TTFT lookup → {ttft_path.name}")

    print("[generate_traces] Done!\n")


if __name__ == "__main__":
    main()
