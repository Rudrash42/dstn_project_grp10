#!/usr/bin/env python3
"""
Generate trace datasets for RL training using the Hardware Cache backend.

NO vLLM, NO LMCache.  Traces are generated synthetically based on
realistic workload patterns, and a hardware calibration pass measures
REAL latencies on your GPU / CPU / Disk so the RL agent trains on
accurate numbers.

Produces:
  - 4 trace CSVs (one per workload: prefix, rag, nocontext, multiturn)
  - ttft_lookup.json  (measured cold/warm latencies per tier)
  - Updated hardware_config.yaml with calibrated latency values
  - calibration_report.csv  (detailed per-tier latency measurements)
  - 4 embedding .npy files (sentence-transformer embeddings for queries)

Usage:
    python data/generate_traces.py                 # full run
    python data/generate_traces.py --cpu-only      # no GPU required
    python data/generate_traces.py --calibrate-only # just measure latencies
"""

import csv
import json
import math
import os
import sys
import time

import numpy as np
import yaml
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_DIR.parent
HW_CONFIG_PATH = PROJECT_ROOT / "configs" / "hardware_config.yaml"
PPO_CONFIG_PATH = PROJECT_ROOT / "configs" / "ppo_config.yaml"

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION (loaded from hardware_config.yaml + defaults)
# ═══════════════════════════════════════════════════════════════

# Defaults — overridden by YAML if present
MAX_QUERIES = 50
CHUNK_SIZE_TOKENS = 256

# Workload shape parameters (can also come from YAML)
PREFIX_LENGTH_TOKENS = 600    # shared prefix ≈ 2-3 chunks
RAG_DOC_LENGTH_TOKENS = 1700  # shared RAG doc ≈ 7 chunks
MULTITURN_TOKENS_PER_TURN = 40
QUESTION_TOKENS_AVG = 15      # average unique question length
CALIBRATION_ITERATIONS = 20   # how many chunks to test per tier

# ═══════════════════════════════════════════════════════════════
# QUERY DATA (kept for embedding generation & query_text column)
# ═══════════════════════════════════════════════════════════════

EXP1_PREFIX = (
    "You are an expert Physiotherapist AI assistant for the RehabQuest platform. "
    "Your role is to provide concise, evidence-based rehabilitation guidance "
    "grounded in peer-reviewed clinical literature and WHO recommendations. "
    "Always cite relevant studies when possible. Limit answers to three paragraphs. "
    "Use metric units. If the question is outside your expertise, state so clearly. "
    "RehabQuest is a healthcare startup specialising in AI-driven musculoskeletal "
    "rehabilitation using computer vision and wearable sensors. The platform tracks "
    "patient exercises in real time, provides corrective feedback, and generates "
    "progress reports for clinicians. You must adhere to HIPAA guidelines and never "
    "provide a definitive diagnosis. Always recommend that patients consult their "
    "treating physician for personalised medical advice before changing their "
    "rehabilitation programme. Respond in professional but approachable language. "
    "Clinical Practice Guidelines: For musculoskeletal rehabilitation, the American "
    "Physical Therapy Association (APTA) recommends a structured, phase-based approach: "
    "Phase I (Acute, Days 0-7): Focus on pain management using cryotherapy, compression, "
    "and elevation. Introduce gentle range-of-motion exercises within pain tolerance. "
    "Apply the PRICE protocol (Protection, Rest, Ice, Compression, Elevation). Monitor "
    "for signs of deep vein thrombosis in immobilised patients. Document baseline pain "
    "levels using the Visual Analogue Scale (VAS). "
    "Phase II (Subacute, Weeks 1-6): Progressive loading following the tissue healing "
    "timeline. Introduce isometric exercises progressing to isotonic by week 3. Begin "
    "proprioceptive training with balance boards and unstable surfaces. Target 80% of "
    "contralateral limb strength before advancing. Monitor inflammatory markers and adjust "
    "intensity accordingly. Apply the principle of graduated return to activity. "
    "Phase III (Remodelling, Weeks 6-12): Sport-specific or task-specific training. "
    "Eccentric strengthening for tendinopathies following the Alfredson protocol. "
    "Plyometric progression using the reactive strength index. Functional movement "
    "screening (FMS) to identify compensatory patterns. Return-to-sport criteria: "
    "90% limb symmetry index on isokinetic testing, successful completion of hop tests "
    "(single, triple, crossover, and timed), and psychological readiness assessed via "
    "the ACL-Return to Sport after Injury (ACL-RSI) scale. "
    "Phase IV (Maintenance, Ongoing): Long-term injury prevention programming. "
    "Periodised strength and conditioning with progressive overload. Neuromuscular "
    "control drills integrated into warm-up routines following the FIFA 11+ protocol. "
    "Annual functional reassessment recommended. Patient education on load management, "
    "sleep hygiene, and nutritional support for tissue recovery. "
    "Documentation Standards: Use the International Classification of Functioning, "
    "Disability and Health (ICF) framework for assessment documentation. Record "
    "objective measures including goniometric range of motion, manual muscle testing "
    "grades (Oxford scale 0-5), and validated patient-reported outcome measures "
    "(PROMs) such as the Lower Extremity Functional Scale (LEFS), Disabilities of "
    "the Arm, Shoulder and Hand (DASH), and Oswestry Disability Index (ODI). "
    "Now answer the following clinical question. "
)

EXP1_QUESTIONS = [
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

EXP4_BASE_HISTORY = "User: Hello AI.\nAssistant: Hi there! How can I help you today?\n"


# ═══════════════════════════════════════════════════════════════
# HARDWARE CALIBRATION
# ═══════════════════════════════════════════════════════════════

def calibrate_hardware(force_cpu=False, iterations=None):
    """
    Run a calibration pass on real hardware to measure actual latencies
    for L1 (GPU), L2 (CPU), L3 (Disk), and cold-miss operations.

    Returns a dict of measured latencies in milliseconds.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from env.hardware.hardware_config import HardwareConfig
    from env.hardware.hardware_cache import HardwareCache

    # Load config
    cfg = HardwareConfig.from_yaml(HW_CONFIG_PATH)
    if force_cpu:
        cfg.force_cpu_mode = True
    cfg.verbose = False
    cfg.enable_operation_log = True

    if iterations is None:
        iterations = CALIBRATION_ITERATIONS

    cache = HardwareCache(cfg)
    print(f"\n  [calibrate] Hardware: {'CUDA GPU' if cache.use_cuda else 'CPU-only'}")
    print(f"  [calibrate] L1={cfg.l1_capacity_mb}MB ({cfg.l1_capacity_chunks} chunks) "
          f"L2={cfg.l2_capacity_mb}MB ({cfg.l2_capacity_chunks} chunks)")
    print(f"  [calibrate] Running {iterations} iterations per tier...")

    results = {
        "l1_hit_ms": [],
        "l2_hit_ms": [],
        "l3_hit_ms": [],
        "cold_miss_ms": [],
        "prefetch_l3_to_l2_ms": [],
        "evict_l1_to_l2_ms": [],
        "evict_l2_to_l3_ms": [],
    }

    # ── Phase 1: Measure cold-miss latency ──
    # Insert brand new chunks to measure creation + GPU allocation time
    cache.reset()
    for i in range(iterations):
        cid = 10000 + i
        tier, latency = cache.access_chunk(cid)
        assert tier == "MISS", f"Expected MISS but got {tier}"
        results["cold_miss_ms"].append(latency)

    # ── Phase 2: Measure L1 hit latency ──
    # Access chunks that are already in L1
    # First, ensure some chunks are in L1
    cache.reset()
    l1_chunks = min(iterations, cfg.l1_capacity_chunks)
    for cid in range(l1_chunks):
        cache.insert_chunks([cid])

    for cid in range(l1_chunks):
        tier, latency = cache.access_chunk(cid)
        if tier == "L1":
            results["l1_hit_ms"].append(latency)

    # ── Phase 3: Measure L2 hit latency ──
    # Fill L1 to overflow some chunks to L2, then access them
    cache.reset()
    total_to_fill = cfg.l1_capacity_chunks + min(iterations, cfg.l2_capacity_chunks)
    for cid in range(total_to_fill):
        cache.insert_chunks([cid])

    # The first chunks should have been evicted to L2
    for cid in range(min(iterations, cfg.l1_capacity_chunks)):
        tier_before = cache.chunk_in_cache(cid)
        if tier_before == "L2":
            tier, latency = cache.access_chunk(cid)
            results["l2_hit_ms"].append(latency)

    # ── Phase 4: Measure L3 hit latency ──
    # Fill L1+L2 to overflow chunks to L3 (disk), then access them
    cache.reset()
    total_to_fill = cfg.l1_capacity_chunks + cfg.l2_capacity_chunks + iterations
    for cid in range(total_to_fill):
        cache.insert_chunks([cid])

    # The first chunks should have been evicted to L3
    for cid in range(iterations):
        tier_before = cache.chunk_in_cache(cid)
        if tier_before == "L3":
            tier, latency = cache.access_chunk(cid)
            results["l3_hit_ms"].append(latency)

    # ── Phase 5: Measure prefetch latency (L3 → L2) ──
    cache.reset()
    total_to_fill = cfg.l1_capacity_chunks + cfg.l2_capacity_chunks + iterations
    for cid in range(total_to_fill):
        cache.insert_chunks([cid])

    for cid in range(iterations):
        tier_before = cache.chunk_in_cache(cid)
        if tier_before == "L3":
            cost = cache.prefetch(cid)
            if cost > 0:
                results["prefetch_l3_to_l2_ms"].append(cost)

    # Clean up
    cache.reset()

    # ── Compute statistics ──
    calibrated = {}
    for key, values in results.items():
        if values:
            calibrated[key] = {
                "mean": round(float(np.mean(values)), 4),
                "median": round(float(np.median(values)), 4),
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
                "std": round(float(np.std(values)), 4),
                "n_samples": len(values),
            }
        else:
            calibrated[key] = {
                "mean": 0.0, "median": 0.0, "min": 0.0,
                "max": 0.0, "std": 0.0, "n_samples": 0,
            }

    # Print summary
    print(f"\n  [calibrate] Results:")
    for key, stats in calibrated.items():
        if stats["n_samples"] > 0:
            print(f"    {key:30s}: {stats['mean']:>8.4f} ms  "
                  f"(median={stats['median']:.4f}, std={stats['std']:.4f}, "
                  f"n={stats['n_samples']})")
        else:
            print(f"    {key:30s}: NO DATA (tier may not have been populated)")

    return calibrated


def save_calibration_report(calibrated, output_path):
    """Save calibration results as CSV for reference."""
    rows = []
    for key, stats in calibrated.items():
        rows.append({
            "metric": key,
            "mean_ms": stats["mean"],
            "median_ms": stats["median"],
            "min_ms": stats["min"],
            "max_ms": stats["max"],
            "std_ms": stats["std"],
            "n_samples": stats["n_samples"],
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [calibrate] Report saved → {output_path.name}")


# ═══════════════════════════════════════════════════════════════
# TRACE GENERATION — Create workload traces from config
# ═══════════════════════════════════════════════════════════════

def load_config_params():
    """Load trace generation parameters from hardware_config.yaml."""
    if HW_CONFIG_PATH.exists():
        with open(HW_CONFIG_PATH) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    return {
        "chunk_size_tokens": raw.get("chunk_size_tokens", CHUNK_SIZE_TOKENS),
        "kv_bytes_per_token": raw.get("kv_bytes_per_token", 12288),
        "chunk_size_bytes": raw.get("chunk_size_bytes", 3_145_728),
        "max_queries": raw.get("max_queries_per_workload", MAX_QUERIES),
        "prefix_tokens": raw.get("prefix_length_tokens", PREFIX_LENGTH_TOKENS),
        "rag_doc_tokens": raw.get("rag_doc_length_tokens", RAG_DOC_LENGTH_TOKENS),
        "multiturn_tpt": raw.get("multiturn_tokens_per_turn", MULTITURN_TOKENS_PER_TURN),
        "question_tokens_avg": raw.get("question_tokens_avg", QUESTION_TOKENS_AVG),
    }


def generate_prefix_trace(params, questions):
    """
    Workload 1: Shared Prefix.

    All queries share the same system-prompt prefix (≈ 2-3 chunks).
    Each query appends a short unique question (≈ 1 chunk).
    """
    chunk_tok = params["chunk_size_tokens"]
    n = min(params["max_queries"], len(questions))

    # Shared prefix chunks
    prefix_chunks = max(1, math.ceil(params["prefix_tokens"] / chunk_tok))
    shared_ids = list(range(prefix_chunks))
    unique_counter = prefix_chunks

    rows = []
    for i in range(n):
        # Each unique question adds ~1 chunk
        unique_chunk_count = max(1, math.ceil(params["question_tokens_avg"] / chunk_tok))
        unique_ids = list(range(unique_counter, unique_counter + unique_chunk_count))
        unique_counter += unique_chunk_count

        all_chunks = shared_ids + unique_ids
        total_tokens = params["prefix_tokens"] + params["question_tokens_avg"]

        rows.append({
            "query_id": i + 1,
            "query_text": (EXP1_PREFIX + questions[i])[:200],
            "input_tokens": total_tokens,
            "chunk_ids_needed": json.dumps(all_chunks),
            "shared_chunk_ids": json.dumps(shared_ids),
            "unique_chunk_ids": json.dumps(unique_ids),
            "num_chunks": len(all_chunks),
        })

    return rows


def generate_rag_trace(params, questions):
    """
    Workload 2: RAG (Shared Document Context).

    All queries share a large document context (≈ 7 chunks).
    Each query appends a unique question (≈ 1 chunk).
    """
    chunk_tok = params["chunk_size_tokens"]
    n = min(params["max_queries"], len(questions))

    # Shared document chunks
    doc_chunks = max(1, math.ceil(params["rag_doc_tokens"] / chunk_tok))
    shared_ids = list(range(doc_chunks))
    unique_counter = doc_chunks

    rows = []
    for i in range(n):
        unique_ids = [unique_counter]
        unique_counter += 1

        all_chunks = shared_ids + unique_ids
        total_tokens = params["rag_doc_tokens"] + params["question_tokens_avg"]

        rows.append({
            "query_id": i + 1,
            "query_text": f"Context: [RAG document ~{params['rag_doc_tokens']} tokens] {questions[i]}"[:200],
            "input_tokens": total_tokens,
            "chunk_ids_needed": json.dumps(all_chunks),
            "shared_chunk_ids": json.dumps(shared_ids),
            "unique_chunk_ids": json.dumps(unique_ids),
            "num_chunks": len(all_chunks),
        })

    return rows


def generate_nocontext_trace(params, questions):
    """
    Workload 3: No Context (Independent Queries).

    Each query is independent — no shared chunks.
    Each query uses exactly 1 chunk.
    """
    n = min(params["max_queries"], len(questions))

    rows = []
    for i in range(n):
        chunk_id = i
        rows.append({
            "query_id": i + 1,
            "query_text": questions[i][:200],
            "input_tokens": params["question_tokens_avg"],
            "chunk_ids_needed": json.dumps([chunk_id]),
            "shared_chunk_ids": json.dumps([]),
            "unique_chunk_ids": json.dumps([chunk_id]),
            "num_chunks": 1,
        })

    return rows


def generate_multiturn_trace(params, questions):
    """
    Workload 4: Multi-Turn Chat.

    Each turn accumulates all previous conversation history.
    Turn N needs chunks [0, 1, ..., N-1, N].
    Chunk 0..N-1 are shared (from past turns), chunk N is unique (new turn).
    """
    chunk_tok = params["chunk_size_tokens"]
    n = min(params["max_queries"], len(questions))

    # Base history is ~1 chunk
    base_tokens = 60  # "User: Hello AI ... Assistant: Hi ..."
    tpt = params["multiturn_tpt"]

    accumulated = []
    rows = []
    for i in range(n):
        new_chunk_id = i
        accumulated.append(new_chunk_id)
        shared_ids = accumulated[:-1]
        unique_ids = [new_chunk_id]

        total_tokens = base_tokens + (i + 1) * tpt

        # Build representative query text
        history = EXP4_BASE_HISTORY
        for j in range(min(i + 1, 3)):  # show first 3 turns for text preview
            history += f"User: {questions[j]}\n"
            history += f"Assistant: Here is answer for turn {j + 1}.\n"

        rows.append({
            "query_id": i + 1,
            "query_text": history[:200],
            "input_tokens": total_tokens,
            "chunk_ids_needed": json.dumps(list(accumulated)),
            "shared_chunk_ids": json.dumps(shared_ids),
            "unique_chunk_ids": json.dumps(unique_ids),
            "num_chunks": len(accumulated),
        })

    return rows


def write_trace_csv(rows, output_path):
    """Write trace rows to CSV."""
    if not rows:
        print(f"  ⚠️  No rows to write for {output_path.name}")
        return

    fieldnames = rows[0].keys()
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → {output_path.name}  ({len(rows)} queries)")


# ═══════════════════════════════════════════════════════════════
# TTFT LOOKUP + CONFIG UPDATES
# ═══════════════════════════════════════════════════════════════

def build_ttft_lookup(calibrated, params):
    """Build ttft_lookup.json from calibrated hardware measurements."""
    chunk_tok = params["chunk_size_tokens"]

    # Use calibrated latencies
    cold_ms = calibrated.get("cold_miss_ms", {}).get("mean", 30.0)
    l1_ms = calibrated.get("l1_hit_ms", {}).get("mean", 0.1)
    l2_ms = calibrated.get("l2_hit_ms", {}).get("mean", 0.25)
    l3_ms = calibrated.get("l3_hit_ms", {}).get("mean", 6.0)

    prefix_chunks = max(1, math.ceil(params["prefix_tokens"] / chunk_tok))
    rag_chunks = max(1, math.ceil(params["rag_doc_tokens"] / chunk_tok))

    lookup = {
        "shared_prefix": {
            "cold_ttft_ms": round(cold_ms * (prefix_chunks + 1), 2),
            "warm_ttft_ms": round(l1_ms * prefix_chunks + cold_ms * 1, 2),
            "avg_input_tokens": params["prefix_tokens"] + params["question_tokens_avg"],
            "shared_prefix_tokens": params["prefix_tokens"],
            "shared_prefix_chunks": prefix_chunks,
            "measurement_source": "hardware_calibration",
        },
        "rag": {
            "cold_ttft_ms": round(cold_ms * (rag_chunks + 1), 2),
            "warm_ttft_ms": round(l1_ms * rag_chunks + cold_ms * 1, 2),
            "avg_input_tokens": params["rag_doc_tokens"] + params["question_tokens_avg"],
            "shared_doc_tokens": params["rag_doc_tokens"],
            "shared_doc_chunks": rag_chunks,
            "measurement_source": "hardware_calibration",
        },
        "nocontext": {
            "cold_ttft_ms": round(cold_ms, 2),
            "warm_ttft_ms": round(cold_ms, 2),
            "avg_input_tokens": params["question_tokens_avg"],
            "measurement_source": "hardware_calibration",
        },
        "multiturn": {
            "cold_ttft_ms": round(cold_ms, 2),
            "warm_ttft_ms_base": round(l1_ms + cold_ms, 2),
            "base_tokens": 60,
            "tokens_per_turn": params["multiturn_tpt"],
            "measurement_source": "hardware_calibration",
        },
        "hardware_latencies": {
            "l1_hit_ms": round(l1_ms, 4),
            "l2_hit_ms": round(l2_ms, 4),
            "l3_hit_ms": round(l3_ms, 4),
            "cold_miss_ms": round(cold_ms, 4),
        },
    }

    return lookup


def update_configs_with_calibration(calibrated):
    """Update hardware_config.yaml and ppo_config.yaml with calibrated latencies."""
    l1_ms = calibrated.get("l1_hit_ms", {}).get("mean", 0.1)
    l2_ms = calibrated.get("l2_hit_ms", {}).get("mean", 0.25)
    l3_ms = calibrated.get("l3_hit_ms", {}).get("mean", 6.0)
    cold_ms = calibrated.get("cold_miss_ms", {}).get("mean", 30.0)
    prefetch_ms = calibrated.get("prefetch_l3_to_l2_ms", {}).get("mean", 6.0)

    # Update hardware_config.yaml
    if HW_CONFIG_PATH.exists():
        with open(HW_CONFIG_PATH) as f:
            hw_cfg = yaml.safe_load(f) or {}

        hw_cfg["l1_hit_latency_ms"] = round(l1_ms, 4)
        hw_cfg["l2_hit_latency_ms"] = round(l2_ms, 4)
        hw_cfg["l3_hit_latency_ms"] = round(l3_ms, 4)
        hw_cfg["cold_compute_per_chunk_ms"] = round(cold_ms, 4)
        hw_cfg["prefetch_l3_to_l2_ms"] = round(prefetch_ms, 4)

        with open(HW_CONFIG_PATH, "w") as f:
            yaml.dump(hw_cfg, f, default_flow_style=False, sort_keys=False)
        print(f"  [config] Updated → {HW_CONFIG_PATH.name}")

    # Update ppo_config.yaml
    if PPO_CONFIG_PATH.exists():
        with open(PPO_CONFIG_PATH) as f:
            ppo_cfg = yaml.safe_load(f) or {}

        ppo_cfg["l1_hit_latency_ms"] = round(l1_ms, 4)
        ppo_cfg["l2_hit_latency_ms"] = round(l2_ms, 4)
        ppo_cfg["l3_hit_latency_ms"] = round(l3_ms, 4)
        ppo_cfg["cold_compute_per_chunk_ms"] = round(cold_ms, 4)
        ppo_cfg["prefetch_l3_to_l2_ms"] = round(prefetch_ms, 4)

        with open(PPO_CONFIG_PATH, "w") as f:
            yaml.dump(ppo_cfg, f, default_flow_style=False, sort_keys=False)
        print(f"  [config] Updated → {PPO_CONFIG_PATH.name}")


# ═══════════════════════════════════════════════════════════════
# EMBEDDING GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_embeddings(trace_name, trace_path, embed_dim=384):
    """Generate sentence-transformer embeddings for query texts in a trace."""
    import pandas as pd

    df = pd.read_csv(trace_path)
    emb_path = DATA_DIR / f"embeddings_{trace_name}.npy"

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from agent.state_encoder import StateEncoder
        encoder = StateEncoder(embed_dim=embed_dim)
        texts = df["query_text"].tolist()
        embs = encoder.encode_and_save(texts, emb_path)
        print(f"  → {emb_path.name}  ({embs.shape})")
    except Exception as e:
        # Fallback: generate random embeddings (for testing without sentence-transformers)
        print(f"  ⚠️  sentence-transformers not available ({e}), using random embeddings")
        embs = np.random.randn(len(df), embed_dim).astype(np.float32)
        np.save(emb_path, embs)
        print(f"  → {emb_path.name}  ({embs.shape}) [random fallback]")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate traces for RL training using hardware cache calibration"
    )
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU-only mode (no GPU required)")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Only run hardware calibration, don't generate traces")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip hardware calibration, use config defaults")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding generation")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of calibration iterations per tier")
    args = parser.parse_args()

    t_start = time.time()
    print("\n" + "=" * 64)
    print("  TRACE GENERATION — Hardware Cache Backend")
    print("  (No vLLM, No LMCache — Pure hardware calibration)")
    print("=" * 64)

    # ── Load config params ──
    params = load_config_params()
    chunk_tok = params["chunk_size_tokens"]
    print(f"\n  Chunk size: {chunk_tok} tokens ({params['chunk_size_bytes']} bytes)")
    print(f"  Max queries per workload: {params['max_queries']}")

    # ── Hardware Calibration ──
    calibrated = None
    if not args.skip_calibration:
        print(f"\n{'=' * 64}")
        print("  PHASE 1: Hardware Calibration")
        print(f"{'=' * 64}")

        calibrated = calibrate_hardware(
            force_cpu=args.cpu_only,
            iterations=args.iterations,
        )

        # Save calibration report
        report_path = DATA_DIR / "calibration_report.csv"
        save_calibration_report(calibrated, report_path)

        # Update configs with measured values
        update_configs_with_calibration(calibrated)

        if args.calibrate_only:
            elapsed = time.time() - t_start
            print(f"\n{'=' * 64}")
            print(f"  CALIBRATION ONLY — Done in {elapsed:.1f}s")
            print(f"{'=' * 64}\n")
            return

    # ── Generate Traces ──
    print(f"\n{'=' * 64}")
    print("  PHASE 2: Generating Workload Traces")
    print(f"{'=' * 64}")

    traces = {}

    # Workload 1: Shared Prefix
    prefix_rows = generate_prefix_trace(params, EXP1_QUESTIONS)
    prefix_path = DATA_DIR / "traces_prefix.csv"
    write_trace_csv(prefix_rows, prefix_path)
    traces["prefix"] = prefix_path

    # Workload 2: RAG (Shared Document)
    rag_rows = generate_rag_trace(params, RAG_QUESTIONS)
    rag_path = DATA_DIR / "traces_rag.csv"
    write_trace_csv(rag_rows, rag_path)
    traces["rag"] = rag_path

    # Workload 3: No Context
    nc_rows = generate_nocontext_trace(params, NO_CONTEXT_QUESTIONS)
    nc_path = DATA_DIR / "traces_nocontext.csv"
    write_trace_csv(nc_rows, nc_path)
    traces["nocontext"] = nc_path

    # Workload 4: Multi-Turn Chat
    mt_rows = generate_multiturn_trace(params, MULTITURN_QUESTIONS)
    mt_path = DATA_DIR / "traces_multiturn.csv"
    write_trace_csv(mt_rows, mt_path)
    traces["multiturn"] = mt_path

    # ── TTFT Lookup ──
    if calibrated:
        ttft_lookup = build_ttft_lookup(calibrated, params)
    else:
        # Use config defaults if calibration was skipped
        ttft_lookup = build_ttft_lookup({
            "l1_hit_ms": {"mean": 0.1},
            "l2_hit_ms": {"mean": 0.25},
            "l3_hit_ms": {"mean": 6.0},
            "cold_miss_ms": {"mean": 30.0},
        }, params)

    ttft_path = DATA_DIR / "ttft_lookup.json"
    with open(ttft_path, "w") as f:
        json.dump(ttft_lookup, f, indent=2)
    print(f"  → {ttft_path.name}")

    # ── Generate Embeddings ──
    if not args.skip_embeddings:
        print(f"\n{'=' * 64}")
        print("  PHASE 3: Generating Query Embeddings")
        print(f"{'=' * 64}")

        # Remove stale embeddings first
        for emb_file in DATA_DIR.glob("embeddings_*.npy"):
            emb_file.unlink()
            print(f"  [clean] Removed stale {emb_file.name}")

        for name, path in traces.items():
            generate_embeddings(name, path)

    # ── Summary ──
    elapsed = time.time() - t_start
    print(f"\n{'=' * 64}")
    print(f"  ✅ DONE! Total time: {elapsed:.1f}s")
    print(f"{'=' * 64}")
    print(f"  Generated files:")
    for name, path in traces.items():
        print(f"    {path.name}")
    print(f"    ttft_lookup.json")
    if calibrated:
        print(f"    calibration_report.csv")
    print(f"    embeddings_*.npy (4 files)")
    print()


if __name__ == "__main__":
    main()
