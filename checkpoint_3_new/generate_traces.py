#!/usr/bin/env python3
"""
Trace Generation for RL Cache Prefetching Training
=====================================================

Runs 4 workloads through vLLM (plain, NO LMCache) and produces:
  - 4 trace CSVs (one per workload) with clean query text
  - 4 embedding .npy files (384-dim MiniLM-L6-v2 vectors)
  - config.yaml updated with measured latency values

Requirements:
  - vLLM (pip install vllm)
  - PyTorch with CUDA
  - sentence-transformers
  - An NVIDIA GPU (RTX 3050 or better)

Usage:
    python generate_traces.py                    # Full generation
    python generate_traces.py --skip-llm         # Skip LLM, just generate CSVs from query data
    python generate_traces.py --max-queries 10   # Quick test with fewer queries

WHAT THIS SCRIPT DOES (step by step):
  1. Loads the Qwen2.5-0.5B model with plain vLLM (no LMCache!)
  2. Runs 4 workloads (prefix, RAG, no-context, multi-turn) through the model
  3. Measures TTFT (Time To First Token) for each query
  4. Deterministically assigns chunk IDs based on token counts + workload structure
  5. Generates sentence embeddings for each query
  6. Saves everything to data/ directory

ERRORS FIXED FROM PREVIOUS VERSION:
  - NO LMCache dependency (was causing import/compatibility errors)
  - NO raw PDF binary data as query text (was producing garbage embeddings)
  - Clean, readable text for RAG context (hardcoded ESG report summary)
  - Proper error handling for vLLM engine creation
  - Chunk ID assignment is pure math, not dependent on cache observation
  - Embeddings are generated from clean query text, not PDF binary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
CONFIG_PATH = SCRIPT_DIR / "config.yaml"

# ═══════════════════════════════════════════════════════════════
# LOAD CONFIG
# ═══════════════════════════════════════════════════════════════

def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


# ═══════════════════════════════════════════════════════════════
# QUERY DATA
# ═══════════════════════════════════════════════════════════════

# Shared system prompt for Experiment 1 (Shared Prefix)
# ~650 tokens → ~3 chunks of 256 tokens each
SYSTEM_PROMPT = (
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
    "90% limb symmetry index on isokinetic testing, successful completion of hop tests, "
    "and psychological readiness assessed via the ACL-RSI scale. "
    "Phase IV (Maintenance, Ongoing): Long-term injury prevention programming. "
    "Periodised strength and conditioning with progressive overload. Neuromuscular "
    "control drills integrated into warm-up routines following the FIFA 11+ protocol. "
    "Annual functional reassessment recommended. "
    "Now answer the following clinical question. "
)

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
    "What are early mobilisation protocols after total knee replacement?",
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

# RAG context — clean, readable text (NOT raw PDF binary like the old version)
RAG_CONTEXT = (
    "Tongaat Hulett and Implats ESG Report Summary\n\n"
    "Tongaat Hulett is a leading agri-processing business focusing on the complementary "
    "activities of sugar production, property development, and starch production. The "
    "company operates in South Africa, Mozambique, Zimbabwe, and Botswana, employing "
    "over 30,000 people at the peak of the sugar milling season. In the 2021 financial "
    "year, Tongaat Hulett produced approximately 1.1 million tons of sugar across its "
    "six sugar mills with a combined crushing capacity exceeding 8 million tons of "
    "sugarcane per season.\n\n"
    "The company has committed to reducing energy intensity by 20% by 2025, with specific "
    "targets for water efficiency improvement. Tongaat Hulett invests in socio-economic "
    "development (SED) and reported total SED expenditure in 2021 aligned with community "
    "needs. The Lost Time Injury Frequency Rate (LTIFR) is a critical safety metric "
    "tracked annually across all operations.\n\n"
    "The company's ESG framework aligns with the UN Sustainable Development Goals and "
    "operates under ISO 45001 certification. The manufactured capital includes six sugar "
    "mills across four countries. Employee training and development spend reached R45 million, "
    "reflecting commitment to human capital investment. Revenue was approximately R16.2 billion "
    "in 2021, with significant capital expenditure directed towards operational efficiency.\n\n"
    "Implats (Impala Platinum Holdings Limited) is one of the world's foremost producers "
    "of platinum group metals (PGMs). The company's operations span South Africa and "
    "Zimbabwe, with managed operations including Impala Rustenburg, Marula, and Zimplats. "
    "Implats' ESG framework is built on three pillars focusing on environmental stewardship, "
    "social responsibility, and governance excellence.\n\n"
    "In 2023, Implats achieved significant safety milestones while investing heavily in "
    "socio-economic development and community projects. The company targets a 30% reduction "
    "in carbon emissions by 2030 and has invested in renewable energy projects including "
    "the 35MW solar PV project at Zimplats. Water recycling rates exceeded targets, and "
    "the company maintains strict environmental compliance across all operations.\n\n"
    "Implats distributed over R50 billion in total value to stakeholders in 2023, including "
    "R28 billion in wages and benefits, R12 billion in taxes and royalties, and R2.3 billion "
    "in dividends. Total mineral reserves stand at approximately 190 million ounces of PGMs. "
    "Production across all operations exceeded 3.2 million ounces of refined PGMs.\n\n"
    "Both companies utilise the six capitals framework (Financial, Manufactured, Intellectual, "
    "Human, Social/Relationship, Natural) to illustrate value creation. The double materiality "
    "principle assesses both inward financial materiality and outward impact materiality for "
    "comprehensive sustainability reporting aligned with global standards.\n\n"
    "Environmental management across both organisations addresses water stewardship, carbon "
    "emissions reduction, waste minimisation, and biodiversity conservation. Tongaat Hulett's "
    "sugarcane operations face increasing climate risks from drought and flooding events, "
    "while Implats' mining operations manage dust emissions, acid mine drainage, and tailings "
    "storage facility safety under stringent regulatory requirements."
)

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

MULTITURN_BASE_HISTORY = "User: Hello AI.\nAssistant: Hi there! How can I help you today?\n"


# ═══════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════

def build_prefix_prompts(questions: List[str]) -> List[str]:
    """All queries share the same system prompt prefix."""
    return [SYSTEM_PROMPT + q for q in questions]


def build_rag_prompts(questions: List[str]) -> List[str]:
    """All queries share the same document context."""
    return [f"Context: {RAG_CONTEXT}\n\nQuestion: {q}" for q in questions]


def build_nocontext_prompts(questions: List[str]) -> List[str]:
    """Each query is independent, no shared context."""
    return list(questions)


def build_multiturn_prompts(questions: List[str]) -> List[str]:
    """Each query includes all prior conversation history (growing context)."""
    prompts = []
    history = MULTITURN_BASE_HISTORY
    for i, question in enumerate(questions, 1):
        history += f"User: {question}\n"
        history += f"Assistant: Here is my answer for turn {i}.\n"
        prompts.append(history + f"User: Can you elaborate further on: {question}")
    return prompts


# ═══════════════════════════════════════════════════════════════
# vLLM ENGINE (NO LMCache!)
# ═══════════════════════════════════════════════════════════════

def build_engine(cfg: dict):
    """
    Build a plain vLLM engine. NO LMCache connector.

    This function requires vLLM to be installed and a CUDA GPU to be available.
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vLLM is not installed. Install it with: pip install vllm")
        print("       This script needs vLLM to run the LLM for trace generation.")
        sys.exit(1)

    model_name = cfg.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    max_model_len = cfg.get("max_model_len", 4096)
    gpu_mem_util = cfg.get("gpu_memory_utilization", 0.80)
    max_new_tokens = cfg.get("max_new_tokens", 20)

    print(f"\n>>> Loading model: {model_name}")
    print(f"    max_model_len={max_model_len}  gpu_mem={gpu_mem_util}")
    print(f"    NOTE: Using plain vLLM (NO LMCache)")

    llm = LLM(
        model=model_name,
        enforce_eager=True,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        disable_log_stats=True,
    )
    print("    Engine loaded ✓")

    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    # Extract KV cache geometry from the model
    kv_info = _get_kv_config(llm, cfg)

    return llm, sp, kv_info


def _get_kv_config(llm, cfg: dict) -> dict:
    """Extract KV cache geometry from the loaded model."""
    chunk_size = cfg.get("chunk_size_tokens", 256)

    try:
        mc = llm.llm_engine.model_config.hf_config
        num_layers = getattr(mc, "num_hidden_layers", 24)
        num_kv_heads = getattr(mc, "num_key_value_heads",
                       getattr(mc, "num_attention_heads", 16))
        hidden_size = getattr(mc, "hidden_size", 896)
        head_dim = hidden_size // getattr(mc, "num_attention_heads", num_kv_heads)
    except Exception:
        num_layers, num_kv_heads, head_dim = 24, 2, 64

    dtype_bytes = 2  # fp16
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes

    info = {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "kv_bytes_per_token": kv_bytes_per_token,
        "chunk_size_tokens": chunk_size,
        "chunk_size_bytes": chunk_size * kv_bytes_per_token,
    }
    print(f"    [kv] {num_layers}L × {num_kv_heads}KVH × {head_dim}d  "
          f"KV/tok={kv_bytes_per_token}B  "
          f"chunk={chunk_size}tok ({chunk_size * kv_bytes_per_token / 1e6:.1f} MB)")
    return info


# ═══════════════════════════════════════════════════════════════
# RUN QUERIES THROUGH LLM
# ═══════════════════════════════════════════════════════════════

def run_queries(
    llm, sp, prompts: List[str], workload_name: str, cfg: dict,
) -> List[dict]:
    """
    Run a list of prompts through the vLLM engine.
    Returns per-query results with timing and token counts.
    """
    max_model_len = cfg.get("max_model_len", 4096)
    max_new_tokens = cfg.get("max_new_tokens", 20)
    max_input = max_model_len - max_new_tokens - 10  # Safety margin

    tokenizer = llm.get_tokenizer()

    print(f"\n{'=' * 60}")
    print(f"  WORKLOAD: {workload_name} ({len(prompts)} queries)")
    print(f"{'=' * 60}")

    results = []
    for i, prompt in enumerate(prompts):
        # Truncate if too long
        token_ids = tokenizer.encode(prompt)
        if len(token_ids) > max_input:
            token_ids = token_ids[:max_input]
            prompt = tokenizer.decode(token_ids, skip_special_tokens=True)

        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sp)
        t1 = time.perf_counter()

        out = outputs[0]
        ptok = len(out.prompt_token_ids)
        gtok = len(out.outputs[0].token_ids)
        latency = t1 - t0

        # Extract TTFT from vLLM metrics
        ttft_s = None
        m = getattr(out, "metrics", None)
        if m is not None:
            arrival = getattr(m, "arrival_time", None)
            first_tk = getattr(m, "first_token_time", None)
            if arrival is not None and first_tk is not None:
                ttft_s = first_tk - arrival

        # Fallback TTFT estimation
        if ttft_s is None and ptok > 0:
            ttft_s = max(latency - gtok / 200.0, latency * 0.1)

        ttft_ms = ttft_s * 1000 if ttft_s else 0.0

        results.append({
            "input_tokens": ptok,
            "output_tokens": gtok,
            "ttft_ms": round(ttft_ms, 2),
            "latency_s": round(latency, 4),
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Q{i+1:>3d}  in={ptok:>5d}tok  TTFT={ttft_ms:>7.1f}ms  "
                  f"lat={latency:.3f}s")

    return results


# ═══════════════════════════════════════════════════════════════
# TRACE CSV GENERATION (chunk ID assignment)
# ═══════════════════════════════════════════════════════════════

def generate_trace_csv(
    workload_type: str,
    query_texts: List[str],
    results: List[dict],
    output_path: Path,
    chunk_size_tokens: int,
) -> None:
    """
    Generate the RL-ready trace CSV with deterministic chunk ID assignment.

    Chunk IDs are assigned based on token counts and workload structure:
    - prefix:    shared prefix chunks + unique suffix chunks per query
    - rag:       shared document chunks + one unique chunk per query
    - nocontext: one independent chunk per query
    - multiturn: accumulated chunks, each turn adds one new chunk

    The query_text in the CSV is the CLEAN question text (not the full prompt
    with context), making it suitable for embedding generation.
    """
    rows = []

    if workload_type == "prefix":
        # Shared prefix is ~650 tokens ≈ 3 chunks
        prefix_chunks = max(1, math.ceil(650 / chunk_size_tokens))
        shared_ids = list(range(prefix_chunks))
        unique_counter = prefix_chunks

        for i, (text, result) in enumerate(zip(query_texts, results)):
            total_chunks = max(1, math.ceil(result["input_tokens"] / chunk_size_tokens))
            num_unique = max(0, total_chunks - prefix_chunks)
            unique_ids = list(range(unique_counter, unique_counter + num_unique))
            unique_counter += num_unique
            all_chunk_ids = shared_ids + unique_ids

            rows.append({
                "query_id": i + 1,
                "query_text": text,  # Clean question text
                "input_tokens": result["input_tokens"],
                "chunk_ids_needed": json.dumps(all_chunk_ids),
                "shared_chunk_ids": json.dumps(shared_ids),
                "unique_chunk_ids": json.dumps(unique_ids),
                "num_chunks": len(all_chunk_ids),
            })

    elif workload_type == "rag":
        # Shared document context ≈ 1700 tokens ≈ 7 chunks
        shared_doc_chunks = max(1, math.ceil(1700 / chunk_size_tokens))
        shared_ids = list(range(shared_doc_chunks))
        unique_counter = shared_doc_chunks

        for i, (text, result) in enumerate(zip(query_texts, results)):
            unique_ids = [unique_counter]
            unique_counter += 1
            all_chunk_ids = shared_ids + unique_ids

            rows.append({
                "query_id": i + 1,
                "query_text": text,  # Clean question text
                "input_tokens": result["input_tokens"],
                "chunk_ids_needed": json.dumps(all_chunk_ids),
                "shared_chunk_ids": json.dumps(shared_ids),
                "unique_chunk_ids": json.dumps(unique_ids),
                "num_chunks": len(all_chunk_ids),
            })

    elif workload_type == "nocontext":
        # Each query is completely independent — 1 unique chunk each
        for i, (text, result) in enumerate(zip(query_texts, results)):
            chunk_id = i
            rows.append({
                "query_id": i + 1,
                "query_text": text,
                "input_tokens": result["input_tokens"],
                "chunk_ids_needed": json.dumps([chunk_id]),
                "shared_chunk_ids": json.dumps([]),
                "unique_chunk_ids": json.dumps([chunk_id]),
                "num_chunks": 1,
            })

    elif workload_type == "multiturn":
        # Each turn accumulates all previous chunks + one new one
        accumulated = []
        for i, (text, result) in enumerate(zip(query_texts, results)):
            new_chunk_id = i
            accumulated.append(new_chunk_id)
            shared_ids = accumulated[:-1]  # All prior chunks
            unique_ids = [new_chunk_id]    # This turn's new chunk

            rows.append({
                "query_id": i + 1,
                "query_text": text,
                "input_tokens": result["input_tokens"],
                "chunk_ids_needed": json.dumps(list(accumulated)),
                "shared_chunk_ids": json.dumps(shared_ids),
                "unique_chunk_ids": json.dumps(unique_ids),
                "num_chunks": len(accumulated),
            })

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"  → {output_path.name} ({len(rows)} queries)")


def generate_trace_csv_no_llm(
    workload_type: str,
    query_texts: List[str],
    output_path: Path,
    chunk_size_tokens: int,
) -> None:
    """
    Generate trace CSV WITHOUT running the LLM.

    Uses estimated token counts based on character count (4 chars ≈ 1 token).
    This is for when you don't have vLLM/GPU but need trace CSVs for testing.
    """
    fake_results = []
    for text in query_texts:
        # Build the full prompt to estimate its token count
        if workload_type == "prefix":
            full = SYSTEM_PROMPT + text
        elif workload_type == "rag":
            full = f"Context: {RAG_CONTEXT}\n\nQuestion: {text}"
        elif workload_type == "multiturn":
            # Multi-turn grows, but we just estimate the question part
            full = text  # Will be handled by the loop
        else:
            full = text

        est_tokens = max(10, len(full) // 4)
        fake_results.append({"input_tokens": est_tokens, "output_tokens": 20,
                            "ttft_ms": 0.0, "latency_s": 0.0})

    # For multiturn, we need to estimate growing token counts
    if workload_type == "multiturn":
        prompts = build_multiturn_prompts(query_texts)
        for i, prompt in enumerate(prompts):
            fake_results[i]["input_tokens"] = max(10, len(prompt) // 4)

    generate_trace_csv(workload_type, query_texts, fake_results,
                       output_path, chunk_size_tokens)


# ═══════════════════════════════════════════════════════════════
# EMBEDDING GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_embeddings(
    trace_path: Path, output_path: Path, embed_dim: int = 384,
) -> np.ndarray:
    """
    Generate MiniLM-L6-v2 embeddings for all query texts in a trace CSV.
    Saves to a .npy file for reuse.
    """
    import pandas as pd

    df = pd.read_csv(trace_path)
    texts = df["query_text"].tolist()

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = embeddings.astype(np.float32)
    except ImportError:
        print("  [WARN] sentence-transformers not installed. Using zero embeddings.")
        embeddings = np.zeros((len(texts), embed_dim), dtype=np.float32)

    np.save(str(output_path), embeddings)
    print(f"  → {output_path.name} shape={embeddings.shape}")
    return embeddings


# ═══════════════════════════════════════════════════════════════
# UPDATE CONFIG WITH MEASURED VALUES
# ═══════════════════════════════════════════════════════════════

def update_config_with_measurements(
    all_results: dict, kv_info: dict, cfg_path: Path,
) -> None:
    """
    Update config.yaml with KV geometry from the actual model
    and measured latencies from the experiments.
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Update KV geometry
    cfg["kv_bytes_per_token"] = kv_info["kv_bytes_per_token"]
    cfg["chunk_size_bytes"] = kv_info["chunk_size_bytes"]

    # Compute cold compute latency from RAG cold TTFT
    rag_results = all_results.get("rag", [])
    if rag_results:
        cold_ttft = rag_results[0]["ttft_ms"]  # First query is always cold
        num_chunks = max(1, math.ceil(rag_results[0]["input_tokens"]
                                      / kv_info["chunk_size_tokens"]))
        cold_per_chunk = round(cold_ttft / num_chunks, 1)
        cfg["cold_compute_per_chunk_ms"] = cold_per_chunk
        print(f"  [config] cold_compute_per_chunk_ms = {cold_per_chunk} "
              f"(from RAG cold TTFT={cold_ttft:.1f}ms / {num_chunks} chunks)")

    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  [config] Updated → {cfg_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate trace datasets for RL cache prefetching training"
    )
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM inference, generate traces from query data only")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Override max queries per workload (default: from config)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding generation")
    args = parser.parse_args()

    t_start = time.time()
    print("\n" + "=" * 60)
    print("  TRACE GENERATION — vLLM (NO LMCache)")
    print("=" * 60)

    cfg = load_config()
    chunk_size_tokens = cfg.get("chunk_size_tokens", 256)
    max_queries = args.max_queries or cfg.get("max_queries", 50)
    embed_dim = cfg.get("embed_dim", 384)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Truncate question lists to max_queries
    prefix_qs = PREFIX_QUESTIONS[:max_queries]
    rag_qs = RAG_QUESTIONS[:max_queries]
    nocontext_qs = NO_CONTEXT_QUESTIONS[:max_queries]
    multiturn_qs = MULTITURN_QUESTIONS[:max_queries]

    # ── Generate Traces ──────────────────────────────────────────

    if args.skip_llm:
        print("\n  [skip-llm] Generating traces without LLM inference...")
        print("  Token counts will be estimated (4 chars ≈ 1 token)")

        generate_trace_csv_no_llm(
            "prefix", prefix_qs,
            DATA_DIR / "traces_prefix.csv", chunk_size_tokens)
        generate_trace_csv_no_llm(
            "rag", rag_qs,
            DATA_DIR / "traces_rag.csv", chunk_size_tokens)
        generate_trace_csv_no_llm(
            "nocontext", nocontext_qs,
            DATA_DIR / "traces_nocontext.csv", chunk_size_tokens)
        generate_trace_csv_no_llm(
            "multiturn", multiturn_qs,
            DATA_DIR / "traces_multiturn.csv", chunk_size_tokens)

    else:
        # Full LLM-based trace generation
        llm, sp, kv_info = build_engine(cfg)
        all_results = {}

        # Build prompts
        prefix_prompts = build_prefix_prompts(prefix_qs)
        rag_prompts = build_rag_prompts(rag_qs)
        nocontext_prompts = build_nocontext_prompts(nocontext_qs)
        multiturn_prompts = build_multiturn_prompts(multiturn_qs)

        # Run through LLM
        results_prefix = run_queries(llm, sp, prefix_prompts, "Shared Prefix", cfg)
        results_rag = run_queries(llm, sp, rag_prompts, "RAG (Shared Doc)", cfg)
        results_nc = run_queries(llm, sp, nocontext_prompts, "No Context", cfg)
        results_mt = run_queries(llm, sp, multiturn_prompts, "Multi-Turn Chat", cfg)

        all_results = {
            "prefix": results_prefix,
            "rag": results_rag,
            "nocontext": results_nc,
            "multiturn": results_mt,
        }

        # Generate trace CSVs
        # NOTE: query_texts are the CLEAN question texts, not full prompts
        generate_trace_csv("prefix", prefix_qs, results_prefix,
                          DATA_DIR / "traces_prefix.csv", chunk_size_tokens)
        generate_trace_csv("rag", rag_qs, results_rag,
                          DATA_DIR / "traces_rag.csv", chunk_size_tokens)
        generate_trace_csv("nocontext", nocontext_qs, results_nc,
                          DATA_DIR / "traces_nocontext.csv", chunk_size_tokens)
        generate_trace_csv("multiturn", multiturn_qs, results_mt,
                          DATA_DIR / "traces_multiturn.csv", chunk_size_tokens)

        # Update config with measured values
        update_config_with_measurements(all_results, kv_info, CONFIG_PATH)

        # Save raw experiment results (for reference)
        import pandas as pd
        raw_rows = []
        for wl_name, wl_results in all_results.items():
            for i, r in enumerate(wl_results):
                r["workload"] = wl_name
                r["query_id"] = i + 1
                raw_rows.append(r)
        pd.DataFrame(raw_rows).to_csv(
            DATA_DIR / "raw_experiment_results.csv", index=False)
        print(f"  → raw_experiment_results.csv")

    # ── Generate Embeddings ──────────────────────────────────────

    if not args.skip_embeddings:
        print(f"\n{'=' * 60}")
        print("  GENERATING EMBEDDINGS (MiniLM-L6-v2)")
        print(f"{'=' * 60}")

        for name in ["prefix", "rag", "nocontext", "multiturn"]:
            trace_path = DATA_DIR / f"traces_{name}.csv"
            emb_path = DATA_DIR / f"embeddings_{name}.npy"
            if trace_path.exists():
                generate_embeddings(trace_path, emb_path, embed_dim)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  DONE! Total time: {elapsed / 60:.1f} minutes")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
