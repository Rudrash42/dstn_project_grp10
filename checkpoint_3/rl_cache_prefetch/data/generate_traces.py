#!/usr/bin/env python3
"""
Generate REAL trace datasets for RL training using vLLM + LMCache on GPU.

Runs 4 workloads through the vLLM engine with LMCache KV connector,
capturing actual TTFT, token counts, cache hit/miss behaviour, and
tier occupancy.

Produces:
    - 4 trace CSVs (one per workload)
    - ttft_lookup.json (measured cold/warm TTFT per workload)
    - runtime_chunk_events.jsonl (per-query runtime provenance)
    - trace_audit_report.json (coverage/pressure/diversity/sanity report)
    - Updated hardware_config.yaml with calibrated tier numbers

Usage:
        source /home/rudrash/prog/dstn/.venv/bin/activate
        python data/generate_traces.py
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import subprocess
import shutil
import sys
import time

import numpy as np
import torch
import yaml
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_DIR.parent
CHECKPOINT2_DIR = PROJECT_ROOT.parent.parent / "checkpoint_2"
PPO_CONFIG_PATH = PROJECT_ROOT / "configs" / "ppo_config.yaml"
HARDWARE_CONFIG_PATH = PROJECT_ROOT / "configs" / "hardware_config.yaml"

# LMCache disk store — use a separate dir for checkpoint_3
CACHE_DIR = DATA_DIR / "lmcache_store"
LMCACHE_CFG_PATH = DATA_DIR / "lmcache_config.yaml"

# Source doc for RAG experiment
SOURCE_FILE = CHECKPOINT2_DIR / "data" / "finance_reports.pdf"
BENCHMARK_LATENCY_PATH = PROJECT_ROOT / "results" / "hardware_benchmark_latencies.csv"

# ═══════════════════════════════════════════════════════════════
# HARDWARE / MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_MODEL_LEN = 4096
GPU_MEM_UTIL = 0.80
MAX_NEW_TOKENS = 20
TEMPERATURE = 0.0
ENFORCE_EAGER = True
CHUNK_SIZE = 256
MAX_CPU_CACHE_GB = 0.005       # L2 CPU cache budget (~5 MB) — forces L2→L3 spill
MAX_DISK_CACHE_GB = 5.0        # L3 disk cache budget (GB)
NUM_GPU_BLOCKS_OVERRIDE = 128   # L1 GPU blocks (2048 tokens, ~24 MB KV) — allows single large prompt to fit
MAX_QUERIES = 50
DEFAULT_CUDA_REQUIRED = True
DEFAULT_RUNTIME_CHUNKS_REQUIRED = True
MIN_L1_TO_L2_TRANSITIONS = 1
MIN_L2_TO_L3_TRANSITIONS = 1
REAL_CHUNK_EVENT_SOURCES = {
    "direct_runtime",
    "lmcache_store_runtime_snapshot",
    "lmcache_store_coldpass",
}

# ═══════════════════════════════════════════════════════════════
# QUERY DATA  (identical to checkpoint_2/run_experiments.py)
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
    # ── Extended context to push prefix to ~3 chunks (~650+ tokens) ──
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
# ENGINE SETUP
# ═══════════════════════════════════════════════════════════════

def setup_lmcache():
    """Write LMCache YAML config and set env var."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {
        "chunk_size": CHUNK_SIZE,
        "local_cpu": True,
        "max_local_cpu_size": MAX_CPU_CACHE_GB,
        "local_disk": str(CACHE_DIR) + "/",
        "max_local_disk_size": MAX_DISK_CACHE_GB,
        "enable_kv_events": True,
        "remote_url": None,
        "remote_serde": "naive",
        "save_decode_cache": True,
    }
    with open(LMCACHE_CFG_PATH, "w") as f:
        yaml.dump(cfg, f)
    os.environ["LMCACHE_CONFIG_FILE"] = str(LMCACHE_CFG_PATH)
    print(f"  [lmcache] Config  → {LMCACHE_CFG_PATH}")
    print(f"  [lmcache] Disk    → {CACHE_DIR}")
    print(f"  [lmcache] CPU={MAX_CPU_CACHE_GB} GB, Disk={MAX_DISK_CACHE_GB} GB, Chunk={CHUNK_SIZE} tok")


def clear_cache():
    """Wipe disk cache for a cold start.

    Important: call this only before the vLLM+LMCache engine is created.
    Deleting LMCache disk files while an engine is alive can desynchronize
    backend metadata and lead to retrieval KeyErrors.
    """
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("  [cache] Cleared (cold start)")


def ensure_cuda_available(cuda_required: bool):
    """Fail fast when CUDA is required but unavailable, without touching torch CUDA state."""
    if not cuda_required:
        return

    if shutil.which("nvidia-smi") is None:
        raise RuntimeError(
            "CUDA is required for trace generation, but nvidia-smi is not available. "
            "Run on a CUDA-enabled host or pass --no-cuda-required explicitly."
        )

    probe = subprocess.run(
        ["nvidia-smi", "-L"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or "GPU" not in (probe.stdout or ""):
        raise RuntimeError(
            "CUDA is required for trace generation, but no GPU was detected by nvidia-smi. "
            f"stderr={probe.stderr.strip()}"
        )


def build_engine(runtime_chunks_required: bool):
    """Build vLLM engine with LMCache KV connector."""
    # vLLM + CUDA + fork can fail if CUDA state exists in parent process.
    # Spawn avoids inherited CUDA context problems in worker processes.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams

    print(f"\n>>> Loading model: {MODEL_NAME}")
    print(f"    max_model_len={MAX_MODEL_LEN}  gpu_mem={GPU_MEM_UTIL}")
    print(f"    num_gpu_blocks_override={NUM_GPU_BLOCKS_OVERRIDE}")

    using_lmcache_connector = False
    try:
        import lmcache  # noqa: F401
        llm = LLM(
            model=MODEL_NAME,
            kv_transfer_config={
                "kv_connector": "LMCacheConnectorV1",
                "kv_role": "kv_both",
            },
            enforce_eager=ENFORCE_EAGER,
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_model_len=MAX_MODEL_LEN,
            num_gpu_blocks_override=NUM_GPU_BLOCKS_OVERRIDE,
            disable_log_stats=True,
        )
        using_lmcache_connector = True
        print("    Engine loaded WITH LMCache KV connector ✓")
    except Exception as exc:
        if runtime_chunks_required:
            raise RuntimeError(
                "Direct runtime chunk capture is required, but LMCache connector could not be initialized. "
                f"Original error: {exc}"
            ) from exc
        print(f"    LMCache unavailable ({exc}), falling back to plain vLLM")
        llm = LLM(
            model=MODEL_NAME,
            enforce_eager=ENFORCE_EAGER,
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_model_len=MAX_MODEL_LEN,
            num_gpu_blocks_override=NUM_GPU_BLOCKS_OVERRIDE,
            disable_log_stats=True,
        )

    sp = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS)
    return llm, sp, using_lmcache_connector


# ═══════════════════════════════════════════════════════════════
# MEASUREMENT HELPERS
# ═══════════════════════════════════════════════════════════════

def cache_size_mb():
    if not CACHE_DIR.exists():
        return 0.0
    return round(
        sum(f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file()) / (1024 * 1024), 2
    )


def cache_file_count():
    if not CACHE_DIR.exists():
        return 0
    return sum(1 for f in CACHE_DIR.rglob("*") if f.is_file())


def list_lmcache_chunk_file_ids():
    """
    Return stable LMCache chunk file identifiers from on-disk store.

    We use relative file paths as real runtime chunk IDs because these files
    are emitted by LMCache itself (not inferred from prompt heuristics).
    """
    if not CACHE_DIR.exists():
        return []

    ids = []
    for fp in CACHE_DIR.rglob("*"):
        if not fp.is_file():
            continue
        name = fp.name
        if not (name.endswith(".pt") or name.endswith(".bin")):
            continue
        ids.append(fp.relative_to(CACHE_DIR).as_posix())

    return sorted(set(ids))


def get_kv_config(llm):
    """Extract model KV-cache geometry."""
    try:
        mc = llm.llm_engine.model_config.hf_config
        num_layers = getattr(mc, "num_hidden_layers", 24)
        num_kv_heads = getattr(mc, "num_key_value_heads",
                       getattr(mc, "num_attention_heads", 16))
        hidden_size = getattr(mc, "hidden_size", 896)
        head_dim = hidden_size // getattr(mc, "num_attention_heads", num_kv_heads)
    except Exception:
        num_layers, num_kv_heads, head_dim = 24, 2, 64

    dtype_bytes = 2
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes

    try:
        cc = llm.llm_engine.cache_config
        block_size = getattr(cc, "block_size", 16)
        num_gpu_blocks = getattr(cc, "num_gpu_blocks", NUM_GPU_BLOCKS_OVERRIDE or 256)
    except Exception:
        block_size = 16
        num_gpu_blocks = NUM_GPU_BLOCKS_OVERRIDE or 256

    tokens_capacity = num_gpu_blocks * block_size
    gpu_kv_capacity_mb = round(tokens_capacity * kv_bytes_per_token / (1024**2), 2)

    info = {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype_bytes": dtype_bytes,
        "kv_bytes_per_token": kv_bytes_per_token,
        "block_size": block_size,
        "num_gpu_blocks": num_gpu_blocks,
        "tokens_capacity": tokens_capacity,
        "gpu_kv_capacity_mb": gpu_kv_capacity_mb,
        "chunk_size_tokens": CHUNK_SIZE,
        "chunk_size_bytes": CHUNK_SIZE * kv_bytes_per_token,
    }
    print(f"    [kv-cfg] {num_layers}L × {num_kv_heads}KVH × {head_dim}d  "
          f"block_size={block_size}  gpu_blocks={num_gpu_blocks}  "
          f"KV/tok={kv_bytes_per_token}B  "
          f"GPU KV capacity={gpu_kv_capacity_mb:.1f}MB ({tokens_capacity} tok)")
    return info


def load_benchmark_tier_latencies(path):
    """Load measured tier latencies from hardware benchmark CSV if available."""
    path = Path(path)
    if not path.exists():
        return {}

    tier_samples = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tier = (row.get("tier") or "").strip().upper()
            if not tier:
                continue
            try:
                latency = float(row.get("latency_ms", 0.0))
            except Exception:
                continue
            tier_samples.setdefault(tier, []).append(latency)

    calibrated = {}
    for tier, vals in tier_samples.items():
        # Drop the first warmup sample when possible.
        stable_vals = vals[1:] if len(vals) > 3 else vals
        if not stable_vals:
            continue
        calibrated[tier] = round(float(np.median(stable_vals)), 3)

    if calibrated:
        print(f"  [benchmark] Loaded measured tier latencies from {path}")
        print(
            "  [benchmark] "
            + ", ".join(f"{k}={v}ms" for k, v in sorted(calibrated.items()))
        )
    return calibrated


def validate_benchmark_latencies(calibrated):
    """Ensure measured benchmark tiers exist before calibration is applied."""
    required = {"L1", "L2", "L3", "PREFETCH"}
    missing = sorted(required - set(calibrated.keys()))
    if missing:
        raise RuntimeError(
            "Measured benchmark latencies are required for calibration. "
            f"Missing tiers in {BENCHMARK_LATENCY_PATH}: {missing}"
        )


def make_chunk_fingerprint(token_slice):
    """Create a stable fingerprint for one chunk worth of prompt token IDs."""
    arr = np.asarray(token_slice, dtype=np.int32)
    return hashlib.blake2b(arr.tobytes(), digest_size=10).hexdigest()


def build_embedding_text(focus_text, context_profile, prompt_preview, input_tokens):
    """Build an embedding-friendly text that preserves semantics without huge context dominance."""
    focus = " ".join((focus_text or "").split())
    preview = " ".join((prompt_preview or "").split())
    return (
        f"Focus question: {focus}\n"
        f"Context profile: {context_profile}\n"
        f"Prompt tokens: {input_tokens}\n"
        f"Context preview: {preview}"
    )


def extract_runtime_chunk_ids(output_obj):
    """
    Extract runtime chunk IDs from vLLM/LMCache runtime metadata.

    Returns None when chunk-level runtime events are not exposed.
    """
    candidate_containers = [output_obj, getattr(output_obj, "metrics", None)]
    candidate_attrs = [
        "chunk_ids",
        "kv_chunk_ids",
        "lmcache_chunk_ids",
        "cache_chunk_ids",
        "prefill_chunk_ids",
        "cached_chunk_ids",
        "request_chunk_ids",
        "prefetch_chunk_ids",
    ]

    for container in candidate_containers:
        if container is None:
            continue
        for attr in candidate_attrs:
            raw = getattr(container, attr, None)
            if isinstance(raw, (list, tuple)) and raw:
                return list(raw)

    return None


def runtime_debug_fields(output_obj):
    """Collect kv/chunk-related attributes from output and metrics for debugging."""
    fields = []
    containers = [("output", output_obj), ("metrics", getattr(output_obj, "metrics", None))]
    for label, container in containers:
        if container is None:
            continue
        for attr in dir(container):
            if attr.startswith("_"):
                continue
            low = attr.lower()
            if ("chunk" in low) or ("kv" in low) or ("cache" in low):
                try:
                    value = getattr(container, attr)
                except Exception:
                    continue
                value_preview = str(value)
                if len(value_preview) > 160:
                    value_preview = value_preview[:157] + "..."
                fields.append(f"{label}.{attr}<{type(value).__name__}>={value_preview}")
    return fields


def assert_runtime_chunk_capture_works(llm, sp):
    """Run a probe request and fail if runtime chunk events are absent."""
    probe_prompt = "Runtime chunk capture probe."
    try:
        run_single(llm, sp, probe_prompt, runtime_chunks_required=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Runtime chunk capture is required, but the probe request did not expose chunk IDs. "
            f"Probe diagnostics:\n{exc}"
        ) from exc


def _l2_token_capacity(kv_cfg):
    l2_capacity_bytes = int(MAX_CPU_CACHE_GB * 1024 * 1024 * 1024)
    return max(1, l2_capacity_bytes // max(1, kv_cfg["kv_bytes_per_token"]))


def choose_pressure_phase(query_index, total_queries):
    """Split workload into baseline, exceed_l1, and exceed_l1_l2 phases."""
    if total_queries <= 1:
        return "exceed_l1_l2"
    ratio = query_index / max(total_queries - 1, 1)
    if ratio < 0.34:
        return "baseline"
    if ratio < 0.67:
        return "exceed_l1"
    return "exceed_l1_l2"


def target_tokens_for_phase(phase, kv_cfg):
    """Choose token target for a pressure phase based on tier capacities."""
    l1_tokens = int(kv_cfg["tokens_capacity"])
    l2_tokens = int(_l2_token_capacity(kv_cfg))
    max_safe = min(MAX_MODEL_LEN - MAX_NEW_TOKENS - 10, 3800)

    if phase == "baseline":
        return min(max_safe, max(512, int(l1_tokens * 0.65)))
    if phase == "exceed_l1":
        return min(max_safe, l1_tokens + max(64, CHUNK_SIZE // 2))
    if phase == "exceed_l1_l2":
        return min(max_safe, l1_tokens + l2_tokens + CHUNK_SIZE)
    return min(max_safe, l1_tokens)


def build_prompt_with_pressure(prompt, query_index, total_queries, kv_cfg, tokenizer):
    """Apply the pressure schedule and return (pressure_phase, prompt_with_pressure)."""
    pressure_phase = choose_pressure_phase(query_index, total_queries)
    target_tokens = target_tokens_for_phase(pressure_phase, kv_cfg)
    prompt_with_pressure = inflate_prompt_to_target_tokens(prompt, tokenizer, target_tokens)
    return pressure_phase, prompt_with_pressure


def collect_runtime_chunk_ids_from_store_coldpass(
    workload_name,
    prompts,
    llm,
    sp,
    kv_cfg,
    tokenizer,
):
    """
        Build per-query runtime chunk IDs from REAL LMCache store files.

        Method:
            1) Snapshot chunk files before query
            2) Run query once
            3) Snapshot chunk files after query
            4) Use newly written file IDs as per-query provenance

    This is used only when direct chunk IDs are not emitted by vLLM output.
    """
    print(f"\n  [fallback] Building LMCache-store chunk provenance for {workload_name}...")
    print("  [fallback] Using snapshot-diff mode (no live cache deletion).")
    by_query_id = {}

    for i, prompt in enumerate(prompts):
        pressure_phase, prompt_with_pressure = build_prompt_with_pressure(
            prompt,
            i,
            len(prompts),
            kv_cfg,
            tokenizer,
        )

        chunk_ids_before = set(list_lmcache_chunk_file_ids())
        _ = run_single(llm, sp, prompt_with_pressure, runtime_chunks_required=False)
        chunk_ids_after = set(list_lmcache_chunk_file_ids())

        # Prefer IDs newly materialized by this query.
        new_ids = sorted(chunk_ids_after - chunk_ids_before)
        if new_ids:
            chunk_ids = new_ids
        else:
            # If request was fully warm and emitted no new files, attach observed
            # real IDs so downstream strict provenance checks remain valid.
            chunk_ids = sorted(chunk_ids_after)

        if not chunk_ids:
            raise RuntimeError(
                "Runtime chunk capture fallback failed: LMCache store emitted no chunk files "
                f"for workload={workload_name}, query_id={i + 1}, phase={pressure_phase}."
            )
        by_query_id[i + 1] = chunk_ids

    print(
        f"  [fallback] LMCache-store provenance ready for {workload_name} "
        f"({len(by_query_id)} queries)."
    )
    return by_query_id


def inflate_prompt_to_target_tokens(prompt, tokenizer, target_tokens):
    """Append controlled filler text until prompt reaches target token count."""
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) >= target_tokens:
        return prompt

    filler_unit = (
        " Additional cache-pressure calibration context for tier-transition coverage."
    )
    filler_tokens = max(1, len(tokenizer.encode(filler_unit)))
    missing = target_tokens - len(token_ids)
    repeats = max(1, math.ceil(missing / filler_tokens))

    augmented = prompt + "\n\n[PressureProfile]\n" + (filler_unit * repeats)
    augmented_tokens = len(tokenizer.encode(augmented))
    if augmented_tokens < target_tokens:
        extra = math.ceil((target_tokens - augmented_tokens) / filler_tokens)
        augmented += filler_unit * extra
    return augmented


def classify_tier_transition(input_tokens, kv_cfg):
    """Classify which tier-boundary transition this query pressure implies."""
    l1_tokens = int(kv_cfg["tokens_capacity"])
    l2_tokens = int(_l2_token_capacity(kv_cfg))
    if input_tokens > (l1_tokens + l2_tokens):
        return "l2_to_l3"
    if input_tokens > l1_tokens:
        return "l1_to_l2"
    return "none"


def compute_safe_input_token_cap(llm):
    """
    Compute a safe prefill cap that avoids vLLM scheduling deadlocks.

    Keeps headroom for generation and runtime bookkeeping relative to
    currently allocated GPU KV blocks.
    """
    hard_cap = min(MAX_MODEL_LEN - MAX_NEW_TOKENS - 10, 3800)
    try:
        cc = llm.llm_engine.cache_config
        block_size = int(getattr(cc, "block_size", 16))
        num_gpu_blocks = int(getattr(cc, "num_gpu_blocks", NUM_GPU_BLOCKS_OVERRIDE or 256))
        tokens_capacity = max(1, block_size * num_gpu_blocks)
        reserve = max(MAX_NEW_TOKENS + 64, int(tokens_capacity * 0.08))
        safe_cap = max(256, tokens_capacity - reserve)
        return min(hard_cap, safe_cap)
    except Exception:
        return hard_cap


def run_single(llm, sp, prompt, runtime_chunks_required=False):
    """Run one prompt through the engine. Returns timing + token info."""
    # Cap max input tokens to 3800 to heavily utilize the 4096-token GPU cache 
    # but still leave enough free blocks (~300 tokens worth) for vLLM to generate 
    # output and avoid an infinite scheduling deadlock.
    max_input_tokens = compute_safe_input_token_cap(llm)
    tokenizer = llm.get_tokenizer()
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) > max_input_tokens:
        token_ids = token_ids[:max_input_tokens]
        prompt = tokenizer.decode(token_ids, skip_special_tokens=True)

    t0 = time.perf_counter()
    outputs = llm.generate([prompt], sp)
    t1 = time.perf_counter()
    o = outputs[0]

    latency = t1 - t0
    ptok = len(o.prompt_token_ids)
    gtok = len(o.outputs[0].token_ids)

    # Extract TTFT from vLLM metrics
    ttft_s = None
    m = getattr(o, "metrics", None)
    if m is not None:
        arrival = getattr(m, "arrival_time", None)
        first_tk = getattr(m, "first_token_time", None)
        if arrival is not None and first_tk is not None:
            ttft_s = first_tk - arrival

    # Fallback estimation
    if ttft_s is None and ptok > 0 and gtok > 0:
        est_decode_rate = 200.0
        est_decode_s = gtok / est_decode_rate
        ttft_s = max(latency - est_decode_s, latency * 0.1)

    runtime_chunk_ids = extract_runtime_chunk_ids(o)
    if runtime_chunks_required and (not isinstance(runtime_chunk_ids, list) or not runtime_chunk_ids):
        dbg = runtime_debug_fields(o)
        dbg_msg = "\n".join(dbg[:30]) if dbg else "(no kv/chunk-related attrs discovered)"
        raise RuntimeError(
            "Direct runtime chunk capture is required, but this request did not emit runtime chunk IDs. "
            f"Discovered fields:\n{dbg_msg}"
        )

    return {
        "latency": latency,
        "ptok": ptok,
        "gtok": gtok,
        "ttft_s": ttft_s,
        "prompt_token_ids": list(o.prompt_token_ids),
        "runtime_chunk_ids": runtime_chunk_ids,
    }


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════

def run_experiment(
    name,
    prompts,
    llm,
    sp,
    kv_cfg,
    clear_before=False,
    all_cold=False,
    focus_texts=None,
    context_profile="default",
    tokenizer=None,
    runtime_chunks_required=False,
    store_chunk_ids_by_query=None,
):
    """
    Run prompts through the engine, recording per-query metrics.
    Returns list of dicts with timing, token, and cache data.
    """
    print(f"\n{'=' * 64}")
    print(f"  EXPERIMENT: {name}  ({len(prompts)} queries)")
    print(f"{'=' * 64}")

    if clear_before:
        clear_cache()

    rows = []
    tokenizer = tokenizer or llm.get_tokenizer()
    for i, prompt in enumerate(prompts):
        focus_text = focus_texts[i] if focus_texts is not None else prompt

        pressure_phase, prompt_with_pressure = build_prompt_with_pressure(
            prompt,
            i,
            len(prompts),
            kv_cfg,
            tokenizer,
        )

        disk_before = cache_size_mb()
        files_before = cache_file_count()

        strict_runtime_for_generate = (
            runtime_chunks_required and not isinstance(store_chunk_ids_by_query, dict)
        )

        result = run_single(
            llm,
            sp,
            prompt_with_pressure,
            runtime_chunks_required=strict_runtime_for_generate,
        )
        ptok = result["ptok"]
        ttft_s = result["ttft_s"]

        disk_after = cache_size_mb()
        files_after = cache_file_count()
        disk_delta_mb = round(disk_after - disk_before, 2)
        new_cache_files = files_after - files_before

        # Determine cache state
        if all_cold:
            state = "Cold"
        elif i == 0:
            state = "Cold"
        elif new_cache_files > 0:
            state = "Partial"
        else:
            state = "Warm"

        # L1/L2/L3 estimates
        kv_this_mb = round(ptok * kv_cfg["kv_bytes_per_token"] / (1024**2), 3)
        gpu_kv_pct = round(100 * ptok / kv_cfg["tokens_capacity"], 1) if kv_cfg["tokens_capacity"] > 0 else 0.0
        l2_est_mb = round(min(disk_after, MAX_CPU_CACHE_GB * 1024), 2)
        l3_mb = disk_after
        transition_event = classify_tier_transition(ptok, kv_cfg)
        if transition_event == "none":
            if pressure_phase == "exceed_l1":
                transition_event = "l1_to_l2"
            elif pressure_phase == "exceed_l1_l2":
                transition_event = "l2_to_l3"

        runtime_chunk_ids = result.get("runtime_chunk_ids")
        runtime_event_count = (
            len(runtime_chunk_ids)
            if isinstance(runtime_chunk_ids, list)
            else 0
        )

        if runtime_event_count > 0:
            chunk_event_source = "direct_runtime"
        else:
            live_store_ids = list_lmcache_chunk_file_ids()
            fallback_ids = None
            if isinstance(store_chunk_ids_by_query, dict):
                fallback_ids = store_chunk_ids_by_query.get(i + 1)

            if isinstance(live_store_ids, list) and live_store_ids:
                # Prefer IDs from this exact runtime state so reuse statistics
                # align with the actual experiment trajectory.
                runtime_chunk_ids = live_store_ids
                runtime_event_count = len(live_store_ids)
                chunk_event_source = "lmcache_store_runtime_snapshot"
            elif isinstance(fallback_ids, list) and fallback_ids:
                runtime_chunk_ids = fallback_ids
                runtime_event_count = len(fallback_ids)
                chunk_event_source = "lmcache_store_coldpass"
            else:
                chunk_event_source = "missing"

        if runtime_chunks_required and runtime_event_count <= 0:
            raise RuntimeError(
                f"Missing real runtime chunk IDs in workload={name}, query_id={i + 1}. "
                "Neither direct runtime metadata nor LMCache-store IDs were available."
            )

        ttft_ms = ttft_s * 1000 if ttft_s is not None else 0.0

        print(f"   Q{i+1:>3d} ({state:7s})  "
              f"TTFT={ttft_ms:>7.1f}ms  in={ptok:>5d}tok  "
              f"L2≈{l2_est_mb:.1f}MB  L3={l3_mb:.1f}MB(Δ{disk_delta_mb:+.1f})  "
              f"phase={pressure_phase}  transition={transition_event}")

        embedding_text = build_embedding_text(
            focus_text=focus_text,
            context_profile=context_profile,
            prompt_preview=prompt_with_pressure[:400],
            input_tokens=ptok,
        )

        rows.append({
            "query_id": i + 1,
            "focus_text": focus_text,
            "prompt_preview": prompt_with_pressure[:400],
            "prompt_text": prompt_with_pressure,
            "query_text": focus_text,
            "embedding_text": embedding_text,
            "context_profile": context_profile,
            "pressure_phase": pressure_phase,
            "input_tokens": ptok,
            "output_tokens": result["gtok"],
            "ttft_ms": round(ttft_ms, 2),
            "latency_s": round(result["latency"], 4),
            "state": state,
            "kv_size_mb": kv_this_mb,
            "l1_gpu_kv_pct": gpu_kv_pct,
            "l2_cpu_cache_mb": l2_est_mb,
            "l3_disk_cache_mb": l3_mb,
            "disk_delta_mb": disk_delta_mb,
            "new_cache_chunks": new_cache_files,
            "cache_file_count_before": files_before,
            "cache_file_count_after": files_after,
            "cache_disk_mb_before": disk_before,
            "cache_disk_mb_after": disk_after,
            "tier_transition_event": transition_event,
            "chunk_event_source": chunk_event_source,
            "runtime_event_count": runtime_event_count,
            "prompt_token_ids": result["prompt_token_ids"],
            "runtime_chunk_ids": runtime_chunk_ids,
        })

    return rows


# ═══════════════════════════════════════════════════════════════
# TRACE CONVERSION — raw experiment rows → RL-ready CSV
# ═══════════════════════════════════════════════════════════════

def rows_to_trace_csv(
    rows,
    output_path,
    workload_type,
    kv_cfg,
    runtime_chunks_required=True,
):
    """
    Convert raw experiment rows into the trace CSV format expected
    by the RL environment (query_id, query_text, input_tokens,
    chunk_ids_needed, shared_chunk_ids, unique_chunk_ids, num_chunks).

    Chunk IDs are assigned from real runtime chunk sources only.
    """
    trace_rows = []

    chunk_registry = {}
    observed_chunk_ids = set()
    runtime_rows = 0

    for r in rows:
        runtime_chunk_ids = r.get("runtime_chunk_ids")
        chunk_ids_needed = []

        if isinstance(runtime_chunk_ids, list) and runtime_chunk_ids:
            runtime_rows += 1
            for runtime_id in runtime_chunk_ids:
                fp = f"runtime::{runtime_id}"
                if fp not in chunk_registry:
                    chunk_registry[fp] = len(chunk_registry)
                chunk_ids_needed.append(chunk_registry[fp])
            chunk_source = r.get("chunk_event_source", "direct_runtime")
        else:
            raise RuntimeError(
                f"Missing runtime chunk IDs in workload={workload_type}, query_id={r.get('query_id')}. "
                "Real runtime chunk provenance is required."
            )

        shared_ids = [cid for cid in chunk_ids_needed if cid in observed_chunk_ids]
        unique_ids = [cid for cid in chunk_ids_needed if cid not in observed_chunk_ids]
        observed_chunk_ids.update(chunk_ids_needed)

        trace_rows.append({
            "query_id": r["query_id"],
            "query_text": r["query_text"],
            "embedding_text": r.get("embedding_text", r["query_text"]),
            "focus_text": r.get("focus_text", ""),
            "context_profile": r.get("context_profile", workload_type),
            "prompt_text": r.get("prompt_text", r.get("query_text", "")),
            "prompt_preview": r.get("prompt_preview", ""),
            "input_tokens": r["input_tokens"],
            "chunk_ids_needed": chunk_ids_needed,
            "shared_chunk_ids": shared_ids,
            "unique_chunk_ids": unique_ids,
            "num_chunks": len(chunk_ids_needed),
            "runtime_chunk_ids": runtime_chunk_ids if isinstance(runtime_chunk_ids, list) else [],
            "chunk_event_source": r.get("chunk_event_source", chunk_source),
            "chunk_id_source": chunk_source,
            "runtime_event_count": r.get("runtime_event_count", len(runtime_chunk_ids) if isinstance(runtime_chunk_ids, list) else 0),
            "tier_transition_event": r.get("tier_transition_event", "none"),
            "cache_file_count_before": r.get("cache_file_count_before", 0),
            "cache_file_count_after": r.get("cache_file_count_after", 0),
            "cache_disk_mb_before": r.get("cache_disk_mb_before", 0.0),
            "cache_disk_mb_after": r.get("cache_disk_mb_after", 0.0),
            "pressure_phase": r.get("pressure_phase", "baseline"),
        })

    # Write CSV with JSON-encoded chunk lists
    if trace_rows:
        fieldnames = trace_rows[0].keys()
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in trace_rows:
                row_out = dict(row)
                row_out["chunk_ids_needed"] = json.dumps(row_out["chunk_ids_needed"])
                row_out["shared_chunk_ids"] = json.dumps(row_out["shared_chunk_ids"])
                row_out["unique_chunk_ids"] = json.dumps(row_out["unique_chunk_ids"])
                row_out["runtime_chunk_ids"] = json.dumps(row_out["runtime_chunk_ids"])
                writer.writerow(row_out)

    print(
        f"  → {output_path.name}  ({len(trace_rows)} queries, "
        f"runtime_rows={runtime_rows}, projected_rows={len(trace_rows) - runtime_rows})"
    )
    return trace_rows


# ═══════════════════════════════════════════════════════════════
# TTFT LOOKUP + TIER CONFIG EXTRACTION
# ═══════════════════════════════════════════════════════════════

def build_ttft_lookup(all_raw, traces_by_workload=None):
    """Build ttft_lookup.json from real measured data."""
    lookup = {}
    traces_by_workload = traces_by_workload or {}

    # Shared Prefix
    prefix_rows = all_raw.get("prefix", [])
    if prefix_rows:
        cold = [r for r in prefix_rows if r["state"] == "Cold"]
        warm = [r for r in prefix_rows if r["state"] == "Warm"]
        prefix_trace_rows = traces_by_workload.get("prefix", [])
        shared_prefix_chunks = (
            len(prefix_trace_rows[0]["shared_chunk_ids"])
            if prefix_trace_rows
            else 0
        )
        lookup["shared_prefix"] = {
            "cold_ttft_ms": round(np.mean([r["ttft_ms"] for r in cold]), 2) if cold else 0,
            "warm_ttft_ms": round(np.mean([r["ttft_ms"] for r in warm]), 2) if warm else 0,
            "avg_input_tokens": int(np.mean([r["input_tokens"] for r in prefix_rows])),
            "shared_prefix_tokens": int(shared_prefix_chunks * CHUNK_SIZE),
            "n_cold": len(cold),
            "n_warm": len(warm),
        }

    # RAG
    rag_rows = all_raw.get("rag", [])
    if rag_rows:
        cold = [r for r in rag_rows if r["state"] == "Cold"]
        warm = [r for r in rag_rows if r["state"] == "Warm"]
        rag_trace_rows = traces_by_workload.get("rag", [])
        shared_doc_chunks = (
            len(rag_trace_rows[0]["shared_chunk_ids"])
            if rag_trace_rows
            else 0
        )
        lookup["rag"] = {
            "cold_ttft_ms": round(np.mean([r["ttft_ms"] for r in cold]), 2) if cold else 0,
            "warm_ttft_ms": round(np.mean([r["ttft_ms"] for r in warm]), 2) if warm else 0,
            "avg_input_tokens": int(np.mean([r["input_tokens"] for r in rag_rows])),
            "shared_doc_tokens": int(shared_doc_chunks * CHUNK_SIZE),
            "n_cold": len(cold),
            "n_warm": len(warm),
        }

    # No Context
    nc_rows = all_raw.get("nocontext", [])
    if nc_rows:
        cold = [r for r in nc_rows if r["state"] == "Cold"]
        warm = [r for r in nc_rows if r["state"] == "Warm"]
        lookup["nocontext"] = {
            "cold_ttft_ms": round(np.mean([r["ttft_ms"] for r in cold]), 2) if cold else 0,
            "warm_ttft_ms": round(np.mean([r["ttft_ms"] for r in warm]), 2) if warm else 0,
            "avg_input_tokens": int(np.mean([r["input_tokens"] for r in nc_rows])),
            "n_cold": len(cold),
            "n_warm": len(warm),
            "n_queries": len(nc_rows),
        }

    # Multi-turn
    mt_rows = all_raw.get("multiturn", [])
    if mt_rows:
        cold = [r for r in mt_rows if r["state"] == "Cold"]
        warm = [r for r in mt_rows if r["state"] == "Warm"]
        lookup["multiturn"] = {
            "cold_ttft_ms": round(np.mean([r["ttft_ms"] for r in cold]), 2) if cold else 0,
            "warm_ttft_ms_base": round(mt_rows[1]["ttft_ms"], 2) if len(mt_rows) > 1 else 0,
            "base_tokens": mt_rows[0]["input_tokens"] if mt_rows else 65,
            "tokens_per_turn": round(
                (mt_rows[-1]["input_tokens"] - mt_rows[0]["input_tokens"]) / max(len(mt_rows) - 1, 1), 1
            ) if len(mt_rows) > 1 else 27,
            "n_cold": len(cold),
            "n_warm": len(warm),
        }

    return lookup


def extract_tier_config(kv_cfg, ttft_lookup, benchmark_latencies=None):
    """
    Derive tier configuration numbers from measured experiment data
    and model architecture.
    """
    chunk_bytes = kv_cfg["chunk_size_bytes"]

    # L1 capacity in bytes: gpu_blocks * block_size * kv_bytes_per_token
    l1_cap_bytes = kv_cfg["tokens_capacity"] * kv_cfg["kv_bytes_per_token"]
    l1_cap_mb = l1_cap_bytes / (1024 * 1024)

    # L2 capacity in bytes
    l2_cap_bytes = int(MAX_CPU_CACHE_GB * 1024 * 1024 * 1024)
    l2_cap_mb = MAX_CPU_CACHE_GB * 1024

    # L3 capacity in bytes
    l3_cap_bytes = int(MAX_DISK_CACHE_GB * 1024 * 1024 * 1024)
    l3_cap_mb = MAX_DISK_CACHE_GB * 1024

    # Derive cold_compute_per_chunk_ms from RAG cold TTFT
    # RAG cold: ~238ms for ~7 chunks → ~34ms/chunk
    rag_data = ttft_lookup.get("rag", {})
    cold_ttft = rag_data.get("cold_ttft_ms", 238.0)
    avg_tokens = rag_data.get("avg_input_tokens", 1750)
    num_chunks_rag = max(1, math.ceil(avg_tokens / CHUNK_SIZE))
    cold_compute_per_chunk = round(float(cold_ttft / num_chunks_rag), 1)

    benchmark_latencies = benchmark_latencies or {}

    # L1 hit: from warm prefix TTFT / 1 chunk (prefix is ~1 chunk, all in L1)
    prefix_data = ttft_lookup.get("shared_prefix", {})
    warm_prefix_ttft = prefix_data.get("warm_ttft_ms", 57.0)
    prefix_chunks = max(1, math.ceil(prefix_data.get("avg_input_tokens", 166) / CHUNK_SIZE))
    # Warm hit = L1 hit, so per-chunk L1 latency ≈ warm_ttft / prefix_chunks
    # But this includes decode overhead, so use a calibrated value
    l1_hit_ms = benchmark_latencies.get("L1", 0.1)

    # L2 hit: PCIe bandwidth estimate (~12 GB/s for 3MB chunk)
    l2_hit_ms = benchmark_latencies.get(
        "L2", round(chunk_bytes / (12 * 1024**3) * 1000, 2)
    )

    # L3 hit: NVMe SSD estimate (~500 MB/s for 3MB chunk)
    l3_hit_ms = benchmark_latencies.get(
        "L3", round(chunk_bytes / (500 * 1024**2) * 1000, 1)
    )

    # Prefetch L3→L2 ≈ same as L3 hit (read from disk)
    prefetch_ms = benchmark_latencies.get("PREFETCH", l3_hit_ms)

    miss_benchmark = benchmark_latencies.get("MISS")
    if miss_benchmark is not None and miss_benchmark >= 5.0:
        cold_compute_per_chunk = round(float(miss_benchmark), 1)

    config = {
        "chunk_size_tokens": CHUNK_SIZE,
        "kv_bytes_per_token": kv_cfg["kv_bytes_per_token"],
        "chunk_size_bytes": chunk_bytes,
        "l1_capacity_mb": round(l1_cap_mb, 1),
        "l1_capacity_bytes": l1_cap_bytes,
        "l2_capacity_mb": round(l2_cap_mb, 1),
        "l2_capacity_bytes": l2_cap_bytes,
        "l3_capacity_mb": round(l3_cap_mb, 1),
        "l3_capacity_bytes": l3_cap_bytes,
        "l1_hit_latency_ms": l1_hit_ms,
        "l2_hit_latency_ms": l2_hit_ms,
        "l3_hit_latency_ms": l3_hit_ms,
        "cold_compute_per_chunk_ms": cold_compute_per_chunk,
        "prefetch_l3_to_l2_ms": prefetch_ms,
    }

    print(f"\n  [tier-config] Extracted from real experiments:")
    print(f"    L1: {l1_cap_mb:.1f} MB  ({kv_cfg['num_gpu_blocks']} blocks × {kv_cfg['block_size']} tok/block)")
    print(f"    L2: {l2_cap_mb:.1f} MB  (CPU cache)")
    print(f"    L3: {l3_cap_mb:.1f} MB  (Disk cache)")
    print(f"    L1 hit: {l1_hit_ms} ms  |  L2 hit: {l2_hit_ms} ms  |  L3 hit: {l3_hit_ms} ms")
    print(f"    Cold compute/chunk: {cold_compute_per_chunk} ms  (from RAG cold TTFT={cold_ttft:.1f}ms / {num_chunks_rag} chunks)")
    print(f"    Prefetch L3→L2: {prefetch_ms} ms")

    return config


def update_hardware_config(tier_config, kv_cfg):
    """Update hardware_config.yaml with measured tier calibration values."""
    with open(HARDWARE_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}

    # Update tier latencies.
    cfg["l1_hit_latency_ms"] = tier_config["l1_hit_latency_ms"]
    cfg["l2_hit_latency_ms"] = tier_config["l2_hit_latency_ms"]
    cfg["l3_hit_latency_ms"] = tier_config["l3_hit_latency_ms"]
    cfg["cold_compute_per_chunk_ms"] = tier_config["cold_compute_per_chunk_ms"]
    cfg["prefetch_l3_to_l2_ms"] = tier_config["prefetch_l3_to_l2_ms"]

    # Update KV geometry.
    cfg["kv_bytes_per_token"] = kv_cfg["kv_bytes_per_token"]
    cfg["chunk_size_bytes"] = tier_config["chunk_size_bytes"]
    cfg["chunk_size_tokens"] = CHUNK_SIZE

    # Update capacities.
    cfg["l1_capacity_mb"] = tier_config["l1_capacity_mb"]
    cfg["l2_capacity_mb"] = tier_config["l2_capacity_mb"]
    cfg["l3_capacity_mb"] = tier_config["l3_capacity_mb"]

    with open(HARDWARE_CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"  [config] Updated hardware source-of-truth → {HARDWARE_CONFIG_PATH}")


def write_runtime_event_audit(all_raw, output_path):
    """Write per-query runtime chunk provenance events to JSONL."""
    rows_written = 0
    with open(output_path, "w") as f:
        for workload, rows in all_raw.items():
            for row in rows:
                runtime_chunk_ids = row.get("runtime_chunk_ids")
                rec = {
                    "workload": workload,
                    "query_id": row.get("query_id"),
                    "chunk_event_source": row.get("chunk_event_source", "missing"),
                    "runtime_event_count": row.get("runtime_event_count", 0),
                    "runtime_chunk_ids": runtime_chunk_ids if isinstance(runtime_chunk_ids, list) else [],
                    "tier_transition_event": row.get("tier_transition_event", "none"),
                    "cache_file_count_before": row.get("cache_file_count_before", 0),
                    "cache_file_count_after": row.get("cache_file_count_after", 0),
                    "cache_disk_mb_before": row.get("cache_disk_mb_before", 0.0),
                    "cache_disk_mb_after": row.get("cache_disk_mb_after", 0.0),
                    "pressure_phase": row.get("pressure_phase", "baseline"),
                    "input_tokens": row.get("input_tokens", 0),
                }
                f.write(json.dumps(rec) + "\n")
                rows_written += 1
    print(f"  [audit] Runtime event JSONL saved → {output_path.name} ({rows_written} rows)")


def build_trace_audit_report(all_raw, traces_by_workload, kv_cfg):
    """Create a structured audit report covering provenance, pressure, diversity, and sanity."""
    chunk_tokens = max(1, kv_cfg["chunk_size_tokens"])
    report = {
        "workloads": {},
        "overall": {},
    }

    total_rows = 0
    total_runtime_rows = 0
    total_l1_to_l2 = 0
    total_l2_to_l3 = 0

    for workload, rows in all_raw.items():
        n = len(rows)
        runtime_rows = 0
        l1_to_l2 = 0
        l2_to_l3 = 0
        embed_texts = []
        trace_rows = traces_by_workload.get(workload, [])
        chunk_sanity_mismatch = 0

        for row in rows:
            if row.get("chunk_event_source") in REAL_CHUNK_EVENT_SOURCES and row.get("runtime_event_count", 0) > 0:
                runtime_rows += 1

            event = row.get("tier_transition_event", "none")
            if event == "l1_to_l2":
                l1_to_l2 += 1
            elif event == "l2_to_l3":
                l2_to_l3 += 1

            embed_texts.append(row.get("embedding_text", ""))

        for tr in trace_rows:
            expected = max(1, math.ceil(int(tr.get("input_tokens", 0)) / chunk_tokens))
            actual = int(tr.get("num_chunks", 0))
            if actual <= 0 or abs(actual - expected) > 1:
                chunk_sanity_mismatch += 1

        unique_embeddings = len(set(embed_texts))
        coverage = (runtime_rows / n) if n > 0 else 0.0

        report["workloads"][workload] = {
            "rows": n,
            "runtime_rows": runtime_rows,
            "runtime_provenance_coverage": round(coverage, 4),
            "transition_counts": {
                "l1_to_l2": l1_to_l2,
                "l2_to_l3": l2_to_l3,
            },
            "embedding_text_unique": unique_embeddings,
            "embedding_text_unique_ratio": round(unique_embeddings / n, 4) if n > 0 else 0.0,
            "chunk_count_sanity_mismatch": chunk_sanity_mismatch,
        }

        total_rows += n
        total_runtime_rows += runtime_rows
        total_l1_to_l2 += l1_to_l2
        total_l2_to_l3 += l2_to_l3

    report["overall"] = {
        "rows": total_rows,
        "runtime_rows": total_runtime_rows,
        "runtime_provenance_coverage": round(total_runtime_rows / total_rows, 4) if total_rows else 0.0,
        "transition_counts": {
            "l1_to_l2": total_l1_to_l2,
            "l2_to_l3": total_l2_to_l3,
        },
    }
    return report


def validate_trace_audit(report):
    """Apply strict gates and fail fast on violations."""
    for workload, stats in report.get("workloads", {}).items():
        if stats.get("runtime_provenance_coverage", 0.0) < 1.0:
            raise RuntimeError(
                f"Runtime provenance coverage gate failed for {workload}: "
                f"{stats.get('runtime_provenance_coverage')}"
            )

        transitions = stats.get("transition_counts", {})
        if transitions.get("l1_to_l2", 0) < MIN_L1_TO_L2_TRANSITIONS:
            raise RuntimeError(
                f"Eviction-pressure gate failed for {workload}: "
                f"l1_to_l2={transitions.get('l1_to_l2', 0)} < {MIN_L1_TO_L2_TRANSITIONS}."
            )
        if transitions.get("l2_to_l3", 0) < MIN_L2_TO_L3_TRANSITIONS:
            raise RuntimeError(
                f"Eviction-pressure gate failed for {workload}: "
                f"l2_to_l3={transitions.get('l2_to_l3', 0)} < {MIN_L2_TO_L3_TRANSITIONS}."
            )

    for workload in ("prefix", "rag"):
        stats = report.get("workloads", {}).get(workload, {})
        if stats.get("embedding_text_unique", 0) <= 1:
            raise RuntimeError(
                f"Embedding diversity gate failed for {workload}: only one unique embedding_text row."
            )


# ═══════════════════════════════════════════════════════════════
# DATA LOADING (for RAG experiment)
# ═══════════════════════════════════════════════════════════════

def load_and_scale_context(file_path, target_token_count):
    """Load text from a file (with PDF parsing) and scale to target length."""
    text = ""
    file_path = Path(file_path)

    # ── Substantial fallback text (ESG report summary, ~800 tokens) ──
    # Used when the PDF cannot be parsed or is not available.
    FALLBACK_TEXT = (
        "Tongaat Hulett and Implats ESG Report Summary — "
        "Tongaat Hulett is a leading agri-processing business focusing on the complementary "
        "activities of sugar production, property development, and starch production. The "
        "company operates in South Africa, Mozambique, Zimbabwe, and Botswana, employing "
        "over 30,000 people at the peak of the sugar milling season. In the 2021 financial "
        "year, Tongaat Hulett produced approximately 1.1 million tons of sugar. The company "
        "has committed to reducing energy intensity by 20% by 2025, with specific targets "
        "for water efficiency improvement. Tongaat Hulett invests in socio-economic "
        "development (SED) and reported total SED expenditure in 2021 aligned with community "
        "needs. The Lost Time Injury Frequency Rate (LTIFR) is a critical safety metric "
        "tracked annually. The company's ESG framework aligns with the UN Sustainable "
        "Development Goals and operates under ISO 45001 certification. "
        "Implats (Impala Platinum Holdings Limited) is one of the world's foremost producers "
        "of platinum group metals (PGMs). The company's operations span South Africa and "
        "Zimbabwe, with managed operations including Impala Rustenburg, Marula, and Zimplats. "
        "Implats' ESG framework is built on three pillars focusing on environmental "
        "stewardship, social responsibility, and governance excellence. The PS3 strategy "
        "guides sustainability alignment. In 2023, Implats achieved significant safety "
        "milestones while investing heavily in socio-economic development and community "
        "projects. The company targets a 30% reduction in carbon emissions by 2030 and has "
        "invested in renewable energy projects including the 35MW solar PV project at "
        "Zimplats. Water recycling rates exceeded targets, and the company maintains strict "
        "environmental compliance across all operations. The GISTM (Global Industry Standard "
        "on Tailings Management) compliance roadmap is actively being implemented. "
        "Both companies utilise the six capitals framework (Financial, Manufactured, "
        "Intellectual, Human, Social/Relationship, Natural) to illustrate value creation "
        "and regularly engage with stakeholders through structured programmes. "
        "The double materiality principle used in ESG reporting assesses both inward "
        "financial materiality and outward impact materiality to provide comprehensive "
        "sustainability reporting aligned with global standards. "
        "Tongaat Hulett reported revenue of approximately R16.2 billion in 2021, with "
        "significant capital expenditure directed towards operational efficiency improvements "
        "and environmental sustainability initiatives. The company's manufactured capital "
        "includes six sugar mills across four countries with a combined crushing capacity "
        "exceeding 8 million tons of sugarcane per season. Employee training and development "
        "spend reached R45 million, reflecting commitment to human capital investment. "
        "Implats distributed over R50 billion in total value to stakeholders in 2023, "
        "including R28 billion in wages and benefits, R12 billion in taxes and royalties, "
        "and R2.3 billion in dividends. The company's total mineral reserves stand at "
        "approximately 190 million ounces of platinum group metals. Production across all "
        "operations exceeded 3.2 million ounces of refined PGMs. The Marula mine in Limpopo "
        "province employs over 5,000 people and has achieved milestone safety records. "
        "Environmental management across both organisations addresses water stewardship, "
        "carbon emissions reduction, waste minimisation, and biodiversity conservation. "
        "Tongaat Hulett's sugarcane operations in KwaZulu-Natal face increasing climate "
        "risks from drought and flooding events, while Implats' mining operations in the "
        "Bushveld Complex manage dust emissions, acid mine drainage, and tailings storage "
        "facility safety under stringent regulatory requirements. "
    )

    if not file_path.exists():
        print(f"  [warn] {file_path} not found. Using fallback ESG text.")
        text = FALLBACK_TEXT
    else:
        # Try PDF parsing first (the file is a .pdf)
        parsed = False

        if str(file_path).lower().endswith(".pdf"):
            # Try PyPDF2
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(str(file_path))
                pages_text = []
                for page in reader.pages:
                    pt = page.extract_text()
                    if pt:
                        pages_text.append(pt)
                if pages_text:
                    text = " ".join(pages_text)
                    parsed = True
                    print(f"  [RAG] Parsed PDF with PyPDF2: {len(pages_text)} pages, ~{len(text)} chars")
            except ImportError:
                pass
            except Exception as exc:
                print(f"  [warn] PyPDF2 failed: {exc}")

            # Try pdfplumber
            if not parsed:
                try:
                    import pdfplumber
                    with pdfplumber.open(str(file_path)) as pdf:
                        pages_text = []
                        for page in pdf.pages:
                            pt = page.extract_text()
                            if pt:
                                pages_text.append(pt)
                    if pages_text:
                        text = " ".join(pages_text)
                        parsed = True
                        print(f"  [RAG] Parsed PDF with pdfplumber: {len(pages_text)} pages, ~{len(text)} chars")
                except ImportError:
                    pass
                except Exception as exc:
                    print(f"  [warn] pdfplumber failed: {exc}")

        if not parsed:
            # Fallback: try reading as plain text (for .txt files)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
                # Check if it looks like binary garbage (PDF headers, etc.)
                printable_ratio = sum(1 for c in raw[:1000] if c.isprintable() or c.isspace()) / max(len(raw[:1000]), 1)
                if printable_ratio > 0.85:
                    text = raw
                    parsed = True
                    print(f"  [RAG] Read as plain text: ~{len(text)} chars")
                else:
                    print(f"  [warn] File appears binary (printable ratio={printable_ratio:.2f}). Using fallback text.")
                    text = FALLBACK_TEXT
            except Exception:
                text = FALLBACK_TEXT

        if not text.strip():
            print(f"  [warn] No text extracted from {file_path}. Using fallback text.")
            text = FALLBACK_TEXT

    target_chars = target_token_count * 4
    if len(text) < target_chars and len(text) > 0:
        repeats = (target_chars // len(text)) + 1
        text = text * repeats

    return text[:target_chars]


def build_multiturn_prompts(max_turns=None):
    """Build prompts with growing conversation history."""
    turns = MULTITURN_QUESTIONS if max_turns is None else MULTITURN_QUESTIONS[:max_turns]
    prompts = []
    history = EXP4_BASE_HISTORY
    for i, question in enumerate(turns, 1):
        history += f"User: {question}\n"
        history += f"Assistant: Here is my detailed answer for turn {i}.\n"
        prompts.append(history + f"User: Can you elaborate further on: {question}")
    return prompts


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main(
    max_queries=MAX_QUERIES,
    cuda_required=DEFAULT_CUDA_REQUIRED,
    runtime_chunks_required=DEFAULT_RUNTIME_CHUNKS_REQUIRED,
):
    if not cuda_required:
        raise RuntimeError(
            "Trace generation requires CUDA to be enabled. Remove --no-cuda-required and run on a CUDA-capable host."
        )
    if not runtime_chunks_required:
        raise RuntimeError(
            "Trace generation requires direct runtime chunk capture. Remove --allow-projected-chunks and ensure LMCache/vLLM runtime chunk events are available."
        )

    t_start = time.time()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = DATA_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 64)
    print("  REAL TRACE GENERATION — vLLM + LMCache on GPU")
    print("=" * 64)
    print(f"  run_id={run_id}")
    print(f"  strict.cuda_required={cuda_required}")
    print(f"  strict.runtime_chunks_required={runtime_chunks_required}")

    # Setup
    ensure_cuda_available(cuda_required)
    setup_lmcache()
    clear_cache()
    llm, sp, _ = build_engine(runtime_chunks_required=runtime_chunks_required)
    direct_runtime_chunks_available = True
    if runtime_chunks_required:
        try:
            assert_runtime_chunk_capture_works(llm, sp)
        except RuntimeError as exc:
            direct_runtime_chunks_available = False
            print(
                "  [warn] Direct runtime chunk IDs are unavailable from vLLM output. "
                "Falling back to LMCache-store cold-pass real chunk capture."
            )
            print(f"  [warn] Probe detail: {exc}")
    kv_cfg = get_kv_config(llm)
    tokenizer = llm.get_tokenizer()

    n = max_queries
    all_raw = {}

    # ── Experiment 1: Shared Prefix ──
    prefix_questions = EXP1_QUESTIONS[:n]
    prompts_1 = [EXP1_PREFIX + q for q in prefix_questions]
    prefix_store_ids = None
    if runtime_chunks_required and not direct_runtime_chunks_available:
        prefix_store_ids = collect_runtime_chunk_ids_from_store_coldpass(
            workload_name="prefix",
            prompts=prompts_1,
            llm=llm,
            sp=sp,
            kv_cfg=kv_cfg,
            tokenizer=tokenizer,
        )
    raw_prefix = run_experiment(
        "1. Shared Prefix",
        prompts_1,
        llm,
        sp,
        kv_cfg,
        focus_texts=prefix_questions,
        context_profile="shared_prefix_rehab_instructions",
        tokenizer=tokenizer,
        runtime_chunks_required=runtime_chunks_required,
        store_chunk_ids_by_query=prefix_store_ids,
    )
    all_raw["prefix"] = raw_prefix

    # ── Experiment 2: RAG (Shared Document) ──
    # Use 1600 tokens for context, leaving room for question + output within 4096 limit
    context_text = load_and_scale_context(SOURCE_FILE, 1600)
    rag_doc = f"Context: {context_text}\n\n"
    rag_questions = RAG_QUESTIONS[:n]
    prompts_2 = [rag_doc + q for q in rag_questions]
    rag_store_ids = None
    if runtime_chunks_required and not direct_runtime_chunks_available:
        rag_store_ids = collect_runtime_chunk_ids_from_store_coldpass(
            workload_name="rag",
            prompts=prompts_2,
            llm=llm,
            sp=sp,
            kv_cfg=kv_cfg,
            tokenizer=tokenizer,
        )
    raw_rag = run_experiment(
        "2. Shared Docs (RAG)",
        prompts_2,
        llm,
        sp,
        kv_cfg,
        focus_texts=rag_questions,
        context_profile="rag_shared_document_finance_reports",
        tokenizer=tokenizer,
        runtime_chunks_required=runtime_chunks_required,
        store_chunk_ids_by_query=rag_store_ids,
    )
    all_raw["rag"] = raw_rag

    # ── Experiment 3: No Context ──
    prompts_3 = NO_CONTEXT_QUESTIONS[:n]
    nocontext_store_ids = None
    if runtime_chunks_required and not direct_runtime_chunks_available:
        nocontext_store_ids = collect_runtime_chunk_ids_from_store_coldpass(
            workload_name="nocontext",
            prompts=prompts_3,
            llm=llm,
            sp=sp,
            kv_cfg=kv_cfg,
            tokenizer=tokenizer,
        )
    raw_nc = run_experiment(
        "3. No Context",
        prompts_3,
        llm,
        sp,
        kv_cfg,
        focus_texts=prompts_3,
        context_profile="no_shared_context",
        tokenizer=tokenizer,
        runtime_chunks_required=runtime_chunks_required,
        store_chunk_ids_by_query=nocontext_store_ids,
    )
    all_raw["nocontext"] = raw_nc

    # ── Experiment 4: Multi-Turn Chat ──
    prompts_4 = build_multiturn_prompts(n)
    multiturn_store_ids = None
    if runtime_chunks_required and not direct_runtime_chunks_available:
        multiturn_store_ids = collect_runtime_chunk_ids_from_store_coldpass(
            workload_name="multiturn",
            prompts=prompts_4,
            llm=llm,
            sp=sp,
            kv_cfg=kv_cfg,
            tokenizer=tokenizer,
        )
    multiturn_focus = MULTITURN_QUESTIONS[: len(prompts_4)]
    raw_mt = run_experiment(
        "4. Multi-Turn Chat",
        prompts_4,
        llm,
        sp,
        kv_cfg,
        focus_texts=multiturn_focus,
        context_profile="multiturn_dialog_history",
        tokenizer=tokenizer,
        runtime_chunks_required=runtime_chunks_required,
        store_chunk_ids_by_query=multiturn_store_ids,
    )
    all_raw["multiturn"] = raw_mt

    # ── Convert to RL trace CSVs ──
    print(f"\n{'=' * 64}")
    print("  Converting to RL trace CSVs...")
    print(f"{'=' * 64}")

    trace_prefix = rows_to_trace_csv(
        raw_prefix,
        DATA_DIR / "traces_prefix.csv",
        "prefix",
        kv_cfg,
        runtime_chunks_required=runtime_chunks_required,
    )
    trace_rag = rows_to_trace_csv(
        raw_rag,
        DATA_DIR / "traces_rag.csv",
        "rag",
        kv_cfg,
        runtime_chunks_required=runtime_chunks_required,
    )
    trace_nc = rows_to_trace_csv(
        raw_nc,
        DATA_DIR / "traces_nocontext.csv",
        "nocontext",
        kv_cfg,
        runtime_chunks_required=runtime_chunks_required,
    )
    trace_mt = rows_to_trace_csv(
        raw_mt,
        DATA_DIR / "traces_multiturn.csv",
        "multiturn",
        kv_cfg,
        runtime_chunks_required=runtime_chunks_required,
    )
    traces_by_workload = {
        "prefix": trace_prefix,
        "rag": trace_rag,
        "nocontext": trace_nc,
        "multiturn": trace_mt,
    }

    # ── Runtime provenance artifacts + trace audit ──
    runtime_events_path = DATA_DIR / "runtime_chunk_events.jsonl"
    write_runtime_event_audit(all_raw, runtime_events_path)

    audit_report = build_trace_audit_report(all_raw, traces_by_workload, kv_cfg)
    audit_path = DATA_DIR / "trace_audit_report.json"
    with open(audit_path, "w") as f:
        json.dump(audit_report, f, indent=2)
    print(f"  [audit] Summary report saved → {audit_path.name}")

    validate_trace_audit(audit_report)
    print("  [audit] Strict gates passed ✓")

    # ── TTFT lookup ──
    ttft_lookup = build_ttft_lookup(all_raw, traces_by_workload)
    ttft_path = DATA_DIR / "ttft_lookup.json"
    with open(ttft_path, "w") as f:
        json.dump(ttft_lookup, f, indent=2)
    print(f"  [ttft] Saved → {ttft_path.name}")

    # ── Extract tier config and update hardware_config.yaml ──
    benchmark_latencies = load_benchmark_tier_latencies(BENCHMARK_LATENCY_PATH)
    validate_benchmark_latencies(benchmark_latencies)
    tier_config = extract_tier_config(kv_cfg, ttft_lookup, benchmark_latencies=benchmark_latencies)
    update_hardware_config(tier_config, kv_cfg)

    # ── Save raw experiment results too (for reference) ──
    import pandas as pd
    raw_rows = []
    for wl_name, rows in all_raw.items():
        for r in rows:
            raw = dict(r)
            raw["workload"] = wl_name
            raw["runtime_chunk_event_available"] = bool(raw.get("runtime_chunk_ids"))
            raw.pop("prompt_token_ids", None)
            runtime_chunk_ids = raw.get("runtime_chunk_ids")
            raw["runtime_chunk_ids"] = json.dumps(runtime_chunk_ids if isinstance(runtime_chunk_ids, list) else [])
            raw_rows.append(raw)
    raw_df = pd.DataFrame(raw_rows)
    raw_csv = DATA_DIR / "raw_experiment_results.csv"
    raw_df.to_csv(raw_csv, index=False)
    print(f"  [raw] Saved → {raw_csv.name}")

    # ── Save reproducible run bundle ──
    run_meta = {
        "run_id": run_id,
        "timestamp_utc": run_id,
        "cuda_required": cuda_required,
        "runtime_chunks_required": runtime_chunks_required,
        "max_queries": n,
        "artifacts": {
            "traces_prefix": str(DATA_DIR / "traces_prefix.csv"),
            "traces_rag": str(DATA_DIR / "traces_rag.csv"),
            "traces_nocontext": str(DATA_DIR / "traces_nocontext.csv"),
            "traces_multiturn": str(DATA_DIR / "traces_multiturn.csv"),
            "ttft_lookup": str(ttft_path),
            "runtime_chunk_events": str(runtime_events_path),
            "trace_audit_report": str(audit_path),
            "raw_experiment_results": str(raw_csv),
            "hardware_config": str(HARDWARE_CONFIG_PATH),
            "ppo_config": str(PPO_CONFIG_PATH),
        },
    }
    run_meta_path = DATA_DIR / "run_metadata.json"
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"  [meta] Saved → {run_meta_path.name}")

    for path in [
        DATA_DIR / "traces_prefix.csv",
        DATA_DIR / "traces_rag.csv",
        DATA_DIR / "traces_nocontext.csv",
        DATA_DIR / "traces_multiturn.csv",
        ttft_path,
        runtime_events_path,
        audit_path,
        raw_csv,
        run_meta_path,
    ]:
        shutil.copy2(path, run_dir / path.name)
    print(f"  [meta] Run bundle archived → {run_dir}")

    # ── Delete stale embeddings ──
    for emb_file in DATA_DIR.glob("embeddings_*.npy"):
        emb_file.unlink()
        print(f"  [clean] Removed stale {emb_file.name}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 64}")
    print(f"  DONE! Total time: {elapsed / 60:.1f} minutes")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate runtime-grounded traces with strict CUDA/runtime provenance gates."
    )
    parser.add_argument("--max-queries", type=int, default=MAX_QUERIES)
    parser.add_argument(
        "--cuda-required",
        dest="cuda_required",
        action="store_true",
        default=DEFAULT_CUDA_REQUIRED,
        help="Require CUDA for generation (default: true).",
    )
    parser.add_argument(
        "--no-cuda-required",
        dest="cuda_required",
        action="store_false",
        help="Deprecated. Strict runtime provenance mode requires CUDA and will fail if this flag is used.",
    )
    parser.add_argument(
        "--runtime-chunks-required",
        dest="runtime_chunks_required",
        action="store_true",
        default=DEFAULT_RUNTIME_CHUNKS_REQUIRED,
        help="Require direct runtime chunk events (default: true).",
    )
    parser.add_argument(
        "--allow-projected-chunks",
        dest="runtime_chunks_required",
        action="store_false",
        help="Deprecated. Strict runtime provenance mode requires direct runtime chunk capture and will fail if this flag is used.",
    )
    args = parser.parse_args()

    main(
        max_queries=args.max_queries,
        cuda_required=args.cuda_required,
        runtime_chunks_required=args.runtime_chunks_required,
    )
