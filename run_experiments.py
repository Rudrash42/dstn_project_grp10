#!/usr/bin/env python3
"""
DSTN Milestone 2 — LMCache KV-Cache Reuse Experiments
=====================================================
Runs four experiments on a local GPU using vLLM + LMCache to measure
the latency and throughput benefits of KV-cache reuse across different
prompt-sharing patterns.

Experiments
-----------
  1. Shared Prefix   – many queries share the same system-prompt prefix
  2. Shared Doc (RAG) – many queries share the same long document context
  3. No Context       – fully independent queries (control / baseline)
  4. Multi-Turn Chat  – incrementally growing conversation history

Requirements
------------
  pip install vllm lmcache pandas matplotlib psutil pyyaml

Hardware
--------
  Tested on RTX 3050 Laptop GPU (4 GB VRAM).
  Change MODEL_NAME / MAX_MODEL_LEN below if you have more VRAM.
"""

import os

import sys
import time
import shutil
import yaml
import psutil
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")          # headless backend – saves to file
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION  (edit these to match your hardware)
# ═══════════════════════════════════════════════════════════════
MODEL_NAME      = "Qwen/Qwen2.5-0.5B-Instruct"  # ~1 GB in fp16
# MODEL_NAME      = "meta-llama/Llama-3.2-1B-Instruct"  # ~2.3 GB in bf16
MAX_MODEL_LEN   = 4096   # vLLM's max_model_len — reduced for 4 GB VRAM
GPU_MEM_UTIL    = 0.80
MAX_NEW_TOKENS  = 20
TEMPERATURE     = 0.0
ENFORCE_EAGER   = True

# Max queries per experiment (None = use all).
# Lower this if you want a quicker test run.
MAX_QUERIES     = 50
MAX_CONTEXT_TOKENS = 2048  # We will test up to this many prompt tokens (must be <= MAX_MODEL_LEN)

# ── Cache tuning knobs ────────────────────────────────────────
# Lower MAX_CPU_CACHE_GB to force evictions so more queries are
# genuinely "cold" (cache miss).  With 2.0 GB the shared prefix
# (~256-token chunk ≈ a few MB) never gets evicted → everything
# looks warm after the first query.  Try 0.05 (50 MB) or even
# 0.01 (10 MB) to provoke real eviction behaviour.
MAX_CPU_CACHE_GB  = 0.05      # L2 CPU cache budget in GB
MAX_DISK_CACHE_GB = 5.0       # L3 disk cache budget in GB
CHUNK_SIZE        = 256       # KV token-chunk granularity

# ── GPU KV-cache size override ────────────────────────────────
# vLLM normally fills ALL remaining VRAM with KV-cache blocks.
# Set this to a small integer to artificially shrink the L1 GPU
# KV cache and force evictions to L2 (CPU) / L3 (Disk).
# Each block holds ~16 tokens of KV data.  Rough guide:
#   256 blocks ≈ 4096 tokens  → moderate eviction pressure
#   512 blocks ≈ 8192 tokens  → mild eviction pressure
#   None       → use all available VRAM (no artificial limit)
# WARNING: Do NOT set below ~200.  vLLM chunked prefill needs
#          enough blocks for at least one full request.
NUM_GPU_BLOCKS_OVERRIDE = 256  # ← tune this to control L1 pressure

BASE_DIR    = Path(__file__).resolve().parent
CACHE_DIR   = BASE_DIR / "lmcache_store"
CONFIG_PATH = BASE_DIR / "lmcache_config.yaml"
RESULTS_CSV = BASE_DIR / "experiment_results_3.csv"
PLOTS_DIR   = BASE_DIR / "plots3"
SOURCE_FILE = BASE_DIR / "data/finance_reports.pdf"  # Put your PDF or TXT path here

# ═══════════════════════════════════════════════════════════════
# 1. LMCache configuration
# ═══════════════════════════════════════════════════════════════

def setup_lmcache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # LMCache v0.3.x config:
    #   - local_disk must be a path STRING (not bool). Set None to disable.
    #   - remote_url is a separate string field (not the disk path).
    cfg = {
        "chunk_size": CHUNK_SIZE,
        "local_cpu": True,
        "max_local_cpu_size": MAX_CPU_CACHE_GB,              # 2 GB RAM (L2) cache
        "local_disk": str(CACHE_DIR) + "/",     # L3 disk path (must be string)
        "max_local_disk_size": MAX_DISK_CACHE_GB,             # 5 GB disk budget
        "remote_url": None,                     # No remote server
        "remote_serde": "naive",
        "save_decode_cache": True,
    }
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f)
    os.environ["LMCACHE_CONFIG_FILE"] = str(CONFIG_PATH)
    print(f"[setup] LMCache config  → {CONFIG_PATH}")
    print(f"[setup] Disk cache dir  → {CACHE_DIR}")
    print(f"[setup] CPU cache limit → {MAX_CPU_CACHE_GB} GB")
    print(f"[setup] Disk cache dir  → {CACHE_DIR}  (limit {MAX_DISK_CACHE_GB} GB)")
    print(f"[setup] Chunk size      → {CHUNK_SIZE} tokens")


def clear_cache():
    """Wipe the on-disk L3 cache so the next experiment starts cold."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("         Cache cleared (cold start)")

# ═══════════════════════════════════════════════════════════════
# 2. Engine helpers
# ═══════════════════════════════════════════════════════════════

def build_engine():
    from vllm import LLM, SamplingParams

    print(f"\n>>> Loading model: {MODEL_NAME}")
    print(f"    max_model_len={MAX_MODEL_LEN}  gpu_mem={GPU_MEM_UTIL}")
    if NUM_GPU_BLOCKS_OVERRIDE is not None:
        print(f"    num_gpu_blocks_override={NUM_GPU_BLOCKS_OVERRIDE}  (≈{NUM_GPU_BLOCKS_OVERRIDE * 16} tokens of L1 KV cache)")

    try:
        import lmcache                                       # noqa: F401
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
        print("    Engine loaded WITH LMCache KV connector")
    except Exception as exc:
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
    return llm, sp

# ═══════════════════════════════════════════════════════════════
# 3. Measurement helpers
# ═══════════════════════════════════════════════════════════════

def cache_size_mb():
    if not CACHE_DIR.exists():
        return 0.0
    return round(
        sum(f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file())
        / (1024 * 1024), 2
    )


def cache_file_count():
    """Number of cache chunk files on disk (each ≈ one KV chunk)."""
    if not CACHE_DIR.exists():
        return 0
    return sum(1 for f in CACHE_DIR.rglob("*") if f.is_file())


def ram_gb():
    return round(psutil.Process(os.getpid()).memory_info().rss / 1024**3, 2)


def gpu_kv_cache_usage(llm):
    """
    Read GPU KV-cache utilisation from the vLLM V1 engine.
    Returns (usage_pct, num_gpu_blocks, total_gpu_blocks) or (None, None, None).
    """
    # ── Method 1: vLLM V1 in-process scheduler (if multiprocessing=0) ──
    try:
        engine_core = llm.llm_engine.engine_core.engine_core
        usage_frac = engine_core.scheduler.kv_cache_manager.usage  # 0.0–1.0
        num_blocks = llm.llm_engine.cache_config.num_gpu_blocks
        pct = round(100 * usage_frac, 1)
        return pct, num_blocks, num_blocks
    except Exception:
        pass
    # ── Method 2: report total GPU blocks from cache config ──
    # In multiprocessing mode we can't access the live scheduler,
    # but we can always read num_gpu_blocks (set during init).
    try:
        num_blocks = llm.llm_engine.cache_config.num_gpu_blocks
        if num_blocks is not None and num_blocks > 0:
            # We don't know live usage, report capacity & None for live %
            return None, num_blocks, num_blocks
    except Exception:
        pass
    # ── Method 3: torch.cuda device-level memory ──
    try:
        free, total = torch.cuda.mem_get_info(0)
        used_mb = round((total - free) / (1024 ** 2), 2)
        total_mb = round(total / (1024 ** 2), 2)
        pct = round(100 * (total - free) / total, 1)
        return pct, used_mb, total_mb
    except Exception:
        return None, None, None


def lmcache_cpu_cache_mb():
    """
    Read LMCache's L2 CPU cache usage in MB via its stats monitor.
    Returns the current CPU cache size in MB, or None.
    """
    try:
        from lmcache.observability import LMCStatsMonitor
        monitor = LMCStatsMonitor.GetOrCreate()
        return round(monitor.local_cache_usage_bytes / (1024 * 1024), 2)
    except Exception:
        return None


def lmcache_disk_cache_mb():
    """
    Read LMCache's L3 disk cache usage in MB via its stats monitor.
    Falls back to scanning the cache directory.
    """
    try:
        from lmcache.observability import LMCStatsMonitor
        monitor = LMCStatsMonitor.GetOrCreate()
        val = monitor.local_storage_usage_bytes
        if val is not None and val > 0:
            return round(val / (1024 * 1024), 2)
    except Exception:
        pass
    return cache_size_mb()  # fallback to directory scan


def run_single(llm, sp, prompt: str):
    """Run one prompt and return (latency_s, prompt_tokens, gen_tokens, text)."""
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], sp)
    t1 = time.perf_counter()
    o = outputs[0]
    return (
        t1 - t0,
        len(o.prompt_token_ids),
        len(o.outputs[0].token_ids),
        o.outputs[0].text,
    )

# ═══════════════════════════════════════════════════════════════
# 4. Experiment runner
# ═══════════════════════════════════════════════════════════════

def run_experiment(name, prompts, llm, sp, clear_before=True, all_cold=False):
    """
    Feed *prompts* one-by-one through the engine, recording timing metrics.

    Parameters
    ----------
    all_cold : bool
        If True every query is labelled "Cold" (use for No-Context baseline).
    """
    print(f"\n{'=' * 64}")
    print(f"  EXPERIMENT  {name}   ({len(prompts)} queries)")
    print(f"{'=' * 64}")

    if clear_before:
        clear_cache()

    rows = []
    for i, prompt in enumerate(prompts):
        # Truncate prompt if it would exceed the model's context window.
        # (vLLM will error rather than silently truncate.)
        # Rough heuristic: 1 token ≈ 3.5 chars for English text.
        max_chars = int(MAX_MODEL_LEN * 3.5) - 100
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars]

        disk_before  = cache_size_mb()
        files_before = cache_file_count()
        ram_before   = ram_gb()
        l2_before    = lmcache_cpu_cache_mb() or 0.0

        latency, ptok, gtok, text = run_single(llm, sp, prompt)

        total_tok  = ptok + gtok
        throughput = total_tok / latency if latency > 0 else 0
        ms_tok     = (latency * 1000) / ptok if ptok > 0 else 0
        disk_after  = cache_size_mb()
        files_after = cache_file_count()
        ram_after   = ram_gb()
        l2_after    = lmcache_cpu_cache_mb() or 0.0
        l3_after    = lmcache_disk_cache_mb()

        # ── Detect cold / warm based on actual cache behaviour ──
        # If new disk-cache files appeared, at least part of the KV
        # was NOT in cache → label as Cold (or Partial if some reuse).
        disk_delta_mb   = round(disk_after - disk_before, 2)
        new_cache_files = files_after - files_before

        if all_cold:
            state = "Cold"
        elif i == 0:
            state = "Cold"          # first query always cold
        elif new_cache_files > 0:
            state = "Partial"       # some new KV written → partial miss
        else:
            state = "Warm"          # no new cache entries → full hit

        # ── GPU KV-cache utilisation ──
        gpu_pct, gpu_detail1, gpu_detail2 = gpu_kv_cache_usage(llm)
        gpu_label = f"{gpu_pct}%" if gpu_pct is not None else f"{gpu_detail1} blks"

        print(
            f"   Q{i+1:>3d} ({state:7s})  "
            f"lat={latency:.3f}s  thr={throughput:>6.0f} tok/s  "
            f"in={ptok:>5d}  out={gtok:>3d}  "
            f"L1(GPU)={gpu_label}  "
            f"L2(CPU)={l2_after:.1f}MB  "
            f"L3(Disk)={l3_after:.1f}MB(Δ{disk_delta_mb:+.1f})"
        )

        rows.append({
            "Experiment":            name,
            "Query_ID":              i + 1,
            "State":                 state,
            "Latency (s)":           round(latency, 4),
            "Throughput (tok/s)":     round(throughput, 2),
            "Cost (ms/token)":       round(ms_tok, 3),
            "Input Tokens":          ptok,
            "Output Tokens":         gtok,
            "L1 GPU KV Usage (%)":   gpu_pct,
            "L2 CPU Cache (MB)":     l2_after,
            "L3 Disk Cache (MB)":    l3_after,
            "Disk Delta (MB)":       disk_delta_mb,
            "New Cache Chunks":      new_cache_files,
            "RAM (GB)":              ram_after,
            "Prompt Preview":        prompt[:80] + "…",
        })

    return rows

# ═══════════════════════════════════════════════════════════════
# 5. TEST DATA
# ═══════════════════════════════════════════════════════════════

# ── Experiment 1: Shared Prefix (physiotherapy persona) ───────

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
    "Now answer the following clinical question. "
)

EXP1_QUESTIONS = [
    "What exercises help with lower back pain?",
    "Is applying ice effective for reducing swelling?",
    "Define correct sitting posture for office workers.",
    "How long should a rotator cuff tear rehabilitation programme last?",
    "What is the recommended rest period after an acute ankle sprain?",
    "Which stretches are most effective for tight hamstrings?",
    "How should a patient progress from non-weight-bearing to full weight-bearing after a tibial fracture?",
    "What is the role of proprioception training after ACL reconstruction?",
    "How many sets and reps are recommended for strengthening the quadriceps post knee surgery?",
    "What are the signs that a patient is overtraining during rehabilitation?",
    "How effective is dry needling for myofascial pain syndrome?",
    "What is the difference between active and passive physiotherapy?",
    "When is it safe to return to sport after a hamstring strain?",
    "How does ultrasound therapy aid soft tissue healing?",
    "What are the best exercises for strengthening the hip abductors?",
    "How should breathing be coordinated during core stability exercises?",
    "What is the McKenzie method and when is it indicated?",
    "How do wearable sensors improve rehabilitation outcomes?",
    "What is the recommended frequency of physiotherapy sessions for chronic neck pain?",
    "How does foam rolling affect muscle recovery?",
    "What are the early mobilisation protocols after total knee replacement?",
    "How should a patient warm up before starting rehabilitation exercises?",
    "What is the evidence for kinesiology taping in shoulder impingement?",
    "How does sleep quality affect musculoskeletal recovery?",
    "What exercises are contraindicated after lumbar discectomy?",
    "How is gait analysis used in rehabilitation planning?",
    "What is the role of hydrotherapy in post-surgical rehabilitation?",
    "How long does it typically take to recover from a grade 2 ligament sprain?",
    "What are the benefits of eccentric training for tendinopathy?",
    "How should rehabilitation differ for elderly patients with hip fractures?",
    "What is the Oswestry Disability Index used for?",
    "How can computer vision detect incorrect squat form?",
    "What are the clinical criteria for diagnosing patellofemoral pain syndrome?",
    "How does chronic pain affect rehabilitation adherence?",
    "What is the recommended load progression for Achilles tendinopathy rehab?",
    "How effective is TENS therapy for post-operative pain management?",
    "What is the difference between isometric and isotonic exercises?",
    "How should rehabilitation be modified for a diabetic patient with peripheral neuropathy?",
    "What are the red flags in low back pain that require immediate referral?",
    "How does obesity affect joint loading during rehabilitation exercises?",
    "What is neuromuscular electrical stimulation and when is it used?",
    "How can a patient self-monitor exercise intensity during home rehabilitation?",
    "What are the stages of tissue healing and how do they guide treatment?",
    "How effective is spinal manipulation for non-specific low back pain?",
    "What is the role of the transverse abdominis in lumbar stability?",
    "How does stress and anxiety impact musculoskeletal pain perception?",
    "What is the minimal detectable change for the Visual Analogue Scale in pain assessment?",
    "How should rehabilitation be adapted for a patient with osteoporosis?",
    "What are the best outcome measures for tracking shoulder rehabilitation progress?",
    "How does dehydration affect muscle performance during exercise?",
    "What is the evidence for Pilates in chronic low back pain rehabilitation?",
    "How long should static stretches be held for optimal flexibility gains?",
    "What are the principles of graded exposure in pain rehabilitation?",
    "How does posture affect cervicogenic headaches?",
    "What is the recommended rehabilitation protocol after a Bankart repair?",
    "How can wearable accelerometers quantify patient activity outside clinic sessions?",
    "What is the difference between a muscle strain and a muscle tear?",
    "How should a patient manage delayed onset muscle soreness?",
    "What exercises help prevent falls in older adults?",
    "How does manual therapy compare to exercise therapy for neck pain?",
    "What is the role of scapular stabilisation exercises in shoulder rehabilitation?",
    "How is range of motion measured and documented in clinical practice?",
    "What are the functional milestones for return to running after a calf tear?",
    "How does vitamin D deficiency affect musculoskeletal health?",
    "What is the clinical significance of muscle imbalance in the lower limb?",
    "How effective is shockwave therapy for plantar fasciitis?",
    "What are the biomechanical risk factors for developing runner's knee?",
    "How should a rehabilitation programme be structured for a sedentary office worker with chronic shoulder pain?",
    "What is the role of balance boards in ankle rehabilitation?",
    "How does fear-avoidance behaviour prolong musculoskeletal recovery?",
    "What is the recommended progression for plantar fasciitis rehabilitation?",
    "How do real-time corrective cues improve exercise performance?",
    "What are the clinical features of frozen shoulder and how is it managed?",
    "How should rehabilitation differ between contact and non-contact sports injuries?",
    "What is the evidence for lumbar support belts in preventing back injury?",
    "How does anti-inflammatory medication interact with tendon healing?",
    "What are the best exercises for strengthening the gluteus medius?",
    "How is functional movement screening used in injury prevention?",
    "What is the role of ergonomics in preventing repetitive strain injuries?",
    "How effective is cold water immersion for athlete recovery?",
    "What are the stages of ACL rehabilitation and their timelines?",
    "How does a herniated disc differ from spinal stenosis in terms of rehabilitation?",
    "What is the evidence for Nordic hamstring curls in injury prevention?",
    "How should rehabilitation goals be set collaboratively with patients?",
    "What is the role of compression garments in managing oedema?",
    "How does cardiovascular fitness affect rehabilitation outcomes?",
    "What exercises are recommended for thoracic spine mobility?",
    "How is clinical reasoning applied when designing a rehabilitation programme?",
    "What is the difference between open and closed kinetic chain exercises?",
    "How do wearable EMG sensors help assess muscle activation patterns?",
    "What is the recommended approach for managing chronic Achilles tendinopathy?",
    "How should a patient with hypermobility syndrome approach strength training?",
    "What are the physiological adaptations expected after 8 weeks of resistance training?",
    "How is pain catastrophising measured and addressed in rehabilitation?",
    "What is the evidence for yoga as a complement to physiotherapy?",
    "How does load management prevent stress fractures in runners?",
    "What are the key principles of the biopsychosocial model in physiotherapy?",
    "How effective is heat therapy compared to cold therapy for muscle spasm?",
    "What are the indications and contraindications for traction therapy?",
    "How should a rehabilitation programme be designed for a post-mastectomy patient with lymphoedema?",
    "What is the minimal clinically important difference for the DASH outcome measure?",
]

EXP1_PROMPTS = [EXP1_PREFIX + q for q in EXP1_QUESTIONS]

# ── Experiment 2: Shared Docs / RAG ──────────────────────────
# A long shared document context prepended to each query.
# We simulate a RAG scenario where the same retrieved document
# is reused across many user questions.

# EXP2_DOCUMENT = (
#     "Tongaat Hulett and Implats ESG Report Summary — "
#     "Tongaat Hulett is a leading agri-processing business focusing on the complementary "
#     "activities of sugar production, property development, and starch production. The "
#     "company operates in South Africa, Mozambique, Zimbabwe, and Botswana, employing "
#     "over 30,000 people at the peak of the sugar milling season. In the 2021 financial "
#     "year, Tongaat Hulett produced approximately 1.1 million tons of sugar. The company "
#     "has committed to reducing energy intensity by 20% by 2025, with specific targets "
#     "for water efficiency improvement. Tongaat Hulett invests in socio-economic "
#     "development (SED) and reported total SED expenditure in 2021 aligned with community "
#     "needs. The Lost Time Injury Frequency Rate (LTIFR) is a critical safety metric "
#     "tracked annually. The company's ESG framework aligns with the UN Sustainable "
#     "Development Goals and operates under ISO 45001 certification. "
#     "Implats (Impala Platinum Holdings Limited) is one of the world's foremost producers "
#     "of platinum group metals (PGMs). The company's operations span South Africa and "
#     "Zimbabwe, with managed operations including Impala Rustenburg, Marula, and Zimplats. "
#     "Implats' ESG framework is built on three pillars focusing on environmental "
#     "stewardship, social responsibility, and governance excellence. The PS3 strategy "
#     "guides sustainability alignment. In 2023, Implats achieved significant safety "
#     "milestones while investing heavily in socio-economic development and community "
#     "projects. The company targets a 30% reduction in carbon emissions by 2030 and has "
#     "invested in renewable energy projects including the 35MW solar PV project at "
#     "Zimplats. Water recycling rates exceeded targets, and the company maintains strict "
#     "environmental compliance across all operations. The GISTM (Global Industry Standard "
#     "on Tailings Management) compliance roadmap is actively being implemented. "
#     "Both companies utilise the six capitals framework (Financial, Manufactured, "
#     "Intellectual, Human, Social/Relationship, Natural) to illustrate value creation "
#     "and regularly engage with stakeholders through structured programmes. "
#     "The double materiality principle used in ESG reporting assesses both inward "
#     "financial materiality and outward impact materiality to provide comprehensive "
#     "sustainability reporting aligned with global standards. "
# )

EXP2_QUESTIONS = [
    "What are the core vision and mission statements of Tongaat Hulett?",
    "Summarize the geographic footprint of Tongaat Hulett's operations across South Africa, Mozambique, Zimbabwe, and Botswana.",
    "What was the total volume of sugar produced by Tongaat Hulett in the 2021 financial year?",
    "How many people were employed by Tongaat Hulett at the peak of the sugar milling season in 2021?",
    "Detail the percentage of Board members at Tongaat Hulett who were non-executive and independent in 2021.",
    "Explain the Manufactured and Financial capitals as described in Tongaat Hulett's business model.",
    "What were the key focus areas for Tongaat Hulett in 2022 to enable high performance and operational excellence?",
    "Summarize the impact of the COVID-19 pandemic on Tongaat Hulett's operations and workforce during 2021.",
    "How much did Tongaat Hulett invest in COVID-19 avoidance, mitigation, and treatment in 2021?",
    "Describe the role and responsibilities of the Social, Ethics, Health and Safety Committee at Tongaat Hulett.",
    "What is Tongaat Hulett's target for energy intensity reduction by the year 2025?",
    "List the market-leading brands for sugar and animal feeds mentioned in the Tongaat Hulett report.",
    "What was the total socio-economic development (SED) expenditure by Tongaat Hulett in 2021?",
    "Detail the change in Tongaat Hulett's scope 1 and scope 2 carbon emissions between 2020 and 2021.",
    "How does Tongaat Hulett define and manage its Intellectual Capital?",
    "What were the primary environmental efficiency investments made by Tongaat Hulett in 2021?",
    "Summarize the findings of the 2021 corporate reputation survey conducted by Tongaat Hulett.",
    "What are the specific targets for water efficiency improvement at Tongaat Hulett by 2025?",
    "Explain the relationship between Tongaat Hulett and its small-scale growers, including the volume of sugarcane supplied.",
    "What was the Lost Time Injury Frequency Rate (LTIFR) for Tongaat Hulett in 2021?",
    "Describe Tongaat Hulett's approach to human rights and child labor in its supply chain.",
    "What were the total volumes of hazardous and non-hazardous waste disposed of by Tongaat Hulett in 2021?",
    "List the third-party certifications held by Tongaat Hulett operations, such as ISO 45001.",
    "How does Tongaat Hulett align its business activities with the UN Sustainable Development Goals?",
    "What was the total revenue generated by Tongaat Hulett for the 2021 financial year?",
    "Explain the significance of the Sugar Industry Masterplan for Tongaat Hulett's South African operations.",
    "Detail the training and development spend for employees at Tongaat Hulett in 2021.",
    "What are the primary climate change risks identified and prioritized by Tongaat Hulett?",
    "Summarize the Success Management programme used for performance management at Tongaat Hulett.",
    "Who provides the independent external assurance for Tongaat Hulett's ESG and Climate Change Report?",
    "What is the stated purpose of Implats according to the 2023 ESG report?",
    "Name the managed operations included in the scope of the Implats 2023 ESG report.",
    "What was the Lost Time Injury Frequency Rate (LTIFR) for the Implats Group in 2023?",
    "Detail the total value distributed by Implats to its stakeholders in the 2023 financial year.",
    "What percentage of the Implats Board of Directors identifies as female?",
    "Summarize the Group CEO's statement on Implats' safety performance and the goal of zero harm.",
    "What are the three pillars of the Implats ESG framework?",
    "How much did Implats invest in socio-economic development and community projects in 2023?",
    "Explain the double materiality principle used in the preparation of Implats' ESG reports.",
    "What are the primary metals produced by Implats and their specific uses in modern technology?",
    "Describe the progress and intended impact of the 35MW solar PV project at Zimplats.",
    "What was the total water recycling and reuse rate achieved by the Implats Group in 2023?",
    "Detail the We Care programme's support for the families of employees who lost their lives in fatal incidents.",
    "What were the key outcomes for Employees in terms of wages and benefits paid by Implats in 2023?",
    "Explain the PS3 strategy and its focus on sustainability alignment at Implats.",
    "What is Implats' specific target for reducing its carbon emissions by 2030?",
    "Describe the progress of the sulphur dioxide (SO2) abatement technology installation at Zimplats.",
    "What were the external ESG ratings received by Implats from agencies like MSCI and S&P Global in 2023?",
    "How does Implats manage air quality and reduce particulate matter emissions across its operations?",
    "Summarize the importance of the Royal Bafokeng Platinum (RBPlat) acquisition for Implats' Western Limb assets.",
    "What is the IRMA audit and when does Implats plan to conduct it at a managed site?",
    "Detail the roles and responsibilities of the Social, Transformation and Remuneration (STR) Committee at Implats.",
    "How does Implats address the issue of gender-based violence (GBV) in its host communities?",
    "What was the total direct and indirect energy consumption for the Implats Group in 2023?",
    "Describe the tailings remining project at Impala Rustenburg and its contribution to the circular economy.",
    "What are the specific occupational health milestones for the South African mining industry that Implats is working toward?",
    "How does Implats ensure business ethics and maintain a zero-tolerance stance on fraud and corruption?",
    "Detail the environmental legal compliance status of Implats' managed operations in 2023.",
    "What are the Tier 1 host communities identified for Impala Rustenburg and Marula mine?",
    "Summarize the Sustaining Livelihoods framework and its four key focus areas at Implats.",
    "Compare the water efficiency targets and recycling performance reported by Tongaat Hulett and Implats.",
    "Contrast the carbon emission reduction strategies of Tongaat Hulett and Implats based on their respective industry contexts.",
    "How do both companies approach stakeholder engagement and what are their primary communication channels?",
    "Analyze the differences in how Tongaat Hulett and Implats report on human capital and workforce diversity.",
    "Compare the governance structures of the two companies, specifically the focus of their board-level ESG committees.",
    "How do both organizations integrate the United Nations Global Compact principles into their respective reporting?",
    "Contrast the socio-economic development (SED) spending and focus areas of Tongaat Hulett and Implats.",
    "Compare the safety performance metrics, such as LTIFR and fatalities, of both companies over the reported periods.",
    "How do Tongaat Hulett and Implats address the physical impacts of climate change, such as droughts or floods, on their operations?",
    "Analyze the role of renewable energy (solar, hydro, etc.) in the decarbonization plans of both companies.",
    "What are the inputs and outcomes of Intellectual Capital as described in both the Tongaat Hulett and Implats ESG reports?",
    "How do both companies manage waste, and what are their respective targets for landfill diversion?",
    "Compare the approach to Double Materiality in the Implats report with the materiality determination process in the Tongaat Hulett report.",
    "How have both companies responded to the COVID-19 pandemic in terms of community support and employee health?",
    "Discuss the use of external assurance providers (BDO, Nexia SAB&T, etc.) in the ESG reporting of both companies.",
    "What were the specific Human Capital challenges, such as restructuring, encountered by Tongaat Hulett in 2021?",
    "Compare the Business Model illustrations and the capitals approach used by both Tongaat Hulett and Implats.",
    "How do both reports address Human Rights and child labor within their respective supply chains?",
    "Contrast the production capacity and revenue of Tongaat Hulett with the production and value distribution of Implats.",
    "Discuss the significance of site-specific versus group-level ESG policies in both companies.",
    "What were the specific reasons for restating prior year data in the Tongaat Hulett 2021 ESG report?",
    "Explain the TCFD Application Table provided in the Tongaat Hulett report and its importance.",
    "Detail the causes of the 20 employee deaths at Tongaat Hulett in 2021, distinguishing between work-related and other causes.",
    "Describe the Success Management values and the self-management practices implemented at Tongaat Hulett.",
    "What are the environmental legal and regulatory compliance procedures for Tongaat Hulett's operational sites?",
    "Summarize the Sustainability Management Framework and the data management systems used by Tongaat Hulett.",
    "Detail the breakdown of sugarcane volumes supplied to Tongaat Hulett mills from different types of farmers.",
    "What was the Total Recordable Injury Frequency Rate (TRIFR) for Tongaat Hulett between 2018 and 2021?",
    "Explain the Material adverse change (MAC) dispute regarding the sale of Tongaat Hulett's starch business.",
    "What are the specific outcomes for Transformation as reported in Tongaat Hulett's 2021 business model?",
    "Describe the eight-stage stakeholder engagement approach used by Implats to ensure participatory management.",
    "What are the specific components and focus areas of the PS3 Strategy at Implats?",
    "Detail the 2023 safety milestones, such as millionaire status shifts, achieved by Implats operations.",
    "Explain the involuntary turnover statistics for the Implats Group in 2023 across its different operating regions.",
    "What were the findings of the independent tailings review board regarding Implats' storage facilities in 2023?",
    "Describe the Gender-based Violence (GBV) response fund and the community partnerships initiated by Implats.",
    "What is the Global Industry Standard on Tailings Management (GISTM) and what is Implats' roadmap for compliance?",
    "Detail the Enterprise and Supplier Development (ESD) initiatives and spend at Implats' South African operations.",
    "How does Implats use woodchips in the concurrent rehabilitation of its tailings dam side slopes?",
    "What was the total amount of Coal, Diesel, and Electricity consumed by the Implats Group in 2023?",
    "Explain the inward-focused financial materiality versus outward-focused impact materiality at Implats.",
    "Describe the We Care programme's specific support for families of deceased employees, including education aid.",
]

# EXP2_PROMPTS = [EXP2_DOCUMENT + " Question: " + q for q in EXP2_QUESTIONS]

# ── Experiment 3: No Context (random, independent queries) ───

EXP3_PROMPTS = [
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
    "What is the highest-grossing film of all time?",
    "Who invented the cotton gin?",
    "What is the main source of vitamin D?",
    "How many strings does a harp typically have?",
    "What is the smallest unit of matter?",
    "Who painted The Creation of Adam?",
    "What is the largest airport in the world by area?",
    "How many bones are in a newborn baby?",
    "What is the longest reigning British monarch?",
    "Who discovered insulin?",
    "What is the main ingredient in miso soup?",
    "How many planets are gas giants in our solar system?",
    "What is the largest freshwater lake in Africa?",
    "Who wrote The Alchemist?",
    "What is the chemical symbol for potassium?",
    "How many players are on a rugby union team?",
    "What is the tallest waterfall in Africa?",
    "Who invented the mechanical clock?",
    "What is the main export of Saudi Arabia?",
    "How many chromosomes does a human gamete contain?",
    "What is the largest desert in Asia?",
    "Who composed Swan Lake?",
    "What is the smallest muscle in the human body?",
    "How many colors are there in the visible spectrum?",
    "What is the largest species of eagle?",
    "Who discovered X-rays?",
    "What is the boiling point of ethanol in Celsius?",
    "How many rings are on the Olympic flag?",
    "What is the longest river in South America?",
    "Who invented the sewing machine?",
    "What is the primary ingredient in pesto?",
    "How many players are on a water polo team?",
    "What is the largest organ in the digestive system?",
    "Who wrote The Old Man and the Sea?",
    "What is the chemical symbol for zinc?",
    "How many bones are in the adult human skull?",
    "What is the highest active volcano in the world?",
    "Who developed the theory of evolution by natural selection?",
    "What is the main ingredient in falafel?",
    "How many faces does an icosahedron have?",
    "What is the largest species of whale?",
    "Who invented the barcode?",
    "What is the smallest planet in our solar system?",
    "How many players are on a cricket team?",
    "What is the largest glacier in the world?",
    "Who wrote The Count of Monte Cristo?",
    "What is the chemical symbol for silver?",
    "How many teeth does a shark typically have in its lifetime?",
    "What is the highest mountain in South America?",
    "Who discovered the law of planetary motion?",
    "What is the primary ingredient in kimchi?",
    "How many symphonies did Beethoven compose?",
    "What is the largest species of turtle?",
    "Who invented the ATM?",
    "What is the boiling point of helium in Celsius?",
    "How many players are on a volleyball team?",
    "What is the largest cave system in the world?",
    "Who wrote The Sound and the Fury?",
    "What is the chemical symbol for lead?",
    "How many ribs does a human typically have?",
    "What is the fastest marine animal?",
    "Who discovered the double helix structure of DNA?",
    "What is the main ingredient in tzatziki?",
    "How many players are on a handball team?",
    "What is the largest peninsula in the world?",
    "Who composed The Nutcracker?",
    "What is the smallest country in Africa?",
    "How many squares are on a chessboard?",
    "What is the largest species of crocodile?",
    "Who invented the microwave oven?",
    "What is the main ingredient in paella?",
    "How many valves does the human heart have?",
    "What is the deepest cave in the world?",
    "Who wrote The Grapes of Wrath?",
    "What is the chemical symbol for platinum?",
    "How many players are on an ice hockey team?",
    "What is the largest species of bat?",
    "Who discovered the proton?",
    "What is the primary ingredient in gua bao?",
    "How many degrees are in an equilateral triangle?",
    "What is the longest canal in the world?",
    "Who composed Clair de Lune?",
    "What is the smallest continent by land area?",
    "How many segments are in an insect's body?",
    "What is the largest species of octopus?",
    "Who invented the helicopter?",
    "What is the main ingredient in ratatouille?",
    "How many players are on a lacrosse team?",
    "What is the highest cliff in the world?",
    "Who wrote Les Misérables?",
    "What is the chemical symbol for magnesium?",
    "How many chambers does a crocodile's heart have?",
    "What is the largest inland sea in the world?",
    "Who discovered electromagnetism?",
    "What is the primary ingredient in ceviche?",
    "How many players are on a field hockey team?",
    "What is the longest tunnel in the world?",
    "Who composed The Four Seasons?",
    "What is the smallest bone in the ear?",
    "How many spikes are on the Statue of Liberty's crown?",
    "What is the largest species of antelope?",
    "Who invented the thermometer?",
    "What is the main ingredient in risotto?",
    "How many players are on a polo team?",
    "What is the highest dam in the world?",
    "Who wrote A Tale of Two Cities?",
    "What is the chemical symbol for chromium?",
    "How many pairs of wings do bees have?",
    "What is the largest species of jellyfish?",
    "Who discovered the structure of the atom?",
    "What is the primary ingredient in sashimi?",
    "How many players are on a curling team?",
    "What is the longest mountain range in the world?",
    "Who composed The Barber of Seville?",
    "What is the smallest planet by mass?",
    "How many bones are in the human hand?",
    "What is the largest species of owl?",
    "Who invented the escalator?",
    "What is the main ingredient in gazpacho?",
    "How many players are on a dodgeball team?",
    "What is the highest plateau in the world?",
    "Who wrote The Sun Also Rises?",
    "What is the chemical symbol for manganese?",
    "How many tentacles does a jellyfish typically have?",
    "What is the largest species of rhinoceros?",
    "Who discovered the photoelectric effect?",
    "What is the primary ingredient in baklava?",
    "How many players are on a synchronized swimming team?",
    "What is the longest strait in the world?",
    "Who composed Rhapsody in Blue?",
    "What is the smallest unit of energy?",
    "How many vertebrae are in the human spine?",
    "Who is Batman?",
    "Recipe for chocolate cake?",
    "Distance to the Moon?",
    "How does a black hole form?",
    "What is the speed of light?",
    "How do airplanes fly?",
    "What causes earthquakes?",
    "How does the immune system work?",
    "What is quantum entanglement?",
    "How is glass made?",
    "Why is the sky blue?",
    "How do vaccines work?",
    "What is the theory of relativity?",
    "How do plants make food?",
    "What is DNA?",
    "How does Wi-Fi work?",
    "What is the tallest mountain on Earth?",
    "How do tides work?",
    "What is the Fibonacci sequence?",
    "How does a combustion engine work?",
    "What is dark matter?",
    "How does electricity work?",
    "What is machine learning?",
    "How do hurricanes form?",
    "What is the Big Bang theory?",
    "How does sound travel?",
    "What is a meme?",
    "How do satellites stay in orbit?",
    "What is the greenhouse effect?",
    "How does the human brain store memories?",
    "What is blockchain?",
    "How does sonar work?",
    "What is CRISPR?",
    "How do noise-canceling headphones work?",
    "What is entropy?",
    "How does a nuclear reactor work?",
    "What is the placebo effect?",
    "How do solar panels work?",
    "What is a supernova?",
    "How does the stock market work?",
    "What is photosynthesis?",
    "How do lithium batteries work?",
    "What is the Turing test?",
    "How does carbon dating work?",
    "What is a neural network?",
    "How does the internet work?",
    "What is tectonic plate movement?",
    "How do antibiotics work?",
    "What is a wormhole?",
    "How does inflation work?",
    "What is the ozone layer?",
    "How does a transformer model work?",
]

# ── Experiment 4: Multi-Turn Chat ────────────────────────────

EXP4_BASE_HISTORY = "User: Hello AI.\nAssistant: Hi there! How can I help you today?\n"

EXP4_QUESTIONS = [
    "Can you explain how the RehabQuest pose tracking works in detail?",
    "What is the calibration procedure used at the start of each session?",
    "How does the T-pose calibration normalise body proportions across different patients?",
    "What are the hardware requirements for real-time pose tracking?",
    "How does the MediaPipe holistic model detect the 33 pose landmarks?",
    "What is the role of cosine similarity in computing joint angles?",
    "How are joint angles computed in three dimensions using vector mathematics?",
    "How is the system validated against the Vicon motion capture gold standard?",
    "What accuracy metrics are used to evaluate pose tracking performance?",
    "How does camera distance affect the accuracy of landmark detection?",
    "What is the minimum GPU specification needed for real-time processing?",
    "How does the system handle occlusion when body parts are partially hidden?",
    "What frame rate is required to achieve clinically acceptable motion tracking?",
    "How are left and right side landmarks differentiated in the holistic model?",
    "What happens if the T-pose calibration is performed incorrectly?",
    "How does the system account for varying patient heights and limb lengths?",
    "Can the pose tracking work with a standard RGB webcam or does it need depth sensors?",
    "How are the 33 landmarks mapped to anatomical joint definitions?",
    "What filtering or smoothing is applied to raw landmark coordinates?",
    "How does the system detect and reject outlier frames during a session?",
    "How is shoulder flexion angle specifically calculated from landmark vectors?",
    "How is knee extension range of motion extracted from the landmark data?",
    "What is the typical latency from movement to on-screen feedback?",
    "How does lighting condition affect landmark detection confidence scores?",
    "What confidence threshold is used to accept or reject a detected landmark?",
    "How does the system track spinal alignment and posture during exercises?",
    "How are exercise repetitions counted from the joint angle time series?",
    "What machine learning model underlies the MediaPipe pose estimator?",
    "How was the MediaPipe model trained and what datasets were used?",
    "Can the system distinguish between correct and compensatory movement patterns?",
    "How is data from multiple sessions stored and compared over time?",
    "What data format is used to export session results for clinician review?",
    "How does the system perform on patients with limb prosthetics or assistive devices?",
    "What are the known failure modes of the cosine similarity angle computation?",
    "How is the world coordinate frame defined relative to the camera?",
    "How does the system handle patients who cannot perform the initial T-pose?",
    "What is the mean absolute error reported against the Vicon gold standard?",
    "How does body-worn clothing affect the accuracy of landmark detection?",
    "Can multiple cameras be used simultaneously to improve tracking accuracy?",
    "How are upper and lower extremity exercises treated differently in the pipeline?",
    "What happens to tracking accuracy when the patient moves out of the camera frame?",
    "How is the skeleton model re-initialised after a tracking loss event?",
    "How are hip joint angles computed and which landmarks are used?",
    "What is the difference between 2D and 3D landmark coordinates in MediaPipe?",
    "How does the system calculate symmetry scores between left and right sides?",
    "What network architecture is used for the pose landmark regression?",
    "How are progress reports generated from accumulated session data?",
    "Can the system operate offline without an internet connection?",
    "How is patient privacy protected when storing video and landmark data?",
    "What are the planned future improvements to the pose tracking pipeline?",
]


# ================= DATA LOADING =================
def load_and_scale_context(file_path, target_token_count):
    text = ""
    file_path = Path(file_path)
    
    # Simple fallback generator if file is missing/empty
    if not file_path.exists():
        print(f"[Warn] {file_path} not found. Generating dummy medical data.")
        base_text = "Physiotherapy involves the holistic approach to prevention, diagnosis, and therapeutic management of pain disorders. "
        text = base_text
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except:
            text = "Error reading file. Using dummy data. "

    # Scale to target length (approx 4 chars per token)
    current_chars = len(text)
    target_chars = target_token_count * 4
    
    if current_chars < target_chars and current_chars > 0:
        repeats = (target_chars // current_chars) + 1
        text = text * repeats
    
    return text[:target_chars]


def build_multiturn_prompts(max_turns=None):
    """
    Build a list of prompts where each successive prompt contains the
    entire conversation history up to that point.
    """
    turns = EXP4_QUESTIONS if max_turns is None else EXP4_QUESTIONS[:max_turns]
    prompts = []
    history = EXP4_BASE_HISTORY
    for i, question in enumerate(turns, 1):
        history += f"User: {question}\n"
        history += f"Assistant: Here is my detailed answer for turn {i}.\n"
        prompts.append(history + f"User: Can you elaborate further on: {question}")
    return prompts

# ═══════════════════════════════════════════════════════════════
# 6. Plotting
# ═══════════════════════════════════════════════════════════════

def generate_plots(df: pd.DataFrame):
    PLOTS_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11})

    experiments = df["Experiment"].unique()

    # ── Plot 1: Cold / Partial / Warm average latency ───────────
    fig, ax = plt.subplots(figsize=(12, 6))
    cold_avg    = df[df["State"] == "Cold"].groupby("Experiment")["Latency (s)"].mean()
    partial_avg = df[df["State"] == "Partial"].groupby("Experiment")["Latency (s)"].mean()
    warm_avg    = df[df["State"] == "Warm"].groupby("Experiment")["Latency (s)"].mean()

    x = np.arange(len(experiments))
    w = 0.25
    bars1 = ax.bar(x - w, [cold_avg.get(e, 0) for e in experiments], w,
                    label="Cold (Cache Miss)", color="#FF6B6B")
    bars2 = ax.bar(x,     [partial_avg.get(e, 0) for e in experiments], w,
                    label="Partial (Partial Hit)", color="#FFD93D")
    bars3 = ax.bar(x + w, [warm_avg.get(e, 0) for e in experiments], w,
                    label="Warm (Full Cache Hit)", color="#4ECDC4")
    ax.set_ylabel("Average Latency (s)")
    ax.set_title("Cold / Partial / Warm Latency by Experiment")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in list(bars1) + list(bars2) + list(bars3):
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "1_cold_vs_warm_latency.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Speedup ratio ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    speedups = []
    for e in experiments:
        c = cold_avg.get(e, 0)
        w_val = warm_avg.get(e, 0)
        speedups.append(c / w_val if w_val > 0 else 1.0)
    colors = ["#FF6B6B" if s < 1.05 else "#4ECDC4" for s in speedups]
    bars = ax.bar(experiments, speedups, color=colors)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No speedup (1x)")
    ax.set_ylabel("Speedup (Cold / Warm)")
    ax.set_title("Cache Speedup by Experiment")
    ax.set_xticklabels(experiments, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, speedups):
        ax.annotate(f"{val:.2f}x",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "2_speedup_ratio.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Per-query latency trace (one subplot per exp) ─
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    axes = axes.flatten()
    palette = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    for idx, exp in enumerate(experiments):
        ax = axes[idx]
        sub = df[df["Experiment"] == exp]
        ax.plot(sub["Query_ID"], sub["Latency (s)"], marker="o",
                markersize=3, color=palette[idx % len(palette)])
        ax.set_title(exp, fontsize=10)
        ax.set_xlabel("Query #")
        ax.set_ylabel("Latency (s)")
        ax.grid(alpha=0.3)
    fig.suptitle("Per-Query Latency Trace", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "3_latency_trace.png", dpi=150)
    plt.close(fig)

    # ── Plot 4: Average throughput per experiment ─────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    thr_avg = df.groupby("Experiment")["Throughput (tok/s)"].mean()
    bars = ax.bar(experiments, [thr_avg.get(e, 0) for e in experiments],
                  color=palette[:len(experiments)])
    ax.set_ylabel("Average Throughput (tok/s)")
    ax.set_title("Average Throughput by Experiment")
    ax.set_xticklabels(experiments, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "4_throughput.png", dpi=150)
    plt.close(fig)

    # ── Plot 5: Multi-turn normalised cost (ms/token) ────────
    mt = df[df["Experiment"].str.contains("Multi-Turn")]
    if not mt.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(mt["Query_ID"], mt["Cost (ms/token)"],
                marker="o", markersize=4, color="#96CEB4")
        ax.set_xlabel("Turn Number")
        ax.set_ylabel("Normalised Cost (ms / input token)")
        ax.set_title("Multi-Turn Chat: Normalised Latency per Input Token")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "5_multiturn_cost.png", dpi=150)
        plt.close(fig)

    # ── Plot 6: All 3 cache levels growth (combined) ─────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    axes = axes.flatten()
    for idx, exp in enumerate(experiments):
        ax = axes[idx]
        sub = df[df["Experiment"] == exp]
        qid = sub["Query_ID"]

        # L1 GPU KV Cache
        if "L1 GPU KV Usage (%)" in sub.columns and sub["L1 GPU KV Usage (%)"].notna().any():
            ax.plot(qid, sub["L1 GPU KV Usage (%)"],
                    marker="s", markersize=3, label="L1 GPU KV (%)",
                    color="#FF6B6B", linewidth=1.5)
        # L2 CPU Cache
        if "L2 CPU Cache (MB)" in sub.columns and sub["L2 CPU Cache (MB)"].notna().any():
            ax2 = ax.twinx()
            ax2.plot(qid, sub["L2 CPU Cache (MB)"],
                     marker="^", markersize=3, label="L2 CPU (MB)",
                     color="#4ECDC4", linewidth=1.5)
            ax2.plot(qid, sub["L3 Disk Cache (MB)"],
                     marker=".", markersize=3, label="L3 Disk (MB)",
                     color="#45B7D1", linewidth=1.5)
            ax2.set_ylabel("Cache Size (MB)", fontsize=8)
            ax2.legend(loc="center right", fontsize=7)
        else:
            ax.plot(qid, sub["L3 Disk Cache (MB)"],
                    marker=".", markersize=3, label="L3 Disk (MB)",
                    color="#45B7D1", linewidth=1.5)

        ax.set_title(exp, fontsize=10)
        ax.set_xlabel("Query #")
        ax.set_ylabel("GPU KV Usage (%)")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Cache Utilisation Across All 3 Levels (L1 GPU / L2 CPU / L3 Disk)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "6_all_cache_levels.png", dpi=150)
    plt.close(fig)

    # ── Plot 7: L1 GPU KV-cache utilisation per query ─────────
    if "L1 GPU KV Usage (%)" in df.columns and df["L1 GPU KV Usage (%)"].notna().any():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
        axes = axes.flatten()
        for idx, exp in enumerate(experiments):
            ax = axes[idx]
            sub = df[df["Experiment"] == exp]
            ax.plot(sub["Query_ID"], sub["L1 GPU KV Usage (%)"],
                    marker="o", markersize=3, color=palette[idx % len(palette)])
            ax.set_title(exp, fontsize=10)
            ax.set_xlabel("Query #")
            ax.set_ylabel("GPU KV Cache Usage (%)")
            ax.set_ylim(-5, 105)
            ax.grid(alpha=0.3)
        fig.suptitle("L1 GPU KV-Cache Utilisation per Query", fontsize=13, y=1.01)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "7_gpu_kv_cache_usage.png", dpi=150)
        plt.close(fig)

    # ── Plot 8: L2 CPU cache growth per experiment ────────────
    if "L2 CPU Cache (MB)" in df.columns and df["L2 CPU Cache (MB)"].notna().any():
        fig, ax = plt.subplots(figsize=(12, 5))
        for idx, exp in enumerate(experiments):
            sub = df[df["Experiment"] == exp]
            ax.plot(sub["Query_ID"], sub["L2 CPU Cache (MB)"],
                    marker=".", markersize=3, label=exp,
                    color=palette[idx % len(palette)])
        ax.set_xlabel("Query #")
        ax.set_ylabel("CPU Cache Size (MB)")
        ax.set_title("L2 CPU Cache (LMCache) Growth During Experiments")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "8_cpu_cache_growth.png", dpi=150)
        plt.close(fig)

    # ── Plot 9: L3 Disk cache growth ──────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, exp in enumerate(experiments):
        sub = df[df["Experiment"] == exp]
        ax.plot(sub["Query_ID"], sub["L3 Disk Cache (MB)"],
                marker=".", markersize=3, label=exp,
                color=palette[idx % len(palette)])
    ax.set_xlabel("Query #")
    ax.set_ylabel("Disk Cache Size (MB)")
    ax.set_title("L3 Disk Cache Growth During Experiments")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "9_disk_cache_growth.png", dpi=150)
    plt.close(fig)

    # ── Plot 10: Per-query cache delta (new KV written) ───────
    if "Disk Delta (MB)" in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
        axes = axes.flatten()
        for idx, exp in enumerate(experiments):
            ax = axes[idx]
            sub = df[df["Experiment"] == exp]
            colors_bar = ["#FF6B6B" if d > 0 else "#4ECDC4"
                          for d in sub["Disk Delta (MB)"]]
            ax.bar(sub["Query_ID"], sub["Disk Delta (MB)"], color=colors_bar)
            ax.set_title(exp, fontsize=10)
            ax.set_xlabel("Query #")
            ax.set_ylabel("New KV Written to Disk (MB)")
            ax.grid(alpha=0.3)
        fig.suptitle("Per-Query Cache Delta (red = cache miss / new KV)",
                     fontsize=13, y=1.01)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "10_cache_delta.png", dpi=150)
        plt.close(fig)

    print(f"\n[plots] Plots saved → {PLOTS_DIR}/")

# ═══════════════════════════════════════════════════════════════
# 7. Summary statistics
# ═══════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)

    for exp in df["Experiment"].unique():
        sub = df[df["Experiment"] == exp]
        cold    = sub[sub["State"] == "Cold"]
        partial = sub[sub["State"] == "Partial"]
        warm    = sub[sub["State"] == "Warm"]

        cold_lat    = cold["Latency (s)"].mean() if len(cold) else 0
        partial_lat = partial["Latency (s)"].mean() if len(partial) else 0
        warm_lat    = warm["Latency (s)"].mean() if len(warm) else 0
        speedup     = cold_lat / warm_lat if warm_lat > 0 else 1.0

        print(f"\n  {exp}")
        print(f"    Queries total     : {len(sub)}")
        print(f"    Cold  (miss)      : {len(cold):>3d}  avg lat {cold_lat:.4f}s")
        print(f"    Partial (partial) : {len(partial):>3d}  avg lat {partial_lat:.4f}s")
        print(f"    Warm  (hit)       : {len(warm):>3d}  avg lat {warm_lat:.4f}s")
        print(f"    Speedup (C/W)     : {speedup:.2f}x")
        print(f"    Avg Throughput    : {sub['Throughput (tok/s)'].mean():.1f} tok/s")
        print(f"    Final disk cache  : {sub['L3 Disk Cache (MB)'].iloc[-1]:.1f} MB")
        if "L1 GPU KV Usage (%)" in sub.columns and sub["L1 GPU KV Usage (%)"].notna().any():
            print(f"    Avg L1 GPU KV use : {sub['L1 GPU KV Usage (%)'].mean():.1f}%")
        if "L2 CPU Cache (MB)" in sub.columns and sub["L2 CPU Cache (MB)"].notna().any():
            print(f"    Final L2 CPU cache: {sub['L2 CPU Cache (MB)'].iloc[-1]:.1f} MB")


# ═══════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    # ── Setup ──
    setup_lmcache()
    llm, sp = build_engine()

    all_rows = []
    n = MAX_QUERIES  # shorthand

    # ── Experiment 1: Shared Prefix ──
    prompts_1 = EXP1_PROMPTS[:n] if n else EXP1_PROMPTS
    all_rows.extend(
        run_experiment("1. Shared Prefix", prompts_1, llm, sp)
    )

    # ── Experiment 2: Shared Docs (RAG) ──
    context_text = load_and_scale_context(SOURCE_FILE, MAX_CONTEXT_TOKENS)
    EXP2_DOCUMENT = f"Context: {context_text}\n\n"
    EXP2_PROMPTS = [EXP2_DOCUMENT + q for q in EXP2_QUESTIONS]
    prompts_2 = EXP2_PROMPTS[:n] if n else EXP2_PROMPTS
    all_rows.extend(
        run_experiment("2. Shared Docs (RAG)", prompts_2, llm, sp)
    )

    # ── Experiment 3: No Context (Random) ──
    prompts_3 = EXP3_PROMPTS[:n] if n else EXP3_PROMPTS
    all_rows.extend(
        run_experiment("3. No Context (Random)", prompts_3, llm, sp, all_cold=True)
    )

    # ── Experiment 4: Multi-Turn Chat ──
    prompts_4 = build_multiturn_prompts(n)
    all_rows.extend(
        run_experiment("4. Multi-Turn Chat", prompts_4, llm, sp)
    )

    # ── Save results ──
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[save] Results CSV → {RESULTS_CSV}")

    # ── Plots & summary ──
    generate_plots(df)
    print_summary(df)

    elapsed = time.time() - t_start
    print(f"\n>>> Total wall time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
