#!/usr/bin/env python3
"""
DSTN Milestone 2 — Caching Strategy Comparison
================================================
Compares three caching configurations using vLLM:
  1. No Caching       – plain vLLM, no KV-cache reuse across requests
  2. GPU VRAM Only    – vLLM's built-in GPU KV-cache (no LMCache)
  3. LMCache (Full)   – LMCache with L1 GPU + L2 CPU + L3 Disk

Runs the same set of prompts (Shared Prefix scenario from run_experiments.py)
under each configuration and plots:
  - Time to First Token (TTFT)
  - End-to-end Latency
  - Throughput (tokens/s)
  - Normalised Cost (ms per input token)
  - Cumulative Throughput over queries
  - Latency Distribution (box plot)

Hardware
--------
  Tested on RTX 3050 Laptop GPU (4 GB VRAM).
  Matches configuration from run_experiments.py.
"""

import os
import sys
import time
import shutil
import yaml
import psutil
import gc
import pandas as pd
import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION  (same as run_experiments.py)
# ═══════════════════════════════════════════════════════════════
MODEL_NAME      = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_MODEL_LEN   = 4096
GPU_MEM_UTIL    = 0.80
MAX_NEW_TOKENS  = 20
TEMPERATURE     = 0.0
ENFORCE_EAGER   = True
MAX_QUERIES     = 50
NUM_GPU_BLOCKS_OVERRIDE = 256

# LMCache settings (same as run_experiments.py)
MAX_CPU_CACHE_GB  = 0.05
MAX_DISK_CACHE_GB = 5.0
CHUNK_SIZE        = 256

BASE_DIR    = Path(__file__).resolve().parent
CACHE_DIR   = BASE_DIR / "lmcache_store"
CONFIG_PATH = BASE_DIR / "lmcache_config.yaml"
RESULTS_CSV = BASE_DIR / "caching_comparison_results.csv"
PLOTS_DIR   = BASE_DIR / "plots_caching_comparison"

# ═══════════════════════════════════════════════════════════════
# PROMPTS  (same Shared Prefix scenario from run_experiments.py)
# ═══════════════════════════════════════════════════════════════

SHARED_PREFIX = (
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

QUESTIONS = [
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
]

PROMPTS = [SHARED_PREFIX + q for q in QUESTIONS]

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def clear_lmcache_store():
    """Wipe the LMCache on-disk store for a cold start."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("   [cache] Disk cache cleared")


def write_lmcache_config():
    """Write the LMCache YAML config file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {
        "chunk_size": CHUNK_SIZE,
        "local_cpu": True,
        "max_local_cpu_size": MAX_CPU_CACHE_GB,
        "local_disk": str(CACHE_DIR) + "/",
        "max_local_disk_size": MAX_DISK_CACHE_GB,
        "remote_url": None,
        "remote_serde": "naive",
        "save_decode_cache": True,
    }
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f)
    os.environ["LMCACHE_CONFIG_FILE"] = str(CONFIG_PATH)


def destroy_engine(llm):
    """Best-effort teardown of the vLLM engine to free GPU memory."""
    # 1. Explicitly close the engine (shuts down engine-core subprocesses)
    try:
        if hasattr(llm, 'llm_engine'):
            engine = llm.llm_engine
            if hasattr(engine, 'close'):
                engine.close()
            elif hasattr(engine, 'shutdown'):
                engine.shutdown()
    except Exception as e:
        print(f"   [cleanup] Engine shutdown note: {e}")

    try:
        del llm
    except Exception:
        pass

    # 2. Kill any remaining child processes that may still hold GPU memory
    try:
        current = psutil.Process()
        children = current.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        psutil.wait_procs(children, timeout=10)
    except Exception:
        pass

    # 3. Aggressive Python + CUDA cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    gc.collect()
    time.sleep(5)


def run_single(llm, sp, prompt: str):
    """
    Run one prompt and return (latency_s, ttft_s, prompt_tokens, gen_tokens, text).
    TTFT is approximated as the time until the first output token is generated.
    For batch-size=1 with generate(), we measure total latency and estimate TTFT
    as latency * (prompt_tokens / (prompt_tokens + gen_tokens)).
    """
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], sp)
    t1 = time.perf_counter()
    o = outputs[0]
    latency = t1 - t0
    ptok = len(o.prompt_token_ids)
    gtok = len(o.outputs[0].token_ids)
    # Estimate TTFT: the prefill phase processes all prompt tokens before
    # generating the first output token. Approximate as:
    #   TTFT ≈ latency × (prompt_tokens / (prompt_tokens + gen_tokens))
    if ptok + gtok > 0:
        ttft = latency * (ptok / (ptok + gtok))
    else:
        ttft = latency
    return latency, ttft, ptok, gtok, o.outputs[0].text


def gpu_kv_cache_usage(llm):
    """Read GPU KV-cache utilisation from the vLLM engine."""
    try:
        engine_core = llm.llm_engine.engine_core.engine_core
        usage_frac = engine_core.scheduler.kv_cache_manager.usage
        num_blocks = llm.llm_engine.cache_config.num_gpu_blocks
        return round(100 * usage_frac, 1), num_blocks
    except Exception:
        pass
    try:
        num_blocks = llm.llm_engine.cache_config.num_gpu_blocks
        return None, num_blocks
    except Exception:
        return None, None


# ═══════════════════════════════════════════════════════════════
# ENGINE BUILDERS (one per caching strategy)
# ═══════════════════════════════════════════════════════════════

def build_engine_no_cache():
    """
    Plain vLLM — no LMCache, no KV reuse across requests.
    We unset the LMCache env var and don't pass kv_transfer_config.
    """
    os.environ.pop("LMCACHE_CONFIG_FILE", None)
    from vllm import LLM, SamplingParams

    print("\n>>> Building engine: NO CACHING")
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


def build_engine_gpu_only():
    """
    vLLM with GPU KV-cache (built-in) but without LMCache.
    The KV cache lives only in GPU VRAM — no CPU/disk spill.
    This is vLLM's default behaviour.
    """
    os.environ.pop("LMCACHE_CONFIG_FILE", None)
    from vllm import LLM, SamplingParams

    print("\n>>> Building engine: GPU VRAM CACHE ONLY")
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


def build_engine_lmcache():
    """
    vLLM + LMCache — full 3-level cache hierarchy:
      L1: GPU VRAM   L2: CPU RAM   L3: Disk
    """
    write_lmcache_config()
    from vllm import LLM, SamplingParams

    print("\n>>> Building engine: LMCACHE (GPU + CPU + Disk)")
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
    sp = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS)
    return llm, sp


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════

def run_experiment(strategy_name, llm, sp, prompts):
    """Run all prompts sequentially and collect per-query metrics."""
    print(f"\n{'=' * 64}")
    print(f"  STRATEGY: {strategy_name}   ({len(prompts)} queries)")
    print(f"{'=' * 64}")

    rows = []
    for i, prompt in enumerate(prompts):
        # Truncate if needed
        max_chars = int(MAX_MODEL_LEN * 3.5) - 100
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars]

        latency, ttft, ptok, gtok, text = run_single(llm, sp, prompt)

        total_tok  = ptok + gtok
        throughput = total_tok / latency if latency > 0 else 0
        ms_tok     = (latency * 1000) / ptok if ptok > 0 else 0
        gen_speed  = gtok / latency if latency > 0 else 0

        gpu_pct, gpu_blocks = gpu_kv_cache_usage(llm)

        print(
            f"   Q{i+1:>3d}  "
            f"lat={latency:.3f}s  ttft={ttft:.3f}s  "
            f"thr={throughput:>6.0f} tok/s  "
            f"in={ptok:>5d}  out={gtok:>3d}  "
            f"GPU KV={gpu_pct}%"
        )

        rows.append({
            "Strategy":                strategy_name,
            "Query_ID":                i + 1,
            "Latency (s)":             round(latency, 4),
            "TTFT (s)":                round(ttft, 4),
            "Throughput (tok/s)":       round(throughput, 2),
            "Generation Speed (tok/s)": round(gen_speed, 2),
            "Cost (ms/token)":         round(ms_tok, 3),
            "Input Tokens":            ptok,
            "Output Tokens":           gtok,
            "GPU KV Usage (%)":        gpu_pct,
            "GPU Blocks":              gpu_blocks,
            "Prompt Preview":          prompt[:80] + "…",
        })

    return rows


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

PALETTE = {
    "No Caching":        "#FF6B6B",
    "GPU VRAM Only":     "#FFD93D",
    "LMCache (Full)":    "#4ECDC4",
}


def _bar_annotate(ax, bars, fmt="{:.3f}"):
    """Add value labels on top of bar chart bars."""
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(fmt.format(h),
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)


def generate_plots(df: pd.DataFrame):
    PLOTS_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11})

    strategies = list(df["Strategy"].unique())
    colors = [PALETTE.get(s, "#999999") for s in strategies]
    x = np.arange(len(strategies))

    # ── Plot 1: Average Time to First Token (TTFT) ───────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ttft_means = [df[df["Strategy"] == s]["TTFT (s)"].mean() for s in strategies]
    bars = ax.bar(x, ttft_means, color=colors, width=0.5)
    ax.set_ylabel("Average TTFT (s)")
    ax.set_title("Time to First Token (TTFT) by Caching Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.grid(axis="y", alpha=0.3)
    _bar_annotate(ax, bars)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "1_ttft_comparison.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Average End-to-End Latency ────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    lat_means = [df[df["Strategy"] == s]["Latency (s)"].mean() for s in strategies]
    bars = ax.bar(x, lat_means, color=colors, width=0.5)
    ax.set_ylabel("Average Latency (s)")
    ax.set_title("End-to-End Latency by Caching Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.grid(axis="y", alpha=0.3)
    _bar_annotate(ax, bars)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "2_latency_comparison.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Average Throughput ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    thr_means = [df[df["Strategy"] == s]["Throughput (tok/s)"].mean() for s in strategies]
    bars = ax.bar(x, thr_means, color=colors, width=0.5)
    ax.set_ylabel("Average Throughput (tok/s)")
    ax.set_title("Throughput by Caching Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.grid(axis="y", alpha=0.3)
    _bar_annotate(ax, bars, fmt="{:.0f}")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "3_throughput_comparison.png", dpi=150)
    plt.close(fig)

    # ── Plot 4: Per-Query Latency Trace ───────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    for s in strategies:
        sub = df[df["Strategy"] == s]
        ax.plot(sub["Query_ID"], sub["Latency (s)"],
                marker="o", markersize=3, label=s,
                color=PALETTE.get(s, "#999999"), linewidth=1.5)
    ax.set_xlabel("Query #")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Per-Query Latency Trace")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "4_latency_trace.png", dpi=150)
    plt.close(fig)

    # ── Plot 5: Per-Query TTFT Trace ──────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    for s in strategies:
        sub = df[df["Strategy"] == s]
        ax.plot(sub["Query_ID"], sub["TTFT (s)"],
                marker="s", markersize=3, label=s,
                color=PALETTE.get(s, "#999999"), linewidth=1.5)
    ax.set_xlabel("Query #")
    ax.set_ylabel("TTFT (s)")
    ax.set_title("Per-Query Time to First Token")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "5_ttft_trace.png", dpi=150)
    plt.close(fig)

    # ── Plot 6: Latency Distribution (Box Plot) ──────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    data_box = [df[df["Strategy"] == s]["Latency (s)"].values for s in strategies]
    bp = ax.boxplot(data_box, labels=strategies, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency Distribution by Caching Strategy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "6_latency_boxplot.png", dpi=150)
    plt.close(fig)

    # ── Plot 7: TTFT Distribution (Box Plot) ─────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    data_ttft = [df[df["Strategy"] == s]["TTFT (s)"].values for s in strategies]
    bp = ax.boxplot(data_ttft, labels=strategies, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("TTFT (s)")
    ax.set_title("TTFT Distribution by Caching Strategy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "7_ttft_boxplot.png", dpi=150)
    plt.close(fig)

    # ── Plot 8: Normalised Cost (ms per input token) ─────────
    fig, ax = plt.subplots(figsize=(10, 6))
    cost_means = [df[df["Strategy"] == s]["Cost (ms/token)"].mean() for s in strategies]
    bars = ax.bar(x, cost_means, color=colors, width=0.5)
    ax.set_ylabel("Cost (ms / input token)")
    ax.set_title("Normalised Latency Cost per Input Token")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.grid(axis="y", alpha=0.3)
    _bar_annotate(ax, bars)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "8_cost_per_token.png", dpi=150)
    plt.close(fig)

    # ── Plot 9: Cumulative Throughput ─────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    for s in strategies:
        sub = df[df["Strategy"] == s].sort_values("Query_ID")
        cum_tokens = (sub["Input Tokens"] + sub["Output Tokens"]).cumsum()
        cum_time = sub["Latency (s)"].cumsum()
        cum_throughput = cum_tokens / cum_time
        ax.plot(sub["Query_ID"], cum_throughput,
                marker=".", markersize=3, label=s,
                color=PALETTE.get(s, "#999999"), linewidth=1.5)
    ax.set_xlabel("Query #")
    ax.set_ylabel("Cumulative Throughput (tok/s)")
    ax.set_title("Cumulative Throughput Over Queries")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "9_cumulative_throughput.png", dpi=150)
    plt.close(fig)

    # ── Plot 10: Generation Speed (output tok/s) ─────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    gen_means = [df[df["Strategy"] == s]["Generation Speed (tok/s)"].mean() for s in strategies]
    bars = ax.bar(x, gen_means, color=colors, width=0.5)
    ax.set_ylabel("Generation Speed (output tok/s)")
    ax.set_title("Decoding Speed by Caching Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.grid(axis="y", alpha=0.3)
    _bar_annotate(ax, bars, fmt="{:.1f}")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "10_generation_speed.png", dpi=150)
    plt.close(fig)

    # ── Plot 11: Speedup relative to No Caching ──────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    baseline_lat = df[df["Strategy"] == "No Caching"]["Latency (s)"].mean()
    speedups = []
    for s in strategies:
        s_lat = df[df["Strategy"] == s]["Latency (s)"].mean()
        speedups.append(baseline_lat / s_lat if s_lat > 0 else 1.0)
    bar_colors = ["#FF6B6B" if sp <= 1.05 else "#4ECDC4" for sp in speedups]
    bars = ax.bar(x, speedups, color=bar_colors, width=0.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline (1x)")
    ax.set_ylabel("Speedup vs No Caching")
    ax.set_title("Latency Speedup Relative to No Caching")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, speedups):
        ax.annotate(f"{val:.2f}x",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "11_speedup.png", dpi=150)
    plt.close(fig)

    # ── Plot 12: Summary Dashboard (2x2) ─────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: TTFT
    ax = axes[0, 0]
    bars = ax.bar(x, ttft_means, color=colors, width=0.5)
    ax.set_ylabel("TTFT (s)")
    ax.set_title("Avg Time to First Token")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _bar_annotate(ax, bars)

    # Top-right: Latency
    ax = axes[0, 1]
    bars = ax.bar(x, lat_means, color=colors, width=0.5)
    ax.set_ylabel("Latency (s)")
    ax.set_title("Avg End-to-End Latency")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _bar_annotate(ax, bars)

    # Bottom-left: Throughput
    ax = axes[1, 0]
    bars = ax.bar(x, thr_means, color=colors, width=0.5)
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Avg Throughput")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _bar_annotate(ax, bars, fmt="{:.0f}")

    # Bottom-right: Speedup
    ax = axes[1, 1]
    bars = ax.bar(x, speedups, color=bar_colors, width=0.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs No Caching")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, speedups):
        ax.annotate(f"{val:.2f}x",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("Caching Strategy Comparison — Summary Dashboard", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "12_summary_dashboard.png", dpi=150)
    plt.close(fig)

    print(f"\n[plots] All plots saved → {PLOTS_DIR}/")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 64)
    print("  SUMMARY — CACHING STRATEGY COMPARISON")
    print("=" * 64)

    baseline_lat = df[df["Strategy"] == "No Caching"]["Latency (s)"].mean()

    for s in df["Strategy"].unique():
        sub = df[df["Strategy"] == s]
        avg_lat  = sub["Latency (s)"].mean()
        avg_ttft = sub["TTFT (s)"].mean()
        avg_thr  = sub["Throughput (tok/s)"].mean()
        avg_cost = sub["Cost (ms/token)"].mean()
        speedup  = baseline_lat / avg_lat if avg_lat > 0 else 1.0

        print(f"\n  {s}")
        print(f"    Queries            : {len(sub)}")
        print(f"    Avg Latency        : {avg_lat:.4f} s")
        print(f"    Avg TTFT           : {avg_ttft:.4f} s")
        print(f"    Avg Throughput     : {avg_thr:.1f} tok/s")
        print(f"    Avg Cost           : {avg_cost:.3f} ms/token")
        print(f"    Speedup vs No-Cache: {speedup:.2f}x")
        print(f"    Median Latency     : {sub['Latency (s)'].median():.4f} s")
        print(f"    P95 Latency        : {sub['Latency (s)'].quantile(0.95):.4f} s")
        print(f"    Min / Max Latency  : {sub['Latency (s)'].min():.4f} / {sub['Latency (s)'].max():.4f} s")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    all_rows = []
    n = MAX_QUERIES
    prompts = PROMPTS[:n] if n else PROMPTS

    # ----------------------------------------------------------------
    # Strategy 1: No Caching
    # ----------------------------------------------------------------
    # For "no caching", we run each prompt independently.
    # vLLM still uses its internal GPU KV cache within a single request,
    # but nothing is reused across requests — each request starts fresh.
    # We clear LMCache disk store and don't load the LMCache connector.
    clear_lmcache_store()
    llm, sp = build_engine_no_cache()
    all_rows.extend(run_experiment("No Caching", llm, sp, prompts))
    destroy_engine(llm)

    # ----------------------------------------------------------------
    # Strategy 2: GPU VRAM Only
    # ----------------------------------------------------------------
    # vLLM's built-in automatic prefix caching keeps KV blocks in GPU
    # VRAM when the same prefix is seen again. No LMCache connector.
    clear_lmcache_store()
    llm, sp = build_engine_gpu_only()
    all_rows.extend(run_experiment("GPU VRAM Only", llm, sp, prompts))
    destroy_engine(llm)

    # ----------------------------------------------------------------
    # Strategy 3: LMCache (Full — GPU + CPU + Disk)
    # ----------------------------------------------------------------
    clear_lmcache_store()
    llm, sp = build_engine_lmcache()
    all_rows.extend(run_experiment("LMCache (Full)", llm, sp, prompts))
    destroy_engine(llm)

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
