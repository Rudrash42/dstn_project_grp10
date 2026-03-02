import os
import time
import yaml
import shutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from vllm import LLM, SamplingParams

# ================= CONFIGURATION =================
# 1. Hardware & Model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" #-GPTQ-Int4"  # A smaller model to fit in 3050's VRAM
GPU_MEM_UTIL = 0.80 
# MAX_MODEL_LEN = 32768  # Enable long context support
MAX_MODEL_LEN = 16384  # Start with 16K max context to ensure it fits in memory, we will test up to 16K tokens

# 2. Test Parameters
# We will test these specific context lengths to find the break-even point
CONTEXT_LENGTHS = [1024, 2048, 4096, 6000, 7500, 8192, 16384] 
# CONTEXT_LENGTHS = [1024, 4096, 8192, 16384, 24000] 
SOURCE_FILE = "data/physiotherapy_wiki.pdf"  # Put your PDF or TXT path here

# 3. LMCache Paths
CACHE_DIR = Path("lmcache_store")
CONFIG_PATH = Path("lmcache_config.yaml")
RESULTS_CSV = Path("stress_test_results.csv")

# ================= DATA LOADING =================
def load_and_scale_context(file_path, target_token_count):
    """
    Reads a file and repeats it to approximate the target token count.
    Rough estimate: 1 token ~= 4 characters.
    """
    text = ""
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    else:
        # Fallback for TXT or if file doesn't exist
        if not file_path.exists():
            print(f"[Warn] {file_path} not found. Using synthetic dummy data.")
            text = "RehabQuest is a physiotherapy AI. " * 100
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

    # Scale to target length
    current_chars = len(text)
    target_chars = target_token_count * 4
    
    if current_chars < target_chars:
        repeats = (target_chars // current_chars) + 1
        text = text * repeats
    
    # Trim to exact target (approx)
    return text[:target_chars]

# ================= SETUP =================
def setup_lmcache():
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2GB RAM Cache, 10GB Disk Cache (Tiers)
    cfg = {
        "chunk_size": 256,
        "local_cpu": True,
        "max_local_cpu_size": 2.0, 
        "local_disk": str(CACHE_DIR.resolve()) + "/",
        "max_local_disk_size": 10.0, 
        "remote_url": None,
        "remote_serde": "naive",
        "save_decode_cache": True,
    }
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f)
    os.environ["LMCACHE_CONFIG_FILE"] = str(CONFIG_PATH)

def get_engine():
    try:
        import lmcache
        return LLM(
            model=MODEL_NAME,
            kv_transfer_config={"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"},
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_model_len=MAX_MODEL_LEN,
            disable_log_stats=True,
            enforce_eager=True # Crucial for 3050 to manage memory manually
        )
    except Exception as e:
        print(f"LMCache Init failed: {e}")
        return None

# ================= RUNNER =================
def run_stress_test():
    setup_lmcache()
    llm = get_engine()
    sp = SamplingParams(temperature=0, max_tokens=10) # We only care about TTFT (Prefill)
    
    results = []

    print(f"\n{'='*60}")
    print(f"STARTING STRESS TEST: {CONTEXT_LENGTHS} tokens")
    print(f"{'='*60}")

    for target_len in CONTEXT_LENGTHS:
        print(f"\n>>> Testing Context Length: {target_len} tokens")
        
        # 1. Prepare Data
        context_text = load_and_scale_context(SOURCE_FILE, target_len)
        prompt = f"{context_text}\n\nUser: Summarize the key medical protocols above.\nAssistant:"
        
        # 2. COLD RUN (Compute Bound)
        # Clear cache to force re-computation
        if CACHE_DIR.exists(): shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir()
        
        start = time.perf_counter()
        llm.generate([prompt], sp)
        cold_latency = time.perf_counter() - start
        
        # 3. WARM RUN (I/O Bound - Fetch from Disk/RAM)
        # We run the exact same prompt again. It should hit the cache.
        start = time.perf_counter()
        llm.generate([prompt], sp)
        warm_latency = time.perf_counter() - start
        
        speedup = cold_latency / warm_latency if warm_latency > 0 else 0
        
        print(f"    [Result] Cold: {cold_latency:.2f}s | Warm: {warm_latency:.2f}s | Speedup: {speedup:.2f}x")
        
        results.append({
            "Context_Tokens": target_len,
            "Cold_Latency": cold_latency,
            "Warm_Latency": warm_latency,
            "Speedup": speedup
        })

    # Save Results
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    
    # Plotting the Break-Even Graph
    plt.figure(figsize=(10, 6))
    plt.plot(df["Context_Tokens"], df["Cold_Latency"], label="Cold (Recompute)", marker="o", color="red")
    plt.plot(df["Context_Tokens"], df["Warm_Latency"], label="Warm (Cached)", marker="o", color="green")
    plt.xlabel("Context Length (Tokens)")
    plt.ylabel("Time to First Token (Seconds)")
    plt.title("LMCache Break-Even Analysis (RTX 3050)")
    plt.legend()
    plt.grid(True)
    plt.savefig("break_even_plot.png")
    print("\nTest Complete. Plot saved to break_even_plot.png")

if __name__ == "__main__":
    run_stress_test()