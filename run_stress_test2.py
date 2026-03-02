import os
import time
import yaml
import shutil
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from vllm import LLM, SamplingParams

# ================= CONFIGURATION =================
# 1. Hardware & Model
# WE MUST USE THE 0.5B MODEL. 
# The 1.5B model takes ~3GB VRAM, leaving 0 space for the cache.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# 2. Memory Settings (CRITICAL FOR 4GB GPU)
# We set this to 0.70. This gives vLLM ~2.8GB.
# The model takes ~1GB. This leaves ~1.8GB for the Cache.
GPU_MEM_UTIL = 0.80 
MAX_MODEL_LEN = 32768  # Enable long context support. We will test up to 16K tokens, but this ensures we can load the model.  

# 3. Test Parameters
# We test smaller contexts to fit in the 0.5B model's cache
CONTEXT_LENGTHS = [1024, 2048, 4096, 6000, 7500, 8192, 16384, 24000] 
SOURCE_FILE = "data/physiotherapy_wiki.pdf"  # Put your PDF or TXT path here

# 4. LMCache Paths
CACHE_DIR = Path("lmcache_store")
CONFIG_PATH = Path("lmcache_config.yaml")
RESULTS_CSV = Path("stress_test_results2.csv")

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

# ================= SETUP =================
def setup_lmcache():
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1.5GB Disk Cache
    cfg = {
        "chunk_size": 256,
        "local_cpu": True,
        "max_local_cpu_size": 1.0,  # 1GB RAM
        "local_disk": str(CACHE_DIR.resolve()) + "/",
        "max_local_disk_size": 3.0, # 3GB Disk
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
            enforce_eager=True, # Mandatory for 3050
            tensor_parallel_size=1
        )
    except Exception as e:
        print(f"LMCache Init failed: {e}")
        return None

# ================= RUNNER =================
def run_stress_test():
    # Force clean memory before starting
    torch.cuda.empty_cache()
    
    setup_lmcache()
    llm = get_engine()
    
    if llm is None:
        print("CRITICAL: Engine failed to load.")
        return

    sp = SamplingParams(temperature=0, max_tokens=5) 
    
    results = []

    print(f"\n{'='*60}")
    print(f"STARTING SAFE MODE STRESS TEST (0.5B Model)")
    print(f"{'='*60}")

    for target_len in CONTEXT_LENGTHS:
        print(f"\n>>> Testing Context Length: {target_len} tokens")
        
        context_text = load_and_scale_context(SOURCE_FILE, target_len)
        prompt = f"{context_text}\n\nUser: Summary?\nAssistant:"
        
        # COLD RUN
        if CACHE_DIR.exists(): shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir()
        
        try:
            start = time.perf_counter()
            llm.generate([prompt], sp)
            cold_latency = time.perf_counter() - start
            
            # WARM RUN
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
        except Exception as e:
            print(f"    [Error] Run failed for {target_len} tokens: {e}")
            break # Stop if we hit OOM even here

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    
    if not df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(df["Context_Tokens"], df["Cold_Latency"], label="Cold", marker="o")
        plt.plot(df["Context_Tokens"], df["Warm_Latency"], label="Warm", marker="o")
        plt.xlabel("Tokens")
        plt.ylabel("Latency (s)")
        plt.title("LMCache Break-Even (0.5B Model)")
        plt.legend()
        plt.grid(True)
        plt.savefig("break_even_plot2.png")
        print("\nSuccess! Plot saved to break_even_plot.png")

if __name__ == "__main__":
    run_stress_test()