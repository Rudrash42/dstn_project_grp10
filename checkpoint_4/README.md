# Stage 4: Extended Action Space (Prefetch & Eviction)

## Overview

In Stages 1–3, the RL agent was responsible solely for proactive prefetching ($L_3 \rightarrow L_2$), while eviction across cache tiers followed a reactive Least Recently Used (LRU) policy. 

In **Stage 4 (Extended Action Space)**, the agent's decision space is generalized to joint prefetch and eviction management:
- **Prefetch Action ($16$ bits)**: MultiBinary mask indicating which candidate chunks in $L_3$ (NVMe SSD) should be proactively migrated to $L_2$ (CPU RAM).
- **Eviction Action ($16$ bits)**: MultiBinary mask indicating which chunks currently occupying $L_2$ (CPU RAM) should be proactively evicted to $L_3$ to avoid evicting high-utility data under high cache pressure.
- **Total Action Space**: $32$-dimensional binary vector (`MultiBinary(32)`).

---

## Architecture & Policy

- **Observation Vector ($403$-dim)**:
  - Query text embedding: $384$-dim (sentence-transformers / `all-MiniLM-L6-v2`)
  - Cache tier occupancy statistics: $3$-dim ($L_1$, $L_2$, $L_3$ fill ratios)
  - Candidate chunk recency & frequency scores: $16$-dim
- **Actor-Critic Policy Network**:
  - Shared feature extractor: MLP ($403 \rightarrow 64 \rightarrow 64$)
  - Action head: $32$ independent sigmoid logits parameterized as Bernoulli distributions.
- **Reward Formulation**:
  $$R_t = \alpha \cdot \Delta T_{\text{saved}} - \beta \cdot M_{\text{migrated}} - \gamma \cdot N_{\text{unused}}$$
  where prefetch and eviction penalties are balanced against latency gains.

---

## Key Hardware Results

Under multi-turn conversation and RAG workloads on RTX 3050 hardware:

| Workload | Strategy | Avg Latency (ms) | Hit Rate (%) | Speedup vs LRU |
| :--- | :--- | :--- | :--- | :--- |
| **prefix** | No Cache | 111.63 | 0.0% | 0.59× |
| | LRU | 65.75 | 32.7% | 1.00× |
| | Oracle | 65.15 | 32.7% | 1.01× |
| | **RL Agent** | **64.75** | **32.7%** | **1.02×** |
| **rag** | No Cache | 296.49 | 0.0% | 0.12× |
| | LRU | 36.63 | 85.8% | 1.00× |
| | Oracle | 36.68 | 85.8% | 1.00× |
| | **RL Agent** | **36.54** | **85.8%** | **1.00×** |
| **multiturn** | No Cache | 960.35 | 0.0% | 0.10× |
| | LRU | 91.82 | 36.5% | 1.00× |
| | Oracle | 66.20 | 66.3% | 1.39× |
| | **RL Agent** | **66.75** | **59.6%** | **1.38×** |

---

## Reproduction

To evaluate the full pipeline including hardware benchmarking and evaluation:
```bash
cd ../checkpoint_3/rl_cache_prefetch
./run_pipeline.sh
```
