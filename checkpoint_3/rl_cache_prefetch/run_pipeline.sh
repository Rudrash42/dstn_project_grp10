#!/bin/bash
# ==============================================================================
# RL Cache Prefetch Pipeline Runner (Hardware-Only)
# ==============================================================================
# This script runs the entire training and evaluation pipeline using the
# Real Hardware Cache backend. No vLLM or LMCache required.
#
# Pipeline stages:
#   STAGE 0: Generate traces (hardware calibration + synthetic workloads)
#   STAGE 1: Train RL agent on real hardware cache
#   STAGE 2: Evaluate against baselines on real hardware
# ==============================================================================

set -e # Exit immediately if any command fails

# Ensure we are in the correct directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$script_dir"

echo "=========================================================================="
echo " 🚀 Starting RL Cache Prefetch Pipeline (Hardware-Only)"
echo "=========================================================================="

echo ""
echo "📊 STAGE 0: DATA GENERATION (Hardware Calibration)"
echo "--------------------------------------------------------------------------"
echo "Calibrating real hardware latencies and generating workload traces..."
python data/generate_traces.py "$@"
if [ $? -ne 0 ]; then
    echo "❌ Error in data/generate_traces.py"
    exit 1
fi

echo ""
echo "💾 STAGE 1: TRAINING ON REAL HARDWARE"
echo "--------------------------------------------------------------------------"
echo "Training the RL agent on the real Hardware Cache..."
echo "(This creates real GPU tensors and physically writes files to NVMe)"
python train_hardware.py
if [ $? -ne 0 ]; then
    echo "❌ Error in train_hardware.py"
    exit 1
fi

echo ""
echo "📊 STAGE 2: EVALUATION ON REAL HARDWARE"
echo "--------------------------------------------------------------------------"
echo "Evaluating the hardware model against baselines..."
python eval/evaluate_hardware.py
if [ $? -ne 0 ]; then
    echo "❌ Error in eval/evaluate_hardware.py"
    exit 1
fi

echo ""
echo "=========================================================================="
echo " ✅ Pipeline Complete!"
echo " All final metrics, CSVs, and plots are available in the 'results/' directory."
echo "=========================================================================="
