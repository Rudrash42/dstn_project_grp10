#!/bin/bash
# ==============================================================================
# RL Cache Prefetch Pipeline Runner
# ==============================================================================
# This script runs the entire training and evaluation pipeline for both the 
# Software Simulation and the Real Hardware Cache components.
# ==============================================================================

set -e # Exit immediately if any command fails

# Ensure we are in the correct directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$script_dir"

echo "=========================================================================="
echo " 🚀 Starting RL Cache Prefetch Full Pipeline"
echo "=========================================================================="

echo ""
echo "📊 STAGE 0: DATA GENERATION"
echo "--------------------------------------------------------------------------"
echo "Generating trace files..."
python data/generate_traces.py
if [ $? -ne 0 ]; then
    echo "❌ Error in data/generate_traces.py"
    exit 1
fi

echo ""
echo "🖥️  STAGE 1: SOFTWARE SIMULATION"
echo "--------------------------------------------------------------------------"
echo "Training the RL agent in the lightweight python Cache Simulator..."
python train.py
if [ $? -ne 0 ]; then
    echo "❌ Error in train.py"
    exit 1
fi

echo "Evaluating the simulation model against baselines..."
python eval/evaluate.py
if [ $? -ne 0 ]; then
    echo "❌ Error in eval/evaluate.py"
    exit 1
fi

echo "Generating comparison plots for the simulation metrics..."
python eval/plot_results.py
if [ $? -ne 0 ]; then
    echo "❌ Error in eval/plot_results.py"
    exit 1
fi


echo ""
echo "💾 STAGE 2: REAL HARDWARE EXECUTION"
echo "--------------------------------------------------------------------------"
echo "Training the RL agent on the real Hardware Cache..."
echo "(This creates real GPU tensors and physically writes files to NVMe)"
python train_hardware.py
if [ $? -ne 0 ]; then
    echo "❌ Error in train_hardware.py"
    exit 1
fi

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
