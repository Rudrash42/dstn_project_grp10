#!/usr/bin/env bash
# ==============================================================================
# RL Cache Prefetch Pipeline Runner
# ==============================================================================
# Runs the full end-to-end flow:
#   1) strict real trace generation
#   2) simulator training + evaluation + plots
#   3) hardware training + evaluation
#
# Usage examples:
#   ./run_pipeline.sh
#   ./run_pipeline.sh --max-queries 50
#   ./run_pipeline.sh --quick
#   ./run_pipeline.sh --skip-hw
# ==============================================================================

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$script_dir"

MAX_QUERIES="${MAX_QUERIES:-50}"
RUN_SIM=1
RUN_HW=1
QUICK=0

print_help() {
	cat <<EOF
Usage: ./run_pipeline.sh [options]

Options:
	--max-queries N   Number of queries per workload for trace generation (default: ${MAX_QUERIES})
	--quick           Run quick training mode for simulator and hardware
	--skip-sim        Skip simulator stage
	--skip-hw         Skip hardware stage
	-h, --help        Show this help

Environment overrides:
	PYTHON_BIN        Explicit Python executable to use
	MAX_QUERIES       Default max queries if --max-queries is not provided
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--max-queries)
			if [[ $# -lt 2 ]]; then
				echo "[error] --max-queries requires a value" >&2
				exit 2
			fi
			MAX_QUERIES="$2"
			shift 2
			;;
		--quick)
			QUICK=1
			shift
			;;
		--skip-sim)
			RUN_SIM=0
			shift
			;;
		--skip-hw)
			RUN_HW=0
			shift
			;;
		-h|--help)
			print_help
			exit 0
			;;
		*)
			echo "[error] Unknown option: $1" >&2
			print_help
			exit 2
			;;
	esac
done

resolve_python() {
	if [[ -n "${PYTHON_BIN:-}" ]]; then
		echo "$PYTHON_BIN"
		return
	fi

	local default_venv="/home/rudrash/prog/dstn/.venv/bin/python"
	if [[ -x "$default_venv" ]]; then
		echo "$default_venv"
		return
	fi

	if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
		echo "${VIRTUAL_ENV}/bin/python"
		return
	fi

	if command -v python3 >/dev/null 2>&1; then
		command -v python3
		return
	fi

	if command -v python >/dev/null 2>&1; then
		command -v python
		return
	fi

	echo ""
}

PYTHON="$(resolve_python)"
if [[ -z "$PYTHON" ]]; then
	echo "[error] No Python interpreter found. Set PYTHON_BIN or activate a venv." >&2
	exit 1
fi

run_step() {
	local label="$1"
	shift
	echo ""
	echo "[step] ${label}"
	echo "[cmd ] $*"
	"$@"
}

echo "=========================================================================="
echo "Starting RL Cache Prefetch Full Pipeline"
echo "========================================================================="
echo "Python: ${PYTHON}"
echo "Max queries: ${MAX_QUERIES}"
echo "Run simulator stage: ${RUN_SIM}"
echo "Run hardware stage: ${RUN_HW}"
echo "Quick mode: ${QUICK}"

echo ""
echo "STAGE 0: DATA GENERATION"
echo "--------------------------------------------------------------------------"
run_step "Generate strict real traces" \
	"$PYTHON" data/generate_traces.py --max-queries "$MAX_QUERIES"

if [[ "$RUN_SIM" -eq 1 ]]; then
	echo ""
	echo "STAGE 1: SOFTWARE SIMULATION"
	echo "--------------------------------------------------------------------------"

	if [[ "$QUICK" -eq 1 ]]; then
		run_step "Train simulator (quick)" "$PYTHON" train.py --quick
	else
		run_step "Train simulator" "$PYTHON" train.py
	fi

	run_step "Evaluate simulator" "$PYTHON" eval/evaluate.py
	run_step "Plot simulator metrics" "$PYTHON" eval/plot_results.py
fi

if [[ "$RUN_HW" -eq 1 ]]; then
	echo ""
	echo "STAGE 2: REAL HARDWARE EXECUTION"
	echo "--------------------------------------------------------------------------"

	if [[ "$QUICK" -eq 1 ]]; then
		run_step "Train hardware (quick)" "$PYTHON" train_hardware.py --quick
	else
		run_step "Train hardware" "$PYTHON" train_hardware.py
	fi

	run_step "Evaluate hardware" "$PYTHON" eval/evaluate_hardware.py
fi

echo ""
echo "=========================================================================="
echo "Pipeline Complete"
echo "All final metrics, CSVs, and plots are in results/"
echo "=========================================================================="
