#!/usr/bin/env bash
# ==============================================================================
# Architecture-Independent Precision + Performance Benchmark Suite for tt-wavelet
# ==============================================================================
# Usage:
#   ./run_bringup_benchmark.sh                # Full overnight benchmark run
#   ./run_bringup_benchmark.sh --test-run     # Pipeline validation test run
#   ./run_bringup_benchmark.sh --seed 42       # Run with specific random seed
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd "$PROJECT_ROOT"

# Parse Command Line Arguments
TEST_RUN=false
SEED=""
EXPLICIT_SCHEMES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test-run)
      TEST_RUN=true
      shift
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --schemes)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        EXPLICIT_SCHEMES+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--test-run] [--seed <INT>] [--schemes <SCHEME1> <SCHEME2>...]"
      exit 1
      ;;
  esac
done

# Source Environment Setup
if [[ -f "scripts/set_env.sh" ]]; then
  echo "[Setup] Sourcing environment from scripts/set_env.sh..."
  source "scripts/set_env.sh"
fi

# Detect Python Executable
PYTHON_EXEC="python3"
if [[ -n "${VENV_PYTHON:-}" && -f "$VENV_PYTHON" ]]; then
  PYTHON_EXEC="$VENV_PYTHON"
elif [[ -d ".venv" && -f ".venv/bin/python3" ]]; then
  PYTHON_EXEC=".venv/bin/python3"
fi

echo "=========================================================================="
echo "          TT-WAVELET AUTOMATED BENCHMARK SUITE          "
echo "=========================================================================="
echo "Python Executable: $PYTHON_EXEC"
echo "Test Run Mode:     $TEST_RUN"
if [[ -n "$SEED" ]]; then
  echo "Random Seed:       $SEED"
fi

# Setup Directory Hierarchy
mkdir -p benchmarks/precision
mkdir -p benchmarks/performance/preflight
mkdir -p benchmarks/performance/1d/plots
mkdir -p benchmarks/performance/2d/plots

# ALL 8 Boundary Modes
ALL_MODES=(symmetric zero constant periodic antisymmetric smooth reflect antireflect)

# Phase 1: Precision Validation Sweep
echo ""
echo "--------------------------------------------------------------------------"
echo "[Phase 1/5] Running Precision & Correctness Validation Suite"
echo "--------------------------------------------------------------------------"
if [[ "$TEST_RUN" == "true" ]]; then
  echo "[Test Run] Running precision validation on representative schemes (1D + 2D)..."
  $PYTHON_EXEC scripts/validate_all_106_schemes_precision.py \
    --schemes db1 bior1.3 sym6 db7 \
    --dimensions 1d 2d \
    --signal-len-1d 256 \
    --matrix-height-2d 32 \
    --matrix-width-2d 32 \
    --output-json benchmarks/precision/precision_results.json \
    --output-tsv benchmarks/precision/precision_results.tsv
else
  echo "[Production Run] Running precision validation on ALL 106 schemes (1D + 2D)..."
  $PYTHON_EXEC scripts/validate_all_106_schemes_precision.py \
    --dimensions 1d 2d \
    --signal-len-1d 1024 \
    --matrix-height-2d 64 \
    --matrix-width-2d 64 \
    --output-json benchmarks/precision/precision_results.json \
    --output-tsv benchmarks/precision/precision_results.tsv
fi

# Phase 2: Common Performance Scheme Selection & Preflight
echo ""
echo "--------------------------------------------------------------------------"
echo "[Phase 2/5] Common Performance Scheme Selection & Mandatory Preflight"
echo "--------------------------------------------------------------------------"
SELECT_CMD=("$PYTHON_EXEC" "scripts/select_and_preflight_schemes.py" "--output-file" "benchmarks/performance/selected_schemes.txt")
if [[ -n "$SEED" ]]; then
  SELECT_CMD+=("--seed" "$SEED")
fi
if [[ ${#EXPLICIT_SCHEMES[@]} -gt 0 ]]; then
  SELECT_CMD+=("--schemes" "${EXPLICIT_SCHEMES[@]}")
fi

"${SELECT_CMD[@]}"

ACCEPTED_SCHEMES=($(cat benchmarks/performance/selected_schemes.txt))
echo "Accepted Performance Schemes for 1D and 2D: ${ACCEPTED_SCHEMES[*]}"

# Phase 3: 1D Performance Sweep
echo ""
echo "--------------------------------------------------------------------------"
echo "[Phase 3/5] Benchmarking 1D Performance Grid (PyWT vs tt-wavelet vs TTNN)"
echo "--------------------------------------------------------------------------"

if [[ "$TEST_RUN" == "true" ]]; then
  echo "[Test Run] Benchmarking 1D representative signal length N=500000 (repeats=3)..."
  $PYTHON_EXEC compare_timings.py \
    --backend all \
    --transform both \
    --wavelets "${ACCEPTED_SCHEMES[@]}" \
    --boundary-modes "${ALL_MODES[@]}" \
    --lengths 500000 \
    --pywt-repeats 3 \
    --pywt-warmup-runs 1 \
    --tt-repeats 3 \
    --tt-warmup-runs 1 \
    --csv benchmarks/performance/1d/summary_1d.tsv \
    --overwrite
else
  echo "[Production Run] Benchmarking 1D sweep grid (100k -> 1M step 10k, exactly 91 lengths, repeats=20)..."
  $PYTHON_EXEC compare_timings.py \
    --backend all \
    --transform both \
    --wavelets "${ACCEPTED_SCHEMES[@]}" \
    --boundary-modes "${ALL_MODES[@]}" \
    --length-start 100000 \
    --length-stop 1000000 \
    --length-step 10000 \
    --pywt-repeats 20 \
    --pywt-warmup-runs 1 \
    --tt-repeats 20 \
    --tt-warmup-runs 1 \
    --csv benchmarks/performance/1d/summary_1d.tsv \
    --overwrite
fi

# Phase 4: 2D Performance Sweep
echo ""
echo "--------------------------------------------------------------------------"
echo "[Phase 4/5] Benchmarking 2D Performance Grid (PyWT vs tt-wavelet vs TTNN)"
echo "--------------------------------------------------------------------------"

if [[ "$TEST_RUN" == "true" ]]; then
  echo "[Test Run] Benchmarking 2D representative shape 1000x500 (repeats=3)..."
  $PYTHON_EXEC compare_timings.py \
    --backend all \
    --transform both \
    --wavelets "${ACCEPTED_SCHEMES[@]}" \
    --boundary-modes "${ALL_MODES[@]}" \
    --shapes "1000x500" \
    --pywt-repeats 3 \
    --pywt-warmup-runs 1 \
    --tt-repeats 3 \
    --tt-warmup-runs 1 \
    --csv benchmarks/performance/2d/summary_2d.tsv \
    --overwrite
else
  echo "[Production Run] Generating 91 matrix shapes 1000x100 -> 1000x1000 (repeats=20)..."
  SHAPES_2D=($(seq 100 10 1000 | sed 's/^/1000x/'))
  $PYTHON_EXEC compare_timings.py \
    --backend all \
    --transform both \
    --wavelets "${ACCEPTED_SCHEMES[@]}" \
    --boundary-modes "${ALL_MODES[@]}" \
    --shapes "${SHAPES_2D[@]}" \
    --pywt-repeats 20 \
    --pywt-warmup-runs 1 \
    --tt-repeats 20 \
    --tt-warmup-runs 1 \
    --csv benchmarks/performance/2d/summary_2d.tsv \
    --overwrite
fi

# Phase 5: Strict Plot Generation
echo ""
echo "--------------------------------------------------------------------------"
echo "[Phase 5/5] Generating 3-Curve Log-Scale Performance Line Plots"
echo "--------------------------------------------------------------------------"
$PYTHON_EXEC scripts/generate_benchmark_plots.py \
  --summary-file benchmarks/performance/1d/summary_1d.tsv \
  --output-dir benchmarks/performance/1d/plots

$PYTHON_EXEC scripts/generate_benchmark_plots.py \
  --summary-file benchmarks/performance/2d/summary_2d.tsv \
  --output-dir benchmarks/performance/2d/plots

echo ""
echo "=========================================================================="
echo "          BENCHMARK SUITE EXECUTION COMPLETE           "
echo "=========================================================================="
echo "Precision Results:    benchmarks/precision/precision_results.tsv"
echo "Selected Schemes:     benchmarks/performance/selected_schemes.txt"
echo "1D Summary:           benchmarks/performance/1d/summary_1d.tsv"
echo "2D Summary:           benchmarks/performance/2d/summary_2d.tsv"
echo "1D Plots Directory:   benchmarks/performance/1d/plots"
echo "2D Plots Directory:   benchmarks/performance/2d/plots"
echo "=========================================================================="
