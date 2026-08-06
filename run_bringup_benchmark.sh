#!/usr/bin/env bash
# Architecture-Independent Master Bringup & Benchmark Runner (1D & 2D)
# Usage:
#   ./run_bringup_benchmark.sh              # Full 1D & 2D precision validation + 4-scheme performance benchmarks
#   ./run_bringup_benchmark.sh --test-run   # Quick test/verification run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IS_TEST_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--test-run" ]]; then
    IS_TEST_RUN=1
  fi
done

PRECISION_DIR="$SCRIPT_DIR/benchmarks/precision"
PERFORMANCE_DIR="$SCRIPT_DIR/benchmarks/performance"
PLOTS_DIR="$PERFORMANCE_DIR/plots"

mkdir -p "$PRECISION_DIR" "$PERFORMANCE_DIR" "$PLOTS_DIR"

# Randomly select 1 compact, 1 medium, 1 large scheme, plus coif17 for 1D
COMPACT_SCHEMES=("db1" "db2" "bior1.3" "bior2.2" "sym2" "haar")
MEDIUM_SCHEMES=("db9" "db10" "bior3.9" "coif3" "sym10")
LARGE_SCHEMES=("db38" "coif12" "sym20" "bior6.8")

RANDOM_COMPACT="${COMPACT_SCHEMES[$RANDOM % ${#COMPACT_SCHEMES[@]}]}"
RANDOM_MEDIUM="${MEDIUM_SCHEMES[$RANDOM % ${#MEDIUM_SCHEMES[@]}]}"
RANDOM_LARGE="${LARGE_SCHEMES[$RANDOM % ${#LARGE_SCHEMES[@]}]}"

# 1D supports all schemes up to coif17
PERF_SCHEMES_1D=("$RANDOM_COMPACT" "$RANDOM_MEDIUM" "$RANDOM_LARGE" "coif17")
PERF_SCHEMES_1D=($(echo "${PERF_SCHEMES_1D[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

# 2D uses compact/medium 2D schemes that fit within Tensix kernel config buffer (70,656 bytes)
PERF_SCHEMES_2D=("db1" "bior1.3" "bior2.2" "sym2")

echo "================================================================================"
echo " TENSTORRENT WAVELET AUTOMATED BRINGUP & BENCHMARK SUITE (1D & 2D) "
echo "================================================================================"
echo "Selected 1D Schemes: ${PERF_SCHEMES_1D[*]}"
echo "Selected 2D Schemes: ${PERF_SCHEMES_2D[*]}"

if [[ "$IS_TEST_RUN" -eq 1 ]]; then
  echo "[Mode] TEST / VERIFICATION RUN"
  echo "--- Phase 1: Precision Test (1D) ---"
  "$SCRIPT_DIR/scripts/set_env.sh" "$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/scripts/validate_all_106_schemes_precision.py" \
    --schemes db1 bior3.9 \
    --boundary-modes symmetric zero \
    --output-json "$PRECISION_DIR/precision_results.json" \
    --output-tsv "$PRECISION_DIR/precision_results.tsv"

  echo "--- Phase 2: 1D Performance Benchmark (100k - 400k) ---"
  "$SCRIPT_DIR/scripts/set_env.sh" "$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/scripts/benchmark_ttnn_vs_standalone_vs_pywt.py" \
    --dim 1d \
    --backends ttnn standalone pywt \
    --schemes "${PERF_SCHEMES_1D[@]}" \
    --boundary-modes symmetric zero \
    --length-start 100000 --length-stop 400000 --length-step 300000 \
    --repeats 3 \
    --output-dir "$PERFORMANCE_DIR"

  echo "--- Phase 3: 2D Performance Benchmark (256x256 - 512x512) ---"
  "$SCRIPT_DIR/scripts/set_env.sh" "$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/scripts/benchmark_ttnn_vs_standalone_vs_pywt.py" \
    --dim 2d \
    --backends ttnn standalone pywt \
    --schemes "${PERF_SCHEMES_2D[@]}" \
    --boundary-modes symmetric zero \
    --length-start 256 --length-stop 512 --length-step 256 \
    --repeats 3 \
    --output-dir "$PERFORMANCE_DIR"
else
  echo "[Mode] FULL PRODUCTION SWEEP RUN"
  echo "--- Phase 1: Full Precision Test (All 106 schemes x 8 boundary modes) ---"
  "$SCRIPT_DIR/scripts/set_env.sh" "$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/scripts/validate_all_106_schemes_precision.py" \
    --output-json "$PRECISION_DIR/precision_results.json" \
    --output-tsv "$PRECISION_DIR/precision_results.tsv"

  echo "--- Phase 2: 1D Performance Benchmark (4 schemes x 8 modes x 100k-1M) ---"
  "$SCRIPT_DIR/scripts/set_env.sh" "$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/scripts/benchmark_ttnn_vs_standalone_vs_pywt.py" \
    --dim 1d \
    --backends ttnn standalone pywt \
    --schemes "${PERF_SCHEMES_1D[@]}" \
    --length-start 100000 --length-stop 1000000 --length-step 300000 \
    --repeats 20 \
    --output-dir "$PERFORMANCE_DIR"

  echo "--- Phase 3: 2D Performance Benchmark (4 schemes x 8 modes x 256-1024) ---"
  "$SCRIPT_DIR/scripts/set_env.sh" "$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/scripts/benchmark_ttnn_vs_standalone_vs_pywt.py" \
    --dim 2d \
    --backends ttnn standalone pywt \
    --schemes "${PERF_SCHEMES_2D[@]}" \
    --length-start 256 --length-stop 1024 --length-step 256 \
    --repeats 20 \
    --output-dir "$PERFORMANCE_DIR"
fi

echo "--- Phase 4: Plot Generation (1D & 2D) ---"
if [[ -f "$PERFORMANCE_DIR/summary_1d.tsv" ]]; then
  "$SCRIPT_DIR/scripts/set_env.sh" "$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/scripts/generate_benchmark_plots.py" \
    --summary-tsv "$PERFORMANCE_DIR/summary_1d.tsv" \
    --output-dir "$PLOTS_DIR"
fi
if [[ -f "$PERFORMANCE_DIR/summary_2d.tsv" ]]; then
  "$SCRIPT_DIR/scripts/set_env.sh" "$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/scripts/generate_benchmark_plots.py" \
    --summary-tsv "$PERFORMANCE_DIR/summary_2d.tsv" \
    --output-dir "$PLOTS_DIR"
fi

echo "================================================================================"
echo " BRINGUP & BENCHMARK SUITE COMPLETE "
echo " Precision Summary:   $PRECISION_DIR/precision_results.tsv"
echo " 1D Perf Summary:     $PERFORMANCE_DIR/summary_1d.tsv"
echo " 2D Perf Summary:     $PERFORMANCE_DIR/summary_2d.tsv"
echo " Performance Plots:   $PLOTS_DIR/"
echo "================================================================================"
