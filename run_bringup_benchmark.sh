#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"
WRAPPER_ARGS=("$@")
set --
source scripts/set_env.sh >/dev/null
set -- "${WRAPPER_ARGS[@]}"
# Bring-up comparisons always use the complete hardware worker grid. The
# standalone-only diagnostic override must not leak in from an interactive
# shell or an earlier investigation.
unset TT_WAVELET_LWT_MAX_CORES

PYTHON="$ROOT_DIR/.venv/bin/python"
SUITE="$ROOT_DIR/scripts/wavelet_benchmark.py"
OUTPUT_BASE="$ROOT_DIR/benchmarks/bringup"
SEED=42
TEST_RUN=false
BUILD=true
STATE_ARGS=()
SCHEMES=()

while (($#)); do
    case "$1" in
        --test-run)
            TEST_RUN=true
            shift
            ;;
        --skip-build)
            BUILD=false
            shift
            ;;
        --seed)
            SEED=$2
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE=$2
            shift 2
            ;;
        --overwrite | --resume)
            STATE_ARGS=("$1")
            shift
            ;;
        --schemes)
            shift
            while (($#)) && [[ $1 != --* ]]; do
                SCHEMES+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ ! -x $PYTHON ]]; then
    echo "Missing virtual-environment Python: $PYTHON" >&2
    exit 2
fi
if [[ $BUILD == true ]]; then
    if [[ ! -f build/CMakeCache.txt ]]; then
        echo "Missing configured build tree; configure it with:" >&2
        echo "  cmake -S . -B build -DBUILD_TT_WAVELET=ON" >&2
        exit 2
    fi
    BUILD_JOBS=${TT_WAVELET_BUILD_JOBS:-$(nproc)}
    echo "Incremental build (standalone runner + TTNN, jobs=$BUILD_JOBS)..."
    cmake --build build \
        --target tt_wavelet_benchmark_runner ttnn \
        --parallel "$BUILD_JOBS"
elif [[ ! -x build/tt_wavelet_benchmark_runner ]]; then
    echo "Missing standalone runner; omit --skip-build or build with:" >&2
    echo "  cmake --build build --target tt_wavelet_benchmark_runner ttnn --parallel \$(nproc)" >&2
    exit 2
fi

COMMON=(--seed "$SEED")
if ((${#SCHEMES[@]})); then
    COMMON+=(--wavelets "${SCHEMES[@]}")
fi

if [[ $TEST_RUN == true ]]; then
    if ((${#SCHEMES[@]} == 0)); then
        COMMON+=(--wavelets db2 coif3 db20 coif17)
    fi
    MODES=(--boundary-modes symmetric antireflect)
    PRECISION_SIZE=(--length 257 --height 35 --width 37)
    PERF_1D_SIZE=(--lengths 100000 500000)
    PERF_2D_SIZE=(--shapes 1000x100 1000x500)
    REPEATS=(--warmup-runs 1 --repeats 3)
else
    MODES=(--boundary-modes symmetric zero constant periodic antisymmetric smooth reflect antireflect)
    PRECISION_SIZE=(--length 1024 --height 64 --width 64)
    PERF_1D_SIZE=(--length-start 100000 --length-stop 1000000 --length-step 10000)
    PERF_2D_SIZE=()
    REPEATS=(--warmup-runs 1 --repeats 20)
fi

echo "Seed: $SEED"
echo "Output: $OUTPUT_BASE"

if ! "$PYTHON" "$SUITE" precision \
    "${COMMON[@]}" "${MODES[@]}" "${PRECISION_SIZE[@]}" "${STATE_ARGS[@]}" \
    --output-dir "$OUTPUT_BASE/precision"; then
    echo "Precision failures recorded; performance remains gated by its mandatory selected-wavelet preflight." >&2
fi

"$PYTHON" "$SUITE" performance-1d \
    "${COMMON[@]}" "${MODES[@]}" "${PERF_1D_SIZE[@]}" "${REPEATS[@]}" "${STATE_ARGS[@]}" \
    --output-dir "$OUTPUT_BASE/performance-1d"

"$PYTHON" "$SUITE" performance-2d \
    "${COMMON[@]}" "${MODES[@]}" "${PERF_2D_SIZE[@]}" "${REPEATS[@]}" "${STATE_ARGS[@]}" \
    --output-dir "$OUTPUT_BASE/performance-2d"

"$PYTHON" "$SUITE" plot \
    --csv \
    "$OUTPUT_BASE/performance-1d/performance_1d.csv" \
    "$OUTPUT_BASE/performance-2d/performance_2d.csv" \
    --output-dir "$OUTPUT_BASE/plots"

echo "Bring-up suite complete: $OUTPUT_BASE"
