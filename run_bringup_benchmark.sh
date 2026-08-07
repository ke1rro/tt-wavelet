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
SETUP_ONLY=false
BUILD=true
FORCE_BOOTSTRAP=false
STATE_ARGS=()
SCHEMES=()

usage() {
    cat <<'EOF'
Usage: ./run_bringup_benchmark.sh [OPTIONS]

Setup/build options:
  --setup-only       Install/integrate/build, then exit without using hardware
  --bootstrap        Reinstall dependencies and reconfigure the existing build
  --skip-build       Skip integration, dependency checks, and compilation

Benchmark options:
  --test-run         Run the targeted smoke matrix instead of the full suite
  --seed N           Reproducible performance-wavelet seed (default: 42)
  --schemes NAMES... Explicit wavelets instead of metadata-based selection
  --output-base DIR  Output root (default: benchmarks/bringup)
  --overwrite        Replace existing result directories
  --resume           Continue compatible result directories
EOF
}

require_value() {
    if [[ -z ${2-} || ${2-} == --* ]]; then
        echo "$1 requires a value" >&2
        exit 2
    fi
}

while (($#)); do
    case "$1" in
        --test-run)
            TEST_RUN=true
            shift
            ;;
        --setup-only)
            SETUP_ONLY=true
            shift
            ;;
        --skip-build)
            BUILD=false
            shift
            ;;
        --bootstrap)
            FORCE_BOOTSTRAP=true
            shift
            ;;
        --seed)
            require_value "$1" "${2-}"
            SEED=$2
            shift 2
            ;;
        --output-base)
            require_value "$1" "${2-}"
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
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ $BUILD == false && ( $SETUP_ONLY == true || $FORCE_BOOTSTRAP == true ) ]]; then
    echo "--skip-build cannot be combined with --setup-only or --bootstrap" >&2
    exit 2
fi

if [[ $BUILD == true ]]; then
    if [[ ! -f tt-metal/CMakeLists.txt ]] || git submodule status --recursive 2>/dev/null | grep -q '^-'; then
        echo "Initializing repository submodules..."
        git submodule update --init --recursive
    fi

    # The integration step is idempotent. It links the live ttnn-wavelet tree
    # into TT-Metal and applies the five required CMake/nanobind hooks exactly
    # once; it never moves or copies the operation sources.
    scripts/setup_ttnn_wavelet_in_ttmetal.sh --skip-build

    NEED_SYSTEM_BOOTSTRAP=false
    NEED_PYTHON_BOOTSTRAP=false
    if [[ $FORCE_BOOTSTRAP == true || ! -f build/CMakeCache.txt ]]; then
        NEED_SYSTEM_BOOTSTRAP=true
    fi
    if ! command -v clang-20 >/dev/null 2>&1 || ! command -v clang++-20 >/dev/null 2>&1; then
        NEED_SYSTEM_BOOTSTRAP=true
    fi
    if [[ $FORCE_BOOTSTRAP == true || ! -x $PYTHON ]] || \
        ! "$PYTHON" -c 'import matplotlib, numpy, pywt, scipy, torch, tqdm' >/dev/null 2>&1; then
        NEED_PYTHON_BOOTSTRAP=true
    fi

    # shellcheck disable=SC1091
    source scripts/common.sh
    if [[ $NEED_SYSTEM_BOOTSTRAP == true ]]; then
        echo "Bootstrapping compiler, TT-Metal system dependencies, Python environment, and CMake..."
        run_tt_metal_install_deps
        ensure_base_deps
        configure_project Release
        NEED_PYTHON_BOOTSTRAP=false
    elif [[ $NEED_PYTHON_BOOTSTRAP == true ]]; then
        echo "Creating Python environment and installing benchmark dependencies..."
        ensure_python_packages
    fi

    BUILD_JOBS=${TT_WAVELET_BUILD_JOBS:-$(nproc)}
    echo "Incremental build (standalone runner + TTNN, jobs=$BUILD_JOBS)..."
    cmake --build build \
        --target tt_wavelet_benchmark_runner ttnn \
        --parallel "$BUILD_JOBS"

    # Refresh extension links after the build. PYTHONPATH from set_env.sh then
    # imports this checkout rather than any server-wide TTNN installation.
    scripts/setup_ttnn_wavelet_in_ttmetal.sh --skip-build

    if [[ $NEED_SYSTEM_BOOTSTRAP == true || $NEED_PYTHON_BOOTSTRAP == true || $FORCE_BOOTSTRAP == true ]]; then
        echo "Installing this TT-Metal checkout into the local virtual environment..."
        "$PYTHON" -m pip install -e "$ROOT_DIR/tt-metal"
    fi
elif [[ ! -x build/tt_wavelet_benchmark_runner ]]; then
    echo "Missing standalone runner; omit --skip-build or build with:" >&2
    echo "  cmake --build build --target tt_wavelet_benchmark_runner ttnn --parallel \$(nproc)" >&2
    exit 2
fi

if [[ ! -x $PYTHON ]]; then
    echo "Missing virtual-environment Python: $PYTHON" >&2
    echo "Run without --skip-build to bootstrap dependencies and build the repository." >&2
    exit 2
fi

if [[ $SETUP_ONLY == true ]]; then
    echo "Setup complete: TTNN-Wavelet integrated; standalone runner and TTNN built."
    exit 0
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
