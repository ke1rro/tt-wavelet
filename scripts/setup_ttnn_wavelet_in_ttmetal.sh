#!/usr/bin/env bash
set -eo pipefail

# Setup & Integration Script for ttnn-wavelet into tt-metal
#
# This script:
# 1. Sets up environment variables (TT_METAL_HOME, TT_METAL_ROOT, LD_LIBRARY_PATH, PYTHONPATH)
# 2. Applies integration patches to tt-metal
# 3. Symlinks ttnn-wavelet operations and tests into tt-metal
# 4. Compiles the C++ tree and updates Python virtualenv bindings

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "========================================================================"
echo "          Setting up ttnn-wavelet in tt-metal repository                "
echo "========================================================================"

# Source environment
if [ -f "${REPO_ROOT}/scripts/set_env.sh" ]; then
    source "${REPO_ROOT}/scripts/set_env.sh"
fi

TT_METAL_HOME="${TT_METAL_HOME:-${REPO_ROOT}/tt-metal}"
TT_METAL_ROOT="${TT_METAL_ROOT:-${REPO_ROOT}/tt-metal}"
VENV_DIR="${REPO_ROOT}/.venv"

export TT_METAL_HOME TT_METAL_ROOT

echo "[1/5] Checking tt-metal directory at: ${TT_METAL_HOME}"
if [ ! -d "${TT_METAL_HOME}" ]; then
    echo "ERROR: TT_METAL_HOME directory not found at ${TT_METAL_HOME}"
    exit 1
fi

echo "[2/5] Applying integration patch..."
PATCH_FILE="${REPO_ROOT}/ttnn-wavelet/patches/integration-hooks.patch"
if [ -f "${PATCH_FILE}" ]; then
    cd "${TT_METAL_HOME}"
    git apply "${PATCH_FILE}" 2>/dev/null || echo "Patch already applied or skipped."
    cd "${REPO_ROOT}"
fi

echo "[3/5] Creating symlinks for wavelet operations and tests..."
mkdir -p "${TT_METAL_HOME}/ttnn/cpp/ttnn/operations"
mkdir -p "${TT_METAL_HOME}/tests/ttnn/unit_tests/operations"

ln -sfn "${REPO_ROOT}/ttnn-wavelet/ttnn/cpp/ttnn/operations/wavelet" \
       "${TT_METAL_HOME}/ttnn/cpp/ttnn/operations/wavelet"

ln -sfn "${REPO_ROOT}/ttnn-wavelet/tests/ttnn/unit_tests/operations/wavelet" \
       "${TT_METAL_HOME}/tests/ttnn/unit_tests/operations/wavelet"

echo "[4/5] Building tt-metal C++ tree with ttnn target..."
cd "${REPO_ROOT}"
cmake --build build -j$(nproc)

echo "[5/5] Updating Python virtualenv bindings symlinks..."
BUILD_TTNN_DIR="${REPO_ROOT}/build/tt-metal/ttnn"
if [ -d "${BUILD_TTNN_DIR}" ]; then
    ln -sf "${BUILD_TTNN_DIR}/_ttnn.so" "${TT_METAL_HOME}/ttnn/ttnn/_ttnn.so"
    ln -sf "${BUILD_TTNN_DIR}/_ttnncpp.so" "${TT_METAL_HOME}/ttnn/ttnn/_ttnncpp.so"
    
    mkdir -p "${TT_METAL_HOME}/build_wavelet/ttnn"
    ln -sf "${BUILD_TTNN_DIR}/_ttnn.so" "${TT_METAL_HOME}/build_wavelet/ttnn/_ttnn.so"
    ln -sf "${BUILD_TTNN_DIR}/_ttnncpp.so" "${TT_METAL_HOME}/build_wavelet/ttnn/_ttnncpp.so"
fi

echo "========================================================================"
echo "          SUCCESS: ttnn-wavelet integrated and built cleanly!           "
echo "========================================================================"
