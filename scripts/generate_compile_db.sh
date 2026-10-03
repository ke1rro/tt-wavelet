#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${ROOT_DIR}/build_clang_tidy}"
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DTT_WAVELET_BUILD_SMOKE=OFF
# Generate both dependency and standalone headers before analyzing host code.
cmake --build "${BUILD_DIR}" --target all_generated_files tt_wavelet_schemes
printf 'Compile database: %s/compile_commands.json\n' "${BUILD_DIR}"
