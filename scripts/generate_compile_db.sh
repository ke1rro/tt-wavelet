#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/ci-clang-tidy"

if command -v ninja >/dev/null 2>&1; then
  GENERATOR="Ninja"
else
  GENERATOR="Unix Makefiles"
fi

CMAKE_ARGS=(
  -S "${ROOT_DIR}/ci"
  -B "${BUILD_DIR}"
  -G "${GENERATOR}"
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  -DTT_WAVELET_SOURCE_ROOT="${ROOT_DIR}"
  -DCI_BUILD_WAVELET_EXECUTABLE=OFF
)

if command -v clang-20 >/dev/null 2>&1 && command -v clang++-20 >/dev/null 2>&1; then
  CMAKE_ARGS+=(
    -DCMAKE_C_COMPILER=clang-20
    -DCMAKE_CXX_COMPILER=clang++-20
  )
fi

if [[ -n "${TT_METAL_PREBUILT_ROOT:-}" ]]; then
  CMAKE_ARGS+=("-DTT_METAL_PREBUILT_ROOT=${TT_METAL_PREBUILT_ROOT}")
else
  CMAKE_ARGS+=("-DTT_METAL_SOURCE_ROOT=${ROOT_DIR}/tt-metal")
fi

cmake "${CMAKE_ARGS[@]}"

if [[ ! -f "${BUILD_DIR}/compile_commands.json" ]]; then
  echo "compile_commands.json was not generated in ${BUILD_DIR}" >&2
  exit 1
fi

echo "Generated ${BUILD_DIR}/compile_commands.json"
