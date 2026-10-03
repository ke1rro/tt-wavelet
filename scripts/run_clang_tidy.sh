#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_clang_tidy"
if [[ ! -f "${BUILD_DIR}/compile_commands.json" ]]; then
  "${ROOT_DIR}/scripts/generate_compile_db.sh" "${BUILD_DIR}"
fi
if [[ $# -eq 0 ]]; then
  set -- "${ROOT_DIR}/src/ttwt/wavelet.cpp"
fi
exec clang-tidy-20 -p "${BUILD_DIR}" "$@"
