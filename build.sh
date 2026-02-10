#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/scripts/common.sh"

BUILD_TYPE=${1:-Release}
# Parallel jobs; default to nproc if not provided
JOBS=${2:-$(nproc)}

log INFO "Starting full build (tt-metal + tt-wavelet) with -j${JOBS}"
export_tt_env
ensure_base_deps
run_tt_metal_install_deps
apply_cmake_fixes
configure_project "$BUILD_TYPE"
cmake --build "$BUILD_DIR" -j"${JOBS}"

log INFO "Installing tt-metal Python bindings into the venv"
activate_venv
python -m pip install -e "$TT_METAL_DIR"

log INFO "Build complete. Outputs in $BUILD_DIR"
log INFO "To keep env vars in this shell: source $SCRIPT_DIR/scripts/set_env.sh"
