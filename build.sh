#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/scripts/common.sh"

BUILD_TYPE="Release"
JOBS="$(nproc)"

is_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

case $# in
  0)
    ;;
  1)
    if is_integer "$1"; then
      JOBS="$1"
    else
      BUILD_TYPE="$1"
    fi
    ;;
  2)
    if is_integer "$1" && ! is_integer "$2"; then
      JOBS="$1"
      BUILD_TYPE="$2"
    elif ! is_integer "$1" && is_integer "$2"; then
      BUILD_TYPE="$1"
      JOBS="$2"
    else
      log ERROR "With 2 arguments pass exactly one number (JOBS) and one string (BUILD_TYPE)." >&2
      log ERROR "Usage: $0 [BUILD_TYPE|JOBS] [BUILD_TYPE|JOBS]" >&2
      exit 1
    fi
    ;;
  *)
    log ERROR "Usage: $0 [BUILD_TYPE|JOBS] [BUILD_TYPE|JOBS]" >&2
    exit 1
    ;;
esac

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
