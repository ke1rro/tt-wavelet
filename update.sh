#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/scripts/common.sh"

BUILD_TYPE=${1:-Release}
TARGET=${2:-tt_wavelet_test}

log INFO "Updating tt-wavelet only (target: $TARGET)"
export_tt_env
ensure_base_deps
apply_cmake_fixes
configure_project "$BUILD_TYPE"
build_tt_wavelet_target "$TARGET"

log INFO "Update done. Target $TARGET rebuilt in $BUILD_DIR"
log INFO "To keep env vars in this shell: source $SCRIPT_DIR/scripts/set_env.sh"
