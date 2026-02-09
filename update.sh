#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/scripts/common.sh"

BUILD_TYPE=${1:-Release}
TARGET=${2:-tt_wavelet_test}

log INFO "Updating tt-wavelet only (target: $TARGET)"
ensure_base_deps
run_tt_metal_install_deps
apply_cmake_fixes
configure_project "$BUILD_TYPE"
build_tt_wavelet_target "$TARGET"

log INFO "Update done. Target $TARGET rebuilt in $BUILD_DIR"
