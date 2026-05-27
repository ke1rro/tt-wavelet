#!/usr/bin/env bash
set -euo pipefail

# Fast rebuild of the lwt target with Tracy enabled.
# Does NOT run tt-metal install_dependencies or touch the full build.
# Assumes the build directory was already configured by ./scripts/build_with_tracy.sh.
# Typical rebuild time: seconds, not 45 minutes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/scripts/common.sh"

BUILD_TYPE=${1:-Release}
TARGET=${2:-lwt}

log INFO "Fast Tracy rebuild (target: $TARGET, no tt-metal install)"
ensure_upd_deps
apply_cmake_fixes
configure_project_tracy "$BUILD_TYPE"
build_tt_wavelet_target "$TARGET"

log INFO "Done. $TARGET rebuilt with Tracy in $BUILD_DIR"
log INFO ""
log INFO "Run with:"
log INFO "  ./build/$TARGET <scheme> <signal_file>"
log INFO ""
log INFO "Offline capture (no GUI needed on device):"
log INFO "  ./tt-metal/tt_metal/third_party/tracy/capture/build/unix/capture-release -o trace.tracy &"
log INFO "  ./build/$TARGET <scheme> <signal_file>  # press Enter when prompted"
