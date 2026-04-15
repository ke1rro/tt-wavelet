#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/common.sh"

BUILD_TYPE=${1:-Release}
# Parallel jobs; default to nproc if not provided
JOBS=${2:-$(nproc)}

log INFO "Starting full build (tt-metal + tt-wavelet) with Tracy enabled"
log INFO "Build configuration: ${BUILD_TYPE}, parallel jobs: ${JOBS}"

export_tt_env
ensure_base_deps
run_tt_metal_install_deps
apply_cmake_fixes

# Override configure_project to add Tracy flag
configure_project_with_tracy() {
  local build_type="$1"
  local generator
  generator=$(select_generator)

  log INFO "Configuring CMake ($generator, ${build_type}) with ENABLE_TRACY=ON"
  cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
    -G "$generator" \
    -DCMAKE_BUILD_TYPE="$build_type" \
    -DCMAKE_C_COMPILER=clang-20 \
    -DCMAKE_CXX_COMPILER=clang++-20 \
    -DBUILD_TT_WAVELET=ON \
    -DENABLE_TRACY=ON \
    -DCMAKE_DISABLE_PRECOMPILE_HEADERS=TRUE \
    -DENABLE_CCACHE=TRUE \
    -DTT_UNITY_BUILDS=ON \
    -DWITH_PYTHON_BINDINGS=ON \
    -DCMAKE_CXX_FLAGS="-std=c++20"
}

configure_project_with_tracy "$BUILD_TYPE"
cmake --build "$BUILD_DIR" -j"${JOBS}"

log INFO "Installing tt-metal Python bindings into the venv"
activate_venv
python -m pip install -e "$TT_METAL_DIR"

log INFO "Build complete. Outputs in $BUILD_DIR"
log INFO ""
log INFO "Next steps:"
log INFO "  1. Build Tracy GUI: cd tt-metal/tt_metal/third_party/tracy/profiler/build/unix && make -j${JOBS}"
log INFO "  2. See TRACY_SETUP.md for usage instructions"
log INFO ""
log INFO "To keep env vars in this shell: source $SCRIPT_DIR/set_env.sh"
