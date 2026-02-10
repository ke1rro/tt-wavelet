#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TT_METAL_DIR="$ROOT_DIR/tt-metal"
VENV_DIR="$ROOT_DIR/.venv"
BUILD_DIR="$ROOT_DIR/build"

log() { printf "[%s] %s\n" "$1" "$2"; }

ensure_clang20() {
  if command -v clang-20 >/dev/null 2>&1 && command -v clang++-20 >/dev/null 2>&1; then
    return
  fi

  if command -v apt-get >/dev/null 2>&1; then
    log INFO "Installing clang-20 toolchain via apt-get"
    sudo apt-get update -y
    sudo apt-get install -y clang-20 clang++-20 lld-20
  else
    log ERROR "clang-20 not found and apt-get unavailable. Install clang-20 manually." >&2
    exit 1
  fi
}

activate_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    log INFO "Creating Python venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
}

ensure_python_packages() {
  activate_venv
  log INFO "Upgrading pip/setuptools/wheel"
  python -m pip install --upgrade pip setuptools wheel
  if [[ -f "$ROOT_DIR/requirements.txt" ]]; then
    log INFO "Installing Python requirements"
    python -m pip install -r "$ROOT_DIR/requirements.txt"
  fi
}

run_tt_metal_install_deps() {
  if [[ ! -x "$TT_METAL_DIR/install_dependencies.sh" ]]; then
    log ERROR "tt-metal install_dependencies.sh not found at $TT_METAL_DIR/install_dependencies.sh" >&2
    exit 1
  fi
  log INFO "Running tt-metal/install_dependencies.sh"
  bash "$TT_METAL_DIR/install_dependencies.sh"
}

apply_cmake_fixes() {
  local fabric_dst="$TT_METAL_DIR/tt_metal/fabric/CMakeLists.txt"
  local scaleout_dst="$TT_METAL_DIR/tools/scaleout/CMakeLists.txt"

  [[ -f "$ROOT_DIR/CMAKE_FABRIC.txt" ]] || { log ERROR "CMAKE_FABRIC.txt missing"; exit 1; }
  [[ -f "$ROOT_DIR/CMAKE_SCALEOUT.txt" ]] || { log ERROR "CMAKE_SCALEOUT.txt missing"; exit 1; }

  log INFO "Patching tt-metal CMake (fabric)"
  cp "$ROOT_DIR/CMAKE_FABRIC.txt" "$fabric_dst"

  log INFO "Patching tt-metal CMake (scaleout)"
  cp "$ROOT_DIR/CMAKE_SCALEOUT.txt" "$scaleout_dst"
}

export_tt_env() {
  export TT_METAL_ROOT="$TT_METAL_DIR"
  export TT_METAL_HOME="$TT_METAL_DIR"
  export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
  export CC=clang-20
  export CXX=clang++-20
  log INFO "TT env set: TT_METAL_HOME=$TT_METAL_HOME"
}

select_generator() {
  if command -v ninja >/dev/null 2>&1; then
    echo "Ninja"
  else
    echo "Unix Makefiles"
  fi
}

configure_project() {
  local build_type="$1"
  local generator
  generator=$(select_generator)

  log INFO "Configuring CMake ($generator, ${build_type})"
  cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
    -G "$generator" \
    -DCMAKE_BUILD_TYPE="$build_type" \
    -DCMAKE_C_COMPILER=clang-20 \
    -DCMAKE_CXX_COMPILER=clang++-20 \
    -DBUILD_TT_WAVELET=ON \
    -DCMAKE_DISABLE_PRECOMPILE_HEADERS=TRUE \
    -DENABLE_CCACHE=TRUE \
    -DTT_UNITY_BUILDS=ON \
    -DWITH_PYTHON_BINDINGS=ON \
    -DCMAKE_CXX_FLAGS="-std=c++20"
}

build_tt_wavelet_target() {
  local target="$1"
  log INFO "Building target: $target"
  cmake --build "$BUILD_DIR" --target "$target" -j"$(nproc)"
}

ensure_base_deps() {
  ensure_clang20
  export_tt_env
  ensure_python_packages
}
