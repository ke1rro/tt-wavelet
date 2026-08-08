#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/scripts/common.sh"

BUILD_TYPE="Release"
JOBS="${TT_WAVELET_BUILD_JOBS:-$(nproc)}"
BOOTSTRAP=false
TARGET=""
ALL_TARGETS=(ttnn lwt ilwt lwt_2d ilwt_2d tt_wavelet_benchmark_runner)

usage() {
  cat <<'EOF'
Usage: ./build.sh [--bootstrap] [--jobs N] [--type TYPE] [target]

Builds TT-Metal, TTNN with the out-of-tree Wavelet operation symlinked in,
and all standalone tt-wavelet executables in build/.

  --bootstrap      Install TT-Metal and Python dependencies first
  --jobs N         Parallel build jobs (default: TT_WAVELET_BUILD_JOBS or nproc)
  --type TYPE      CMake build type (default: Release)
  --target TARGET  Build one target; equivalent to passing TARGET positionally

Targets: ttnn, lwt, ilwt, lwt_2d, ilwt_2d, tt_wavelet_benchmark_runner
Without a target, builds all targets above.
EOF
}

is_supported_target() {
  local candidate="$1"
  local known_target
  for known_target in "${ALL_TARGETS[@]}"; do
    [[ "$candidate" == "$known_target" ]] && return 0
  done
  return 1
}

set_target() {
  local candidate="$1"
  [[ -z "$TARGET" ]] || { log ERROR "Only one target can be specified" >&2; exit 2; }
  is_supported_target "$candidate" || {
    log ERROR "Unknown target: $candidate" >&2
    usage >&2
    exit 2
  }
  TARGET="$candidate"
}

while (($#)); do
  case "$1" in
    --bootstrap)
      BOOTSTRAP=true
      shift
      ;;
    --jobs)
      [[ ${2-} =~ ^[1-9][0-9]*$ ]] || { log ERROR "--jobs requires a positive integer" >&2; exit 2; }
      JOBS=$2
      shift 2
      ;;
    --type)
      [[ -n ${2-} ]] || { log ERROR "--type requires a value" >&2; exit 2; }
      BUILD_TYPE=$2
      shift 2
      ;;
    --target)
      [[ -n ${2-} ]] || { log ERROR "--target requires a target name" >&2; exit 2; }
      set_target "$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      log ERROR "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      set_target "$1"
      shift
      ;;
  esac
done

if [[ $BOOTSTRAP == true ]]; then
  ensure_base_deps
  run_tt_metal_install_deps
else
  export_tt_env
fi

"$SCRIPT_DIR/scripts/setup_ttnn_wavelet_in_ttmetal.sh" --skip-build
configure_project "$BUILD_TYPE"

if [[ -n "$TARGET" ]]; then
  log INFO "Building target $TARGET (jobs=$JOBS)"
  cmake --build "$BUILD_DIR" --parallel "$JOBS" --target "$TARGET"
else
  log INFO "Building TT-Metal, TTNN-Wavelet, and standalone tt-wavelet (jobs=$JOBS)"
  cmake --build "$BUILD_DIR" --parallel "$JOBS" --target "${ALL_TARGETS[@]}"
fi

"$SCRIPT_DIR/scripts/setup_ttnn_wavelet_in_ttmetal.sh" --skip-build
log INFO "Build complete. Outputs are in $BUILD_DIR"
log INFO "To use the local runtime: source $SCRIPT_DIR/scripts/set_env.sh"
