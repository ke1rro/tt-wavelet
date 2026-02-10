#!/usr/bin/env bash
# Export TT-Metal env vars relative to repo root, regardless of current directory.
# Usage:
#   source scripts/set_env.sh            # export into current shell
#   scripts/set_env.sh --print           # print exports
#   scripts/set_env.sh <cmd> [args...]   # run command with env set

set -euo pipefail

resolve_root() {
  if root=$(git rev-parse --show-toplevel 2>/dev/null); then
    printf "%s" "$root"
    return
  fi
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  printf "%s" "$(cd "$script_dir/.." && pwd)"
}

ROOT_DIR=$(resolve_root)
export TT_METAL_ROOT="$ROOT_DIR/tt-metal"
export TT_METAL_HOME="$ROOT_DIR/tt-metal"
export TT_METAL_RUNTIME_ROOT="$ROOT_DIR/tt-metal"
export CC=clang-20
export CXX=clang++-20

if [[ ${1-} == "--print" ]]; then
  cat <<ENV
export TT_METAL_ROOT="$TT_METAL_ROOT"
export TT_METAL_HOME="$TT_METAL_HOME"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_RUNTIME_ROOT"
export CC=clang-20
export CXX=clang++-20
ENV
  exit 0
fi

# If executed with a command, run it with env set (does not modify parent shell).
if (( $# > 0 )); then
  "$@"
else
  # If script is sourced, vars persist; if executed directly, this only affects child shell.
  printf "TT env set. TT_METAL_HOME=%s\n" "$TT_METAL_HOME"
  if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf "Note: to persist in your shell, run: source scripts/set_env.sh\n" >&2
  fi
fi
