#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TT_METAL_DIR="$ROOT_DIR/tt-metal"

log() { printf "[%s] %s\n" "$1" "$2"; }

# Accept both .git directories and gitfile pointers (common for submodules)
if ! git -C "$TT_METAL_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  log ERROR "tt-metal is not a git worktree at $TT_METAL_DIR (did you init submodules?)." >&2
  exit 1
fi

restore_file() {
  local rel="$1"
  local dst="$TT_METAL_DIR/$rel"
  if ! git -C "$TT_METAL_DIR" show "HEAD:$rel" > "$dst"; then
    log ERROR "Failed to restore $rel from HEAD" >&2
    exit 1
  fi
}

log INFO "Reverting tt-metal CMakeLists.txt files to submodule HEAD"
restore_file "tt_metal/fabric/CMakeLists.txt"
restore_file "tools/scaleout/CMakeLists.txt"

# Confirm files now match HEAD
if git -C "$TT_METAL_DIR" diff --quiet -- tt_metal/fabric/CMakeLists.txt tools/scaleout/CMakeLists.txt; then
  log INFO "CMake files restored to HEAD."
else
  log ERROR "CMake files still differ from HEAD; please check manually." >&2
  exit 1
fi

log INFO "Done. Submodule clean status:"
git -C "$TT_METAL_DIR" status --short | sed 's/^/[TT-METAL] /'
