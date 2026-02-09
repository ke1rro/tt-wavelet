#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TT_METAL_DIR="$ROOT_DIR/tt-metal"

log() { printf "[%s] %s\n" "$1" "$2"; }

if [[ ! -d "$TT_METAL_DIR/.git" ]]; then
  log ERROR "tt-metal does not look like a git checkout at $TT_METAL_DIR" >&2
  exit 1
fi

log INFO "Reverting tt-metal CMakeLists.txt files to submodule state"
git -C "$TT_METAL_DIR" checkout -- tt_metal/fabric/CMakeLists.txt tools/scaleout/CMakeLists.txt
log INFO "Done. Submodule clean status:"
git -C "$TT_METAL_DIR" status --short | sed 's/^/[TT-METAL] /'
