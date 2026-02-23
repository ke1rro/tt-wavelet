#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TT_METAL_DIR="$ROOT_DIR/tt-metal"

log() {
  local level="$1"; shift
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$level] $*" >&2
}

log INFO "Building Tracy GUI with enchantum support"

# Find enchantum include path
ENCHANTUM_INCLUDE=""
if [[ -d "$TT_METAL_DIR/.cpmcache/enchantum" ]]; then
  ENCHANTUM_INCLUDE=$(find "$TT_METAL_DIR/.cpmcache/enchantum" -type d -name "include" | head -1)
fi

if [[ -z "$ENCHANTUM_INCLUDE" ]]; then
  log ERROR "Enchantum not found in .cpmcache"
  log ERROR "Run './scripts/build_with_tracy.sh' first to download enchantum via CPM"
  exit 1
fi

log INFO "Found enchantum at: $ENCHANTUM_INCLUDE"

# Build Tracy GUI
TRACY_BUILD_DIR="$TT_METAL_DIR/tt_metal/third_party/tracy/profiler/build/unix"
TRACY_MAKEFILE="$TRACY_BUILD_DIR/build.mk"

if [[ ! -f "$TRACY_MAKEFILE" ]]; then
  log ERROR "Tracy Makefile not found at $TRACY_MAKEFILE"
  exit 1
fi

# Backup original Makefile
MAKEFILE_BACKUP="$TRACY_MAKEFILE.backup"
cp "$TRACY_MAKEFILE" "$MAKEFILE_BACKUP"
log INFO "Created Makefile backup: $MAKEFILE_BACKUP"

# Patch Makefile to add enchantum include path
log INFO "Patching Makefile to add enchantum include path..."
sed -i "s|^INCLUDES :=|INCLUDES := -I${ENCHANTUM_INCLUDE} |" "$TRACY_MAKEFILE"

# Verify patch
if grep -q "enchantum" "$TRACY_MAKEFILE"; then
  log INFO "Makefile patched successfully"
else
  log ERROR "Failed to patch Makefile"
  mv "$MAKEFILE_BACKUP" "$TRACY_MAKEFILE"
  exit 1
fi

log INFO "Building Tracy GUI..."
cd "$TRACY_BUILD_DIR"
make -j"$(nproc)"

# Restore original Makefile
log INFO "Restoring original Makefile..."
mv "$MAKEFILE_BACKUP" "$TRACY_MAKEFILE"

TRACY_EXECUTABLE="$TRACY_BUILD_DIR/Tracy-release"
if [[ ! -f "$TRACY_EXECUTABLE" ]]; then
  log ERROR "Tracy GUI build failed - executable not found"
  exit 1
fi

log INFO "Tracy GUI built successfully: $TRACY_EXECUTABLE"

# Create symlink in project root
SYMLINK_PATH="$ROOT_DIR/tracy-gui"
if [[ -L "$SYMLINK_PATH" ]] || [[ -e "$SYMLINK_PATH" ]]; then
  rm -f "$SYMLINK_PATH"
fi

ln -sf "$TRACY_EXECUTABLE" "$SYMLINK_PATH"
log INFO "Created symlink: $SYMLINK_PATH -> $TRACY_EXECUTABLE"

log INFO "Tracy GUI build complete!"
log INFO "Run with: ./tracy-gui"
