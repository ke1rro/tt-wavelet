#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/common.sh"

BUILD_TYPE=${1:-Release}
JOBS=${2:-18}

log INFO "Quick build (без install_dependencies)"
export_tt_env
apply_cmake_fixes
configure_project "$BUILD_TYPE"

log INFO "✅ Конфігурація завершена! Запускаю збірку з -j${JOBS}"
cmake --build "$BUILD_DIR" -j"${JOBS}"

log INFO "✅ ЗБІРКА ЗАВЕРШЕНА!"
log INFO "Outputs in $BUILD_DIR"
