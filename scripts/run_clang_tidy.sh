#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILE_DB="${ROOT_DIR}/build/ci-clang-tidy/compile_commands.json"

if [[ ! -f "${COMPILE_DB}" || "${ROOT_DIR}/ci/CMakeLists.txt" -nt "${COMPILE_DB}" || \
      "${ROOT_DIR}/scripts/generate_compile_db.sh" -nt "${COMPILE_DB}" ]]; then
  "${ROOT_DIR}/scripts/generate_compile_db.sh"
fi

if ! command -v clang-tidy-hook >/dev/null 2>&1; then
  echo "clang-tidy-hook is unavailable; run this script through pre-commit" >&2
  exit 1
fi

cd "${ROOT_DIR}"
exec clang-tidy-hook "$@"
