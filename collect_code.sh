#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
OUT="${2:-tt-wavelet_all_code.txt}"

find "$ROOT" \
  \( \
    -path '*/.git' -o \
    -path '*/build' -o \
    -path '*/cmake-build-*' -o \
    -path '*/kernels/generated' -o \
    -path '*/kernels/compute/generated' -o \
    -path '*/tt_wavelet/include/schemes' -o \
    -path '*/tt_wavelet/include/schemes/generated' -o \
    -path '*/lifting_schemes' \
  \) -prune -o \
  -type f \
  \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) \
  -print0 |
sort -z |
while IFS= read -r -d '' file; do
  printf "===== %s =====\n\n" "$file"
  cat "$file"
  printf "\n\n"
done > "$OUT"

echo "Written to $OUT"
