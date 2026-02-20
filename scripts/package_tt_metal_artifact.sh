#!/usr/bin/env bash
set -euo pipefail

TT_METAL_DIR="${1:-tt-metal}"
OUT_DIR="${2:-tt-metal-package}"
LIB_DIR="${TT_METAL_DIR}/build/lib"

if [[ ! -d "${TT_METAL_DIR}" ]]; then
  echo "tt-metal directory not found: ${TT_METAL_DIR}" >&2
  exit 1
fi

if [[ ! -d "${LIB_DIR}" ]]; then
  echo "tt-metal library directory not found: ${LIB_DIR}" >&2
  exit 1
fi

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}/lib" "${OUT_DIR}/include"

if [[ ! -f "${LIB_DIR}/libtt_metal.so" ]]; then
  echo "libtt_metal.so not found in ${LIB_DIR}" >&2
  exit 1
fi

cp "${LIB_DIR}/libtt_metal.so" "${OUT_DIR}/lib/"

if [[ -f "${LIB_DIR}/libdevice.so" ]]; then
  cp "${LIB_DIR}/libdevice.so" "${OUT_DIR}/lib/"
fi

if [[ -f "${LIB_DIR}/_ttnn.so" ]]; then
  cp "${LIB_DIR}/_ttnn.so" "${OUT_DIR}/lib/"
fi

INCLUDE_DIRS=(
  "ttnn"
  "ttnn/cpp"
  "ttnn/cpp/ttnn/deprecated"
  "tt_metal/api"
  "tt_metal"
  "tt_metal/include"
  "tt_metal/hostdevcommon/api"
  "tt_metal/third_party/umd/device/api"
  "tt_metal/hw/inc"
  "tt_stl"
  "tt_stl/tt_stl"
)

for rel_dir in "${INCLUDE_DIRS[@]}"; do
  src="${TT_METAL_DIR}/${rel_dir}"
  dst="${OUT_DIR}/include/${rel_dir}"
  if [[ -d "${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    cp -R "${src}" "${dst}"
  fi
done

shopt -s nullglob
NLOHMANN_CANDIDATES=(
  "${TT_METAL_DIR}/build/_deps/nlohmann_json-src/include"
  "${TT_METAL_DIR}/build/_deps/nlohmann_json-src/single_include"
  "${TT_METAL_DIR}/.cpmcache/nlohmann_json"/*/include
  "${TT_METAL_DIR}/.cpmcache/nlohmann_json"/*/single_include
)
for candidate in "${NLOHMANN_CANDIDATES[@]}"; do
  if [[ -d "${candidate}/nlohmann" ]]; then
    cp -R "${candidate}/nlohmann" "${OUT_DIR}/include/nlohmann"
    break
  fi
done
shopt -u nullglob

echo "Packaged tt-metal artifact at ${OUT_DIR}"
