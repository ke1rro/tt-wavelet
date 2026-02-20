#!/usr/bin/env bash
set -euo pipefail

TT_METAL_DIR="${1:-tt-metal}"
OUT_DIR="${2:-tt-metal-package}"
BUILD_DIR="${TT_METAL_DIR}/build"

if [[ ! -d "${TT_METAL_DIR}" ]]; then
  echo "tt-metal directory not found: ${TT_METAL_DIR}" >&2
  exit 1
fi

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "tt-metal build directory not found: ${BUILD_DIR}" >&2
  exit 1
fi

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}/lib" "${OUT_DIR}/include"

find_in_build() {
  local filename="$1"
  shift

  for candidate in "$@"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  local found
  found="$(find "${BUILD_DIR}" -type f -name "${filename}" | head -n 1 || true)"
  if [[ -n "${found}" ]]; then
    echo "${found}"
    return 0
  fi

  return 1
}

TT_METAL_LIB="$(find_in_build "libtt_metal.so" \
  "${BUILD_DIR}/lib/libtt_metal.so" \
  "${BUILD_DIR}/tt_metal/libtt_metal.so")" || {
  echo "libtt_metal.so not found under ${BUILD_DIR}" >&2
  exit 1
}
cp "${TT_METAL_LIB}" "${OUT_DIR}/lib/"

DEVICE_LIB="$(find_in_build "libdevice.so" \
  "${BUILD_DIR}/lib/libdevice.so" \
  "${BUILD_DIR}/tt_metal/third_party/umd/device/libdevice.so" || true)"
if [[ -n "${DEVICE_LIB}" ]]; then
  cp "${DEVICE_LIB}" "${OUT_DIR}/lib/"
fi

TTNN_SO="$(find_in_build "_ttnn.so" \
  "${BUILD_DIR}/lib/_ttnn.so" \
  "${BUILD_DIR}/ttnn/_ttnn.so" || true)"
if [[ -n "${TTNN_SO}" ]]; then
  cp "${TTNN_SO}" "${OUT_DIR}/lib/"
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
