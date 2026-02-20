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

copy_matching_build_libs() {
  local pattern="$1"
  local found=0

  while IFS= read -r lib_path; do
    cp "${lib_path}" "${OUT_DIR}/lib/$(basename "${lib_path}")"
    found=1
  done < <(find "${BUILD_DIR}" \( -type f -o -type l \) -name "${pattern}" | sort -u)

  if [[ "${found}" -eq 1 ]]; then
    return 0
  fi

  return 1
}

copy_header_tree() {
  local header_dir="$1"
  shift

  for candidate_root in "$@"; do
    if [[ -d "${candidate_root}/${header_dir}" ]]; then
      mkdir -p "$(dirname "${OUT_DIR}/include/${header_dir}")"
      cp -R "${candidate_root}/${header_dir}" "${OUT_DIR}/include/${header_dir}"
      return 0
    fi
  done

  return 1
}

require_header_tree() {
  local header_dir="$1"
  shift

  if ! copy_header_tree "${header_dir}" "$@"; then
    echo "Required header directory '${header_dir}' was not found for packaging" >&2
    exit 1
  fi
}

require_header_file() {
  local header_name="$1"
  shift

  for candidate_root in "$@"; do
    if [[ -f "${candidate_root}/${header_name}" ]]; then
      cp "${candidate_root}/${header_name}" "${OUT_DIR}/include/${header_name}"
      return 0
    fi
  done

  echo "Required header file '${header_name}' was not found for packaging" >&2
  exit 1
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

# Runtime dependencies needed by libtt_metal/libdevice during consumer linking.
copy_matching_build_libs "libtt_stl.so*" || true
copy_matching_build_libs "libtracy.so*" || true
copy_matching_build_libs "libfmt.so*" || true
copy_matching_build_libs "libspdlog.so*" || true

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
REFLECT_CANDIDATES=(
  "${TT_METAL_DIR}/build/_deps/reflect-src"
  "${TT_METAL_DIR}/.cpmcache/reflect"/*
)
require_header_file "reflect" "${REFLECT_CANDIDATES[@]}"

FMT_INCLUDE_ROOTS=(
  "${TT_METAL_DIR}/build/_deps/fmt-src/include"
  "${TT_METAL_DIR}/.cpmcache/fmt"/*/include
)
require_header_tree "fmt" "${FMT_INCLUDE_ROOTS[@]}"

SPDLOG_INCLUDE_ROOTS=(
  "${TT_METAL_DIR}/build/_deps/spdlog-src/include"
  "${TT_METAL_DIR}/.cpmcache/spdlog"/*/include
)
require_header_tree "spdlog" "${SPDLOG_INCLUDE_ROOTS[@]}"

TT_LOGGER_INCLUDE_ROOTS=(
  "${TT_METAL_DIR}/build/_deps/tt-logger-src/include"
  "${TT_METAL_DIR}/.cpmcache/tt-logger"/*/include
)
require_header_tree "tt-logger" "${TT_LOGGER_INCLUDE_ROOTS[@]}"

ENCHANTUM_INCLUDE_ROOTS=(
  "${TT_METAL_DIR}/build/_deps/enchantum-src/enchantum/include"
  "${TT_METAL_DIR}/build/_deps/enchantum-src/include"
  "${TT_METAL_DIR}/.cpmcache/enchantum"/*/enchantum/include
  "${TT_METAL_DIR}/.cpmcache/enchantum"/*/include
  "${TT_METAL_DIR}/.cpmcache/enchantum"/*/single_include
)
require_header_tree "enchantum" "${ENCHANTUM_INCLUDE_ROOTS[@]}"

NLOHMANN_INCLUDE_ROOTS=(
  "${TT_METAL_DIR}/build/_deps/nlohmann_json-src/include"
  "${TT_METAL_DIR}/build/_deps/nlohmann_json-src/single_include"
  "${TT_METAL_DIR}/.cpmcache/nlohmann_json"/*/include
  "${TT_METAL_DIR}/.cpmcache/nlohmann_json"/*/single_include
)
require_header_tree "nlohmann" "${NLOHMANN_INCLUDE_ROOTS[@]}"
shopt -u nullglob

echo "Packaged tt-metal artifact at ${OUT_DIR}"
