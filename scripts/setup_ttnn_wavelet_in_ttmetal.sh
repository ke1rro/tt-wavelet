#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Idempotently expose the out-of-tree TTNN wavelet operation to TT-Metal.
# Sources stay in ttnn-wavelet; TT-Metal sees them through symbolic links.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
TT_METAL_HOME=$REPO_ROOT/tt-metal
BUILD_DIR=$REPO_ROOT/build
BUILD=true
JOBS=${TT_WAVELET_BUILD_JOBS:-$(nproc)}

usage() {
    cat <<'EOF'
Usage: scripts/setup_ttnn_wavelet_in_ttmetal.sh [--skip-build] [--jobs N]

Creates the TTNN-Wavelet source/test links and applies the required TT-Metal
registration hooks. By default, also builds the standalone runner and TTNN.
EOF
}

while (($#)); do
    case "$1" in
        --skip-build)
            BUILD=false
            shift
            ;;
        --jobs)
            if [[ -z ${2-} || ! ${2-} =~ ^[1-9][0-9]*$ ]]; then
                echo "--jobs requires a positive integer" >&2
                exit 2
            fi
            JOBS=$2
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -d $TT_METAL_HOME/.git && ! -f $TT_METAL_HOME/CMakeLists.txt ]]; then
    echo "TT-Metal checkout is missing at $TT_METAL_HOME" >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 2
fi

operation_source=$REPO_ROOT/ttnn-wavelet/ttnn/cpp/ttnn/operations/wavelet
test_source=$REPO_ROOT/ttnn-wavelet/tests/ttnn/unit_tests/operations/wavelet
for source_dir in "$operation_source" "$test_source"; do
    if [[ ! -d $source_dir ]]; then
        echo "Required TTNN-Wavelet source directory is missing: $source_dir" >&2
        exit 2
    fi
done

nanobind_file=$TT_METAL_HOME/ttnn/cpp/ttnn-nanobind/__init__.cpp
operations_cmake=$TT_METAL_HOME/ttnn/cpp/ttnn/operations/CMakeLists.txt
patch_file=$REPO_ROOT/ttnn-wavelet/patches/integration-hooks.patch
for integration_file in "$nanobind_file" "$operations_cmake" "$patch_file"; do
    if [[ ! -f $integration_file ]]; then
        echo "Required integration file is missing: $integration_file" >&2
        exit 2
    fi
done

count_line() {
    local file=$1
    local pattern=$2
    local count
    count=$(grep -E -c "$pattern" "$file" || true)
    printf '%s\n' "${count:-0}"
}

hook_count=0
hook_count=$((hook_count + $(count_line "$nanobind_file" '^[[:space:]]*#include "ttnn/operations/wavelet/wavelet_nanobind.hpp"[[:space:]]*$')))
hook_count=$((hook_count + $(count_line "$nanobind_file" '^[[:space:]]*wavelet::bind_wavelet_operations\(mod\);[[:space:]]*$')))
hook_count=$((hook_count + $(count_line "$operations_cmake" '^[[:space:]]*add_subdirectory\(wavelet\)[[:space:]]*$')))
hook_count=$((hook_count + $(count_line "$operations_cmake" '^[[:space:]]*TTNN::Ops::Wavelet[[:space:]]*$')))
hook_count=$((hook_count + $(count_line "$operations_cmake" '^[[:space:]]*\$<TARGET_OBJECTS:TTNN::Ops::Wavelet>[[:space:]]*$')))

case $hook_count in
    0)
        echo "Applying TTNN-Wavelet registration hooks"
        git -C "$TT_METAL_HOME" apply --check "$patch_file"
        git -C "$TT_METAL_HOME" apply "$patch_file"
        ;;
    5)
        echo "TTNN-Wavelet registration hooks already present"
        ;;
    *)
        echo "TTNN-Wavelet registration is partial or duplicated ($hook_count/5 hooks)." >&2
        echo "Refusing to modify TT-Metal; inspect $nanobind_file and $operations_cmake" >&2
        exit 2
        ;;
esac

link_tree() {
    local source_path=$1
    local link_path=$2
    mkdir -p "$(dirname "$link_path")"
    if [[ -L $link_path ]]; then
        ln -sfn "$source_path" "$link_path"
    elif [[ -e $link_path ]]; then
        echo "Refusing to replace non-symlink path: $link_path" >&2
        exit 2
    else
        ln -s "$source_path" "$link_path"
    fi
}

link_tree "$operation_source" "$TT_METAL_HOME/ttnn/cpp/ttnn/operations/wavelet"
link_tree "$test_source" "$TT_METAL_HOME/tests/ttnn/unit_tests/operations/wavelet"
echo "TTNN-Wavelet sources linked into TT-Metal (no files moved or copied)"

if [[ $BUILD == true ]]; then
    if [[ ! -f $BUILD_DIR/CMakeCache.txt ]]; then
        echo "Missing configured build tree: $BUILD_DIR" >&2
        echo "Use ./run_bringup_benchmark.sh to bootstrap and build everything." >&2
        exit 2
    fi
    cmake --build "$BUILD_DIR" \
        --target tt_wavelet_benchmark_runner ttnn \
        --parallel "$JOBS"
fi

build_ttnn_dir=$BUILD_DIR/tt-metal/ttnn
python_binding_dir=$TT_METAL_HOME/ttnn/ttnn
if [[ -f $build_ttnn_dir/_ttnn.so ]]; then
    ln -sfn "$build_ttnn_dir/_ttnn.so" "$python_binding_dir/_ttnn.so"
fi
if [[ -f $build_ttnn_dir/_ttnncpp.so ]]; then
    ln -sfn "$build_ttnn_dir/_ttnncpp.so" "$python_binding_dir/_ttnncpp.so"
fi

echo "TTNN-Wavelet integration ready"
