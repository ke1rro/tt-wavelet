#!/bin/bash
set -e

BUILD_TYPE=${1:-Release}
CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== tt-wavelet build (clang-20, ${BUILD_TYPE}) ==="

cmake -B "$CUR_DIR/build" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_C_COMPILER=clang-20 \
    -DCMAKE_CXX_COMPILER=clang++-20 \
    -DBUILD_TT_WAVELET=ON \
    -DCMAKE_DISABLE_PRECOMPILE_HEADERS=TRUE \
    -DENABLE_CCACHE=TRUE \
    -DTT_UNITY_BUILDS=ON \
    -DTT_ENABLE_LIGHT_METAL_TRACE=ON \
    -DWITH_PYTHON_BINDINGS=ON \
    -DCMAKE_CXX_FLAGS="-std=c++20" \
    -S "$CUR_DIR"

cmake --build "$CUR_DIR/build" -j"$(nproc)"

pip3 install -e "$CUR_DIR/tt-metal/"

export TT_METAL_HOME="$CUR_DIR/tt-metal"
echo "=== Done. TT_METAL_HOME=$TT_METAL_HOME ==="
