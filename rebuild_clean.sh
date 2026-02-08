#!/bin/bash

# Script to clean and rebuild tt-wavelet project
# Use this when you get protobuf version errors

set -e  # Exit on error

PROJECT_ROOT="/root/tt-wavelet"

echo "================================================"
echo "Cleaning up old generated protobuf files..."
echo "================================================"

# Remove any .pb.h and .pb.cc files from source tree (but not from build or cache)
find "${PROJECT_ROOT}/third-party/tt-metal" -name "*.pb.h" -type f \
  -not -path "*/build/*" \
  -not -path "*/.cpmcache/*" \
  -print -delete 2>/dev/null || true

find "${PROJECT_ROOT}/third-party/tt-metal" -name "*.pb.cc" -type f \
  -not -path "*/build/*" \
  -not -path "*/.cpmcache/*" \
  -print -delete 2>/dev/null || true

echo ""
echo "================================================"
echo "Removing old build directory..."
echo "================================================"

if [ -d "${PROJECT_ROOT}/build" ]; then
    rm -rf "${PROJECT_ROOT}/build"
    echo "Removed ${PROJECT_ROOT}/build"
else
    echo "No build directory found"
fi

echo ""
echo "================================================"
echo "Creating fresh build directory..."
echo "================================================"

mkdir -p "${PROJECT_ROOT}/build"
cd "${PROJECT_ROOT}/build"

echo ""
echo "================================================"
echo "Creating toolchain directories..."
echo "================================================"

# Create toolchain directories that CMake expects in source tree
mkdir -p "${PROJECT_ROOT}/third-party/tt-metal/runtime/hw/toolchain/wormhole"
mkdir -p "${PROJECT_ROOT}/third-party/tt-metal/runtime/hw/toolchain/blackhole"
mkdir -p "${PROJECT_ROOT}/third-party/tt-metal/runtime/hw/toolchain/grayskull"
echo "Created toolchain directories"

echo ""
echo "================================================"
echo "Running CMake..."
echo "================================================"

cmake -DBUILD_TT_WAVELET=ON ..

echo ""
echo "================================================"
echo "Building project..."
echo "================================================"

# Detect number of cores
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "Using ${NPROC} parallel jobs..."

make -j${NPROC}

echo ""
echo "================================================"
echo "Build completed successfully!"
echo "================================================"
