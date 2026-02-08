#!/bin/bash

# Script to clean up old generated protobuf files from source tree
# These files should only exist in the build directory

echo "Cleaning up old generated protobuf files from source tree..."

# Remove any .pb.h and .pb.cc files from source tree
find third-party/tt-metal -name "*.pb.h" -type f -not -path "*/build/*" -not -path "*/.cpmcache/*" -delete
find third-party/tt-metal -name "*.pb.cc" -type f -not -path "*/build/*" -not -path "*/.cpmcache/*" -delete

echo "Done! Old protobuf files removed."
echo ""
echo "Now you can build cleanly:"
echo "  rm -rf build"
echo "  mkdir build"
echo "  cd build"
echo "  cmake .."
echo "  make -j\$(nproc)"
