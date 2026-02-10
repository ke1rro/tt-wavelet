#!/usr/bin/env bash
# Source this file to export TT-Metal/clang env vars into your current shell
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TT_METAL_ROOT="$ROOT_DIR/tt-metal"
export TT_METAL_HOME="$ROOT_DIR/tt-metal"
export TT_METAL_RUNTIME_ROOT="$ROOT_DIR/tt-metal"
export CC=clang-20
export CXX=clang++-20
printf "TT env set for shell. TT_METAL_HOME=%s\n" "$TT_METAL_HOME"
