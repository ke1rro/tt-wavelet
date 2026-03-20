
## Table of Contents

- [Clone the repository](#clone-the-repository)
- [Prerequisites](#prerequisites)
- [Environment setup](#environment-setup)
- [Build & update scripts](#build--update-scripts)
- [Environment variables](#environment-variables)
- [Running](#running)
- [Koyeb ssh](#koyeb-ssh)

# Clone the repository

```bash
git clone --recurse-submodules https://github.com/ke1rro/tt-wavelet.git
cd tt-wavelet
```

# Prerequisites

- clang-20 / clang++-20 (used for all builds)
- lld-20 (linker) and ninja (preferred generator) if available
- CMake 3.20+
- Python 3.9.6+

# Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

# Build & update scripts

These scripts live in the repo root and keep tt-metal patched with fixed CMake files during every run.

| Script | Purpose | Notes |
| --- | --- | --- |
| `./build.sh [Debug\|Release]` | Full build (tt-metal deps + tt-wavelet) | Runs `tt-metal/install_dependencies.sh`, ensures clang-20, sets env vars, applies CMake fixes, configures & builds, installs tt-metal Python bindings into `.venv`. |
| `./update.sh [Debug\|Release] [target]` | Fast rebuild of tt-wavelet only | Re-applies CMake fixes and env; default target `tt_wavelet_test`. Does **not** run tt-metal install_dependencies. |
| `./scripts/revert_tt_metal_cmake.sh` | Restore tt-metal CMakeLists to submodule state | Use before committing if you want a clean submodule. |
| `source ./scripts/set_env.sh` | Persist env vars in current shell | Exports TT_METAL_ROOT/TT_METAL_HOME/TT_METAL_RUNTIME_ROOT and CC/CXX=clang-20. |

Common behavior:
- Exports `TT_METAL_ROOT`, `TT_METAL_HOME`, `TT_METAL_RUNTIME_ROOT` to `$(pwd)/tt-metal`, and `CC/CXX=clang-20/clang++-20`.
- Auto-creates `.venv`, upgrades `pip/setuptools/wheel`, installs `requirements.txt`.
- Applies patched CMake files: `cmakes/CMAKE_FABRIC.txt -> tt-metal/tt_metal/fabric/CMakeLists.txt`, `cmakes/CMAKE_SCALEOUT.txt -> tt-metal/tools/scaleout/CMakeLists.txt`.

# Environment variables

Export these environment variables before running TT-Metal code (scripts set them automatically):

```bash
export TT_METAL_ROOT=$(pwd)/tt-metal
export TT_METAL_HOME=$(pwd)/tt-metal
export TT_METAL_RUNTIME_ROOT=$(pwd)/tt-metal
export CC=clang-20
export CXX=clang++-20
```

Quick way to set them in your shell:

```bash
source ./scripts/set_env.sh
```

# Running

```bash
./build/tt-wavelet/tt_wavelet_test
```

# Koyeb ssh

<https://github.com/koyeb/tenstorrent-examples/tree/main/tt-ssh>

```bash
ssh -p <PORT> root@<IP_ADDRESS>
```

# TT-Metal environment helper

Set the required TT_METAL_* vars from any working directory inside the repo:

- `source scripts/set_env.sh` — exports `TT_METAL_ROOT`, `TT_METAL_HOME`, `TT_METAL_RUNTIME_ROOT`, `CC`, `CXX` (repo root auto-detected via git). Use once per shell/session.
- `scripts/set_env.sh --print` — prints the export lines.
- `scripts/set_env.sh <command>` — runs `<command>` with env pre-set.

Tip: add `source /path/to/repo/scripts/set_env.sh` to your shell rc so the vars are ready in every shell under the repo.
