
## Table of Contents

- [Clone the repository](#clone-the-repository)
- [Prerequisites](#prerequisites)
- [Environment setup](#environment-setup)
- [Build & update scripts](#build--update-scripts)
- [Manual compilation (optional)](#manual-compilation-optional)
- [Environment variables](#environment-variables)
- [Running](#running)
- [Run pre-commit hooks manually](#run-pre-commit-hooks-manually)
- [Submodule commands](#submodule-commands)
- [Koyeb ssh](#koyeb-ssh)

# Clone the repository

```bash
git clone --recurse-submodules https://github.com/ke1rro/tt-wavelet.git
cd tt-wavelet

or
git clone https://github.com/ke1rro/tt-wavelet.git
cd tt-wavelet
git submodule update --init --recursive
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
| `./update.sh [Debug\|Release] [target]` | Fast rebuild of tt-wavelet only | Still re-applies CMake fixes and env; default target `tt_wavelet_test`. Skips rebuilding tt-metal binaries unless dependencies demand it. |
| `./scripts/revert_tt_metal_cmake.sh` | Restore tt-metal CMakeLists to submodule state | Use before committing if you want a clean submodule. |
| `source ./scripts/set_env.sh` | Persist env vars in current shell | Exports TT_METAL_ROOT/TT_METAL_HOME/TT_METAL_RUNTIME_ROOT and CC/CXX=clang-20. |

Common behavior:
- Exports `TT_METAL_ROOT`, `TT_METAL_HOME`, `TT_METAL_RUNTIME_ROOT` to `$(pwd)/tt-metal`, and `CC/CXX=clang-20/clang++-20`.
- Auto-creates `.venv`, upgrades `pip/setuptools/wheel`, installs `requirements.txt`.
- Applies patched CMake files: `CMAKE_FABRIC.txt -> tt-metal/tt_metal/fabric/CMakeLists.txt`, `CMAKE_SCALEOUT.txt -> tt-metal/tools/scaleout/CMakeLists.txt`.

# Manual compilation (optional)

Initialize and update submodules including tt-metal and its dependencies:

```bash
git submodule update --init --recursive
git submodule foreach --recursive 'git lfs fetch --all && git lfs pull'
```

# Compilation without helper scripts (not recommended)

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make -j$(nproc)
```

## With TT-Metal (Wormhole) support

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TT_WAVELET=ON ../
make -j$(nproc)
```

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

# Run pre-commit hooks manually

```bash
pre-commit run --all-files
```

# Submodule commands

```bash
# update submodule to the latest commit on the remote branch
git submodule update --remote

# check status
git submodule status
```

The tt-metal submodule is pinned to the `stable` branch in `.gitmodules`.

>[!WARNING]
>Destruction alert! The following commands will change the structure and all team members will need to use `git submodule update --remote` to update their local submodule to the new branch. Make sure to inform your team about the change.

```bash
# Change submodule branch
cd third-party/tt-metal
git checkout <other-branch>
cd ../..
git add third-party/tt-metal
git commit -m "Update tt-metal to different branch"
```

```bash
[skip ci] maybe used in commit message to skip CI if chore changes only
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
