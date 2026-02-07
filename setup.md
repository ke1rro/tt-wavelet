
## Table of Contents

- [Clone the repository](#clone-the-repository)
- [Prerequisites](#prerequisites)
- [Setup environment](#setup-environment)
- [Build dependencies](#build-dependencies)
- [Compilation](#compilation)
- [Environment variables](#environment-variables)
- [Running](#running)
- [Run pre-commit hooks manually](#run-pre-commit-hooks-manually)
- [Submodule commands](#submodule-commands)
- [Koyeb ssh](#koyeb-ssh)

# Clone the repository

```bash
git clone --recurse-submodules https://github.com/ke1rro/tt-wavelet.git
cd tt-wavelet
```

# Prerequisites

- g++ (with C++20 support)
- CMake 3.20+
- Python 3.9.6+
- Tenstorrent Wormhole hardware (for running on device)

# Setup environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

# Build dependencies

Initialize and update submodules including tt-metal and its dependencies:

```bash
git submodule update --init --recursive
git submodule foreach --recursive 'git lfs fetch --all && git lfs pull'
```

# Compilation

## Build without TT-Metal (for development/testing)

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make -j$(nproc)
```

## Build with TT-Metal (Wormhole) support

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TT_WAVELET=ON ../
make -j$(nproc)
```

# Environment variables

Export these environment variables before running TT-Metal code:

```bash
export TT_METAL_HOME=$(realpath ./third-party/tt-metal/)
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
