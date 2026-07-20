# tt-wavelet

`tt-wavelet` implements one-level FP32 lifting wavelet transforms on Tenstorrent Wormhole B0 and Blackhole. Production LWT and ILWT use dependency-local three-slot L1 workspaces, native `32x16` compute pages, and direct terminal DRAM output. All 106 generated schemes are compiled through the local pinned TT-Metal/SFPI toolchain.

## Setup

```bash
git clone --recurse-submodules https://github.com/ke1rro/tt-wavelet.git
cd tt-wavelet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The supported toolchain includes clang/clang++ 20, Ninja, CMake 3.20 or newer, and the TT-Metal revision in the `tt-metal` submodule.

## Build

Use the full build for bootstrap or after changing TT-Metal:

```bash
unset ARCH_NAME
./build.sh Release
```

For ordinary tt-wavelet changes:

```bash
unset ARCH_NAME
./update.sh Release lwt
source ./scripts/set_env.sh
```

`update.sh` reconfigures the project, reapplies the repository's TT-Metal CMake fixes, and rebuilds only the requested tt-wavelet target. A successful host build does not compile SFPI; run a device command after kernel changes.

## Run

```bash
./build/lwt db7 signal.txt
./build/lwt --boundary-mode periodic db7 signal.txt
./build/lwt --inverse db7 signal.txt
./build/lwt --benchmark --repeats 20 --warmup-runs 5 --length 5000000 db7
```

The default boundary mode is `symmetric`. Other supported modes are `zero`, `constant`, `periodic`, `antisymmetric`, `smooth`, `reflect`, and `antireflect`.

The architecture is detected by TT-Metal. Do not set `ARCH_NAME` on hardware. For controlled layout A/B tests only:

```bash
export TT_WAVELET_LWT_WORKSPACE_LAYOUT=auto
export TT_WAVELET_LWT_WORKSPACE_LAYOUT=row-major
export TT_WAVELET_LWT_WORKSPACE_LAYOUT=tile-native
```

## Validate

Activate the repository virtual environment before Python validation:

```bash
source .venv/bin/activate
source ./scripts/set_env.sh
python3 compare.py --wavelet db1 --tolerance 1e-5
python3 compare.py --wavelet db7 --tolerance 1e-3
python3 scripts/validate_lwt_boundaries.py
python3 scripts/validate_ilwt.py
python3 scripts/validate_ilwt_stability.py
python3 scripts/validate_synthetic_k17.py
python3 scripts/validate_ilwt_geometry.py
python3 scripts/check_ncrisc_elf_size.py --architecture wormhole_b0
```

## Documentation

- [Production LWT and ILWT](docs/LWT.md)
- [Scheduling, layouts, and L1 accounting](docs/LWT_MEMORY_MODES.md)
- [Native narrow-tile workspace](docs/LWT_TILE_NATIVE_OPTIMIZATION.md)
- [Horizontal SFPU stencil](docs/HORIZONTAL_STENCIL.md)
- [Vertical SFPU stencil](docs/VERTICAL_STENCIL.md)
- [Definitions](docs/DEFINITIONS.md)
