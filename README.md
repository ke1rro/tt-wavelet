# TT-Wavelet

TT-Wavelet provides standalone lifting wavelet transforms for Tenstorrent TTNN tensors through the
`ttwt` Python package.

## Features

- 1D DWT / IDWT and 2D DWT / IDWT.
- A catalog of 106 wavelet schemes and eight boundary modes.
- Direct TTNN Tensor and MemoryConfig interoperability.
- Package-local runtime JIT resources.

## Status

Version **0.1.0 is a release candidate**, not a published release. The current validation target is
Ubuntu 22.04 x86_64, CPython 3.10 and Blackhole p150b. Other platforms and Wormhole are unvalidated.

Installed-wheel integration and representative db1 transforms have passed hardware validation.
Full-scheme qualification found numerical failures, including `dmey` and boundary extrapolation
cases. The catalog is not a guarantee of uniform numerical accuracy; release qualification remains
open.

## Requirements

- Ubuntu 22.04 LTS, x86_64.
- CPython 3.10.
- `ttnn==0.79.0`.
- Tenstorrent Blackhole hardware with separately provisioned driver and firmware; p150b validated.
- Matching **SFPI 7.78.0 [935]**, provisioned separately through Tenstorrent's
  [official release](https://github.com/tenstorrent/sfpi/releases/tag/7.78.0).

TTNN's pip wheel does not install SFPI. Its compiler lookup checks `ttnn/runtime/sfpi` and
`/opt/tenstorrent/sfpi`. TT-Wavelet neither downloads nor bundles the compiler.

## Installation

Install a locally built candidate wheel in a CPython 3.10 environment:

```bash
python -m pip install /path/to/tt_wavelet-0.1.0-cp310-cp310-linux_x86_64.whl
```

Pip resolves the pinned TTNN runtime. Hardware prerequisites must be provisioned separately.
TT-Wavelet is not yet published on PyPI; public `pip install tt-wavelet` is not currently
advertised. See Development for source builds.

## Quick start

```python
import ttnn
import ttwt

# x is a device-resident FP32 ROW_MAJOR ttnn.Tensor.
cA, cD = ttwt.dwt(x, "db1")
```

## API

- `ttwt.dwt_coeff_len`
- `ttwt.dwt`
- `ttwt.idwt`
- `ttwt.dwt_2d`
- `ttwt.idwt_2d`

Function docstrings describe arguments, boundary modes and output layouts.

## Runtime model

`ttwt` calls its private extension `ttwt._ttwt`, which shares the official TTNN / Metalium runtime.
TT-Wavelet ships its own Wavelet JIT kernels and generated scheme headers. It does not bundle TTNN,
Metalium, UMD, STL or SFPI.

## Hardware validation

Regular CI is intended to be hardware-free; accelerator validation is performed separately.
Published releases must be manually qualified on the advertised hardware. Every commit is not
hardware-tested. No release CI is configured yet.

## Development

Clone with submodules. The build baseline is TT-Metal / TTNN `v0.79.0`; compiler and dependency
prerequisites must be provisioned independently.

```bash
git clone --recurse-submodules https://github.com/ke1rro/tt-wavelet.git
cd tt-wavelet
cmake -S . -B build -G Ninja \
  -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 \
  -DTT_WAVELET_BUILD_PYTHON=ON
cmake --build build -j
PYTHONPATH="$PWD/build/python" python -m pytest tests/python/test_companion_module.py
```

To build a local wheel with the matching external development SDK:

```bash
CMAKE_PREFIX_PATH=/path/to/ttnn-0.79.0-sdk CC=clang-20 CXX=clang++-20 \
  python -m build --wheel
```

The official runtime wheel does not provide development CMake packages. Native implementation is in
`src/ttwt/`, Python bindings in `bindings/`, the Python package in `python/ttwt/`, and tests in
`tests/cpp/` and `tests/python/`. Installed-wheel, resource-failure and full numerical qualification
checks live alongside the development regression tests.

Formatting uses clang-format 19.1.4, cmake-format 0.6.13, Black 26.3.1 and mdformat 0.7.22. Run
Black with `black --check .` and Markdown checks with `mdformat --check README.md`. Apply C++ and
CMake checks only to owned files; exclude `third_party/`, generated outputs and build directories.
Historical research reports are excluded from Markdown formatting.

## License

Imported TT-Metal sources retain Apache-2.0 notices; legacy MIT attribution and bundled third-party
license notices are preserved. See [LICENSE](LICENSE) and [licenses/](licenses/). The root MIT
license does not relicense imported code.
