# TT-Wavelet

TT-Wavelet implements discrete wavelet transforms on Tenstorrent accelerators using lifting-scheme
factorizations. It provides one-level forward and inverse 1D and 2D DWT operations for TTNN tensors
through the `ttwt` Python package.

Wavelet FIR filter banks are represented as sequences of lifting steps derived from
[Lifting Factorization of PyWavelets filter banks](https://github.com/draklowell/lifting-factorization).
TT-Wavelet consumes the resulting decompositions; its runtime does not embed that project's
Python/SageMath factorization pipeline.

## Features

- One-level 1D DWT / IDWT and 2D DWT / IDWT on device.
- 106 discrete wavelet schemes from the [PyWavelets](https://github.com/PyWavelets/pywt) catalog.
- Eight boundary modes: `zero`, `constant`, `symmetric`, `reflect`, `periodic`, `smooth`,
  `antisymmetric`, and `antireflect`.
- Ordinary TTNN tensors as inputs and outputs; direct `MemoryConfig` interoperability.
- Package-local Wavelet JIT resources; TTNN / Metalium remain external dependencies.

TT-Wavelet is numerically compatible with PyWavelets for the supported one-level 1D and 2D DWT/IDWT
operations, with the `dmey` limitation below. The project catalog covers all 106 discrete wavelets
and all eight supported boundary modes. Compatibility refers to numerical transform behavior,
wavelet definitions, and boundary handling, not the full PyWavelets Python API. `periodic` is not
PyWavelets' `periodization`, which is unsupported.

**`dmey` is numerically unstable:** its lifting factorization has a known error and it is excluded
from numerical validation. See the
[factorization results](https://github.com/draklowell/lifting-factorization#3-factoring-results).
PyWavelets describes `dmey` as a
[discrete FIR approximation of Meyer](https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html)
and also omits it from its
[multilevel accuracy tests](https://github.com/PyWavelets/pywt/blob/main/pywt/tests/test_multilevel.py#L42-L46)
because its accuracy is very low.

## Requirements

- Ubuntu 22.04 LTS, x86_64, CPython 3.10.
- `ttnn==0.79.0`; software baseline: TTNN / TT-Metal v0.79.0.
- Tenstorrent hardware with compatible driver and firmware.
- **SFPI 7.78.0 [935]**, available from the
  [official compiler release](https://github.com/tenstorrent/sfpi/releases/tag/7.78.0).

The official TTNN pip wheel does not bundle SFPI. A matching compiler installation must be available
to TTNN at runtime, under `ttnn/runtime/sfpi` or `/opt/tenstorrent/sfpi`. TT-Wavelet does not
download or bundle it.

## Installation

### PyPI (available after the first release)

The package has not yet been published. After publication:

```bash
pip install tt-wavelet
```

The package declares `ttnn==0.79.0` as a dependency; pip installs that runtime provider. Driver,
firmware, and SFPI provisioning remain separate prerequisites.

### Development

The SUBMODULE build uses the pinned TT-Metal / TTNN v0.79.0 source baseline. Linux build
prerequisites include clang-20, the GCC 12 C++ development toolchain, Ninja, pkg-config, and the
`libhwloc-dev`, `libnuma-dev`, and `libcapstone-dev` packages.

```bash
git clone --recurse-submodules https://github.com/ke1rro/tt-wavelet.git
cd tt-wavelet
```

For an existing clone:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Build and install a local wheel:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install build wheel cmake==4.2.3

CC=clang-20 \
CXX=clang++-20 \
CMAKE_BUILD_PARALLEL_LEVEL=2 \
python -m build --wheel \
  -Ccmake.define.TT_WAVELET_TT_METAL_MODE=SUBMODULE \
  -Cbuild-dir=build

python -m pip install dist/*.whl
```

## Quick start

The example also requires PyTorch. The 1D API returns stick-padded tensors; only the requested
logical length of the reconstruction is valid.

```python
import torch
import ttnn
import ttwt

device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=[0]
)
try:
    signal = torch.arange(64, dtype=torch.float32)
    x = ttnn.from_torch(
        signal, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    cA, cD = ttwt.dwt(x, "db4")
    reconstructed = ttwt.idwt(cA, cD, "db4", original_length=signal.numel())
    result = ttnn.to_torch(reconstructed).flatten()[: signal.numel()]

    # 2D inputs use tile layout; the inverse takes the original logical shape.
    image = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
    x2d = ttnn.from_torch(
        image, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    ll, lh, hl, hh = ttwt.dwt_2d(x2d, "db4")
    reconstructed = ttwt.idwt_2d(ll, lh, hl, hh, "db4", output_shape=image.shape)
finally:
    ttnn.close_mesh_device(device)
```

## API

- `ttwt.dwt_coeff_len`
- `ttwt.dwt`
- `ttwt.idwt`
- `ttwt.dwt_2d`
- `ttwt.idwt_2d`

## Hardware support

TT-Wavelet has been validated across all currently supported Wormhole and Blackhole SKUs targeted by
TTNN 0.79.0.

Regular CI checks builds, packaging, and host-side tests without an accelerator. Hardware
qualification is performed separately and is required before a public release is tagged; it is not
guaranteed for every commit or pull request.

## Acknowledgements

The original implementation of TT-Wavelet was developed through the
[Tenstorrent Bounty Program](https://github.com/tenstorrent/tt-metal/issues/40494). We thank
Tenstorrent for sponsoring the work and supporting its development and hardware validation.

## References

- [Lifting Factorization of PyWavelets filter banks](https://github.com/draklowell/lifting-factorization).
- [PyWavelets](https://github.com/PyWavelets/pywt).
- [PyWavelets: A Python package for wavelet analysis](https://doi.org/10.21105/joss.01237), *Journal
  of Open Source Software*.
- [Tenstorrent TT-Metal / TTNN](https://github.com/tenstorrent/tt-metal).
- [Original LWT / ILWT bounty](https://github.com/tenstorrent/tt-metal/issues/40494).

## Citation

Please cite this software using [CITATION.cff](CITATION.cff). TT-Wavelet authors: Nikita Lenyk and
Andrii Kryvyi. The references above credit the separate factorization and PyWavelets work.

## License

TT-Wavelet is licensed under the [Apache License 2.0](LICENSE). Third-party and derived material
retains its applicable licenses and notices; see [licenses/](licenses/) and [NOTICE](NOTICE).
