# TT-Wavelet

TT-Wavelet provides standalone lifting wavelet transforms for Tenstorrent TTNN tensors through the
`ttwt` Python package.

## Features

- 1D DWT / IDWT and 2D DWT / IDWT.
- A catalog of 106 wavelet schemes and eight boundary modes.
- Direct TTNN Tensor and MemoryConfig interoperability.
- Package-local runtime JIT resources.

п## Requirements

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

## License

Imported TT-Metal sources retain Apache-2.0 notices; legacy MIT attribution and bundled third-party
license notices are preserved. See [LICENSE](LICENSE) and [licenses/](licenses/). The root MIT
license does not relicense imported code.
