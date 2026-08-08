# tt-wavelet

`tt-wavelet` implements one-level FP32 lifting wavelet transforms on Tenstorrent Wormhole  and Blackhole.

## Setup

```bash
git clone --recurse-submodules https://github.com/ke1rro/tt-wavelet.git
cd tt-wavelet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build

`build.sh` uses the TT-Metal revision pinned in the `tt-metal` submodule and
builds the complete local stack:

- TT-Metal and the TTNN Python bindings;
- TTNN-Wavelet, linked into TT-Metal from this repository's single
  `ttnn-wavelet` source tree; and
- the standalone `lwt`, `ilwt`, `lwt_2d`, `ilwt_2d`, and benchmark binaries.

On a new machine, install TT-Metal's system and Python dependencies first:

```bash
./build.sh --bootstrap
```

For a normal incremental build:

```bash
./build.sh
```

To rebuild one target after a focused source change, pass its name:

```bash
./build.sh lwt
./build.sh --target ttnn --jobs $(nproc)
```

Supported targets are:

- `ttnn` – TTNN runtime, Python bindings, and the linked TTNN-Wavelet operation.
- `lwt` – standalone forward 1D lifting wavelet transform.
- `ilwt` – standalone inverse 1D lifting wavelet transform.
- `lwt_2d` – standalone forward 2D lifting wavelet transform.
- `ilwt_2d` – standalone inverse 2D lifting wavelet transform.
- `tt_wavelet_benchmark_runner` – standalone benchmark runner used by the benchmark scripts.

```bash
./build.sh --jobs 16
./build.sh --type Debug
```

After a build, enable the local runtime before running a binary or importing
TTNN:

```bash
source ./scripts/set_env.sh
```
