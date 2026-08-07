# TTNN Wavelet Setup & Integration Guide

This guide explains how to set up, integrate, build, and benchmark `ttnn-wavelet` operations inside the `tt-metal` repository.

---

## 1. Quick Setup (Automated)

Run the unified wrapper from the root of `tt-wavelet`:

```bash
./run_bringup_benchmark.sh --setup-only
```

This script will automatically:
1. Check out the exact `tt-metal` gitlink pinned by `tt-wavelet`, initialize
   missing submodules, and install system/Python dependencies.
2. Register Wavelet structurally in TT-Metal CMake and nanobind files.
3. Symlink the Wavelet operation and tests into TT-Metal.
4. Configure a Tracy-enabled CMake build and build
   `tt_wavelet_benchmark_runner` and `ttnn`.
5. Update local Python binary bindings (`_ttnn.so` and `_ttnncpp.so`).
6. Reject profiler-disabled or server-wide TT-Metal/UMD/Tracy libraries before
   any hardware benchmark starts.

The registration editor is idempotent and repairs misplaced hooks left by the
old line-number-based patch. The wrapper will not discard a dirty TT-Metal
checkout when its revision differs from the pin; save or clean that submodule
first.

---

## 2. Manual Integration Steps (If Porting to New tt-metal Tree)

If you need to transfer `ttnn-wavelet` into a separate or updated `tt-metal` repository manually, follow these steps:

### Step 1: Copy/Symlink C++ Operations
Symlink or copy the `wavelet` operations folder into `tt-metal`:

```bash
ln -sfn /path/to/tt-wavelet/ttnn-wavelet/ttnn/cpp/ttnn/operations/wavelet \
       /path/to/tt-metal/ttnn/cpp/ttnn/operations/wavelet
```

### Step 2: Update CMake & Nanobind Registration
Run the structure-aware registration editor. Do not apply the historical
zero-context patch to a different TT-Metal revision:

```bash
python3 scripts/integrate_ttnn_wavelet.py --tt-metal /path/to/tt-metal
```

### Step 3: Compile C++ Targets
Build the TTNN shared library:

```bash
cd /path/to/tt-wavelet
cmake --build build --target tt_wavelet_benchmark_runner ttnn --parallel $(nproc)
```

### Step 4: Symlink Built Python Libraries
Symlink the built C++ shared libraries into Python:

```bash
ln -sf /path/to/tt-wavelet/build/tt-metal/ttnn/_ttnn.so /path/to/tt-metal/ttnn/ttnn/_ttnn.so
ln -sf /path/to/tt-wavelet/build/tt-metal/ttnn/_ttnncpp.so /path/to/tt-metal/ttnn/ttnn/_ttnncpp.so
```

---

## 3. Running Precision Verification (All 106 Schemes + All Boundary Modes)

To run the complete numerical precision suite comparing **PyWavelets**, **Standalone tt-wavelet**, and **TTNN ttnn-wavelet** across all 106 schemes and 8 boundary modes:

```bash
source scripts/set_env.sh
python3 scripts/validate_all_106_schemes_precision.py
```

This generates:
- `precision_results.json`: Full detailed metric report per wavelet and boundary mode.
- `precision_results.tsv`: Summary table for easy inspection.

---

## 4. Running Performance Benchmarks (Timing Sweep)

To run the timing sweep modeled after overnight benchmark runs:

```bash
source scripts/set_env.sh
python3 scripts/benchmark_ttnn_vs_standalone_vs_pywt.py \
    --length-start 10000 \
    --length-stop 100000 \
    --length-step 30000 \
    --output-dir docs/overnight_boundary_timings_ttnn
```

This generates:
- Per-wavelet timing log files (`docs/overnight_boundary_timings_ttnn/<wavelet>/<wavelet>_timings.log`).
- `summary.tsv`: Master timing summary comparing PyWavelets, Standalone, and TTNN.
