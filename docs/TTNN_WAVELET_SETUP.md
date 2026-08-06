# TTNN Wavelet Setup & Integration Guide

This guide explains how to set up, integrate, build, and benchmark `ttnn-wavelet` operations inside the `tt-metal` repository.

---

## 1. Quick Setup (Automated)

Run the provided setup script from the root of `tt-wavelet`:

```bash
source scripts/set_env.sh
./scripts/setup_ttnn_wavelet_in_ttmetal.sh
```

This script will automatically:
1. Export environment variables (`TT_METAL_HOME`, `TT_METAL_ROOT`, `LD_LIBRARY_PATH`, `PYTHONPATH`).
2. Apply integration hooks to `tt-metal` CMake and Python bindings.
3. Symlink `ttnn-wavelet/ttnn/cpp/ttnn/operations/wavelet` into `tt-metal`.
4. Symlink `ttnn-wavelet/tests/ttnn/unit_tests/operations/wavelet` into `tt-metal`.
5. Compile the C++ tree using `cmake --build build -j$(nproc)`.
6. Update Python virtualenv binary bindings (`_ttnn.so` and `_ttnncpp.so`).

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
Apply `ttnn-wavelet/patches/integration-hooks.patch` or manually add:
1. In `tt-metal/ttnn/cpp/ttnn/operations/CMakeLists.txt`:
   ```cmake
   add_subdirectory(wavelet)
   ```
2. In `tt-metal/ttnn/cpp/ttnn/nanobind/__init__.cpp`:
   ```cpp
   #include "ttnn/operations/wavelet/wavelet_pybind.hpp"
   // Inside bind_registered_operations:
   ttnn::operations::wavelet::py_module(m_wavelet);
   ```

### Step 3: Compile C++ Targets
Build the TTNN shared library:

```bash
cd /path/to/tt-wavelet
cmake --build build -j$(nproc)
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
