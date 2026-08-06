# Tenstorrent Blackhole Bringup & Verification Prompt Guide

> **AGENT / TASK DIRECTIVE**: Use this document directly as a prompt and specification guide when bringing up, verifying, and benchmarking **`ttnn-wavelet`** and **`tt-wavelet`** on **Tenstorrent Blackhole (`blackhole`)** architecture.

---

## 1. Project Context & Invariants Reference

Before executing any edits or bringup runs, inspect and adhere to the project directives in [`CONTEXT.md`](file:///home/user/tt-wavelet/CONTEXT.md):
1. **Source of Truth**: Standalone `tt-wavelet` is the algorithmic reference. `ttnn-wavelet` and `tt-wavelet` must remain synchronized in code structure, math flow, SFPI stencils, and planner logic.
2. **Performance Goal**: `ttnn-wavelet` device execution time must achieve parity ($\le 1\%$ difference) with standalone `tt-wavelet`.
3. **Troubleshooting Protocol (No Guessing)**:
   If any hardware failure, hang, NoC alignment error, circular buffer (CB) assertion, or FP32 numerical mismatch occurs during bringup:
   - **Do not guess or make blind trial-and-error edits**.
   - Read the exact runtime log, stack trace, and device watcher output.
   - Inspect the actual local source code (`tt-wavelet`, `ttnn-wavelet`).
   - Consult local ISA documentation in [`tt-isa-documentation`](file:///home/user/tt-wavelet/tt-isa-documentation) (specifically Blackhole vs Wormhole FP/SFPI behavior).
   - Consult local `tt-metal` technical reports and headers in [`tt-metal/tech_reports`](file:///home/user/tt-wavelet/tt-metal/tech_reports) and `tt-metal/tt_metal/hw/ckernels`.

---

## 2. Scope of Bringup: Standalone `tt-wavelet` + `ttnn-wavelet`

Bringup must validate and benchmark **BOTH** implementations across all four transform operations:
- **1D Forward Discrete Wavelet Transform (1D LWT / DWT)**
- **1D Inverse Discrete Wavelet Transform (1D ILWT / IDWT)**
- **2D Forward Discrete Wavelet Transform (2D LWT / DWT)**
- **2D Inverse Discrete Wavelet Transform (2D ILWT / IDWT)**

---

## 3. Input Shape Verification Contracts for `ttnn-wavelet`

`ttnn-wavelet` must seamlessly support and be tested on both **unbatched (normal)** and **batched** input tensor representations without requiring manual host-side reshapes:

| Transform | Tensor Representation | Shape | Layout | Supported Memory |
| :--- | :--- | :--- | :--- | :--- |
| **1D LWT (Normal)** | Unbatched 1D Signal | `[N]` | `ROW_MAJOR` | DRAM / L1 Interleaved |
| **1D LWT (Batched)** | Batched 1D Signal | `[B, 1, 1, N]` | `ROW_MAJOR` | DRAM / L1 Interleaved |
| **1D LWT (Stick-Native)** | Physical 128B Stick Matrix | `[S, 32]` ($S=\lceil N/32 \rceil$) | `ROW_MAJOR` | DRAM / L1 Interleaved |
| **1D ILWT (Normal)** | Unbatched Coefficients | `[Wc]` | `ROW_MAJOR` | DRAM / L1 Interleaved |
| **1D ILWT (Batched)** | Batched Coefficients | `[B, 1, 1, Wc]` | `ROW_MAJOR` | DRAM / L1 Interleaved |
| **2D LWT (Normal)** | Unbatched 2D Image | `[H, W]` | `TILE` (32x32) | DRAM / L1 Interleaved |
| **2D LWT (Batched)** | Batched 2D Image | `[B, 1, H, W]` | `TILE` (32x32) | DRAM / L1 Interleaved |
| **2D ILWT (Normal)** | Unbatched 2D Subbands | `[Hc, Wc]` | `TILE` (32x32) | DRAM / L1 Interleaved |
| **2D ILWT (Batched)** | Batched 2D Subbands | `[B, 1, Hc, Wc]` | `TILE` (32x32) | DRAM / L1 Interleaved |

---

## 4. Blackhole Architectural Considerations

When deploying and validating wavelet transforms on Blackhole (`blackhole`), account for the following hardware-specific specs:

### A. NoC Alignment Requirements
- **Wormhole**: 32-byte DRAM read alignment (`kConfigNocAlignmentBytes = 32`).
- **Blackhole**: **64-byte** DRAM read alignment (`kConfigNocAlignmentBytes = 64`).
- **Implementation**: In [`lwt_config.hpp`](file:///home/user/tt-wavelet/ttnn-wavelet/ttnn/cpp/ttnn/operations/wavelet/device/protocol/lwt_config.hpp) and [`lwt_2d_config.hpp`](file:///home/user/tt-wavelet/ttnn-wavelet/ttnn/cpp/ttnn/operations/wavelet/device/protocol/lwt_2d_config.hpp), all route and chunk configuration metadata buffers use **64-byte alignment** (`kConfigNocAlignmentBytes = 64`), ensuring portable NoC DMA transfers across both architectures.

### B. DRAM Controller Bank Striping & Dual-Mode NoC Addressing
- Blackhole features an expanded DRAM controller architecture.
- **Physical Page Size Invariant**: 1D signals allocated with **128-byte physical DRAM pages** (`[S, 32]` where $S = \lceil N / 32 \rceil$) stripe across all Blackhole DRAM controllers.
- **Dual-Mode NoC Addressing**: [`stick_cache.hpp`](file:///home/user/tt-wavelet/ttnn-wavelet/ttnn/cpp/ttnn/operations/wavelet/device/kernels/primitives/stick_cache.hpp) automatically checks `page_size`:
  - `PagePerStick` (`page_size == 128`): Uses `get_noc_addr(source_page + stick)`.
  - `OffsetWithinPage` (`page_size > 128`): Uses `get_noc_addr(source_page, stick * 128)`.

### C. Tensix Core Grid & Architecture Policy
- [`policy.hpp`](file:///home/user/tt-wavelet/ttnn-wavelet/ttnn/cpp/ttnn/operations/wavelet/planner/policy.hpp) queries `tt::ARCH::Blackhole` at runtime and adapts:
  - Tensix core grid dimensions and harvested core maps.
  - L1 workspace budget (`kDefaultL1SignalBudgetBytes = 768 * 1024`).
  - Active worker core count for multi-chunk partitioning.

### D. Expanded Kernel ELF Capacity & Aggressive Inlining Strategy
- **Blackhole ELF Capacity**: Blackhole significantly expands Tensix instruction RAM (I-RAM) and kernel ELF size limits compared to Wormhole's 16 KiB NCRISC limit.
- **Aggressive Inlining Strategy**:
  - On Blackhole, kernels can perform **more aggressive compile-time inlining** (`ALWI` / `always_inline`) for boundary stencil helpers, horizontal/vertical lifting loops, and scale fusion.
  - Inlining eliminates function-call prologues/epilogues and instruction pipeline stalls, lowering cycle counts on Tensix math threads.
- **Verification of Large Schemes**:
  - Because large schemes (`coif17`, `coif12`, `sym20`, `db38`, `bior6.8`, `dmey`) contain up to 51 predict/update steps and 102 taps, verify that large 1D/2D LWT and ILWT kernels compile cleanly without exceeding I-RAM/L1 limits while benefitting from full inlining performance.

---

## 5. Environment Setup & Git Commit Pinning

### A. Repository & Submodule Commit Pinning
> **CRITICAL BUILD REQUIREMENT**: To prevent C++ header mismatch, API breaking changes, or build failures, the `tt-metal` checkout/submodule must be pinned to the exact tested commit.

- **`tt-wavelet` Root Repository Commit**: `2b384424510d3e8b95a03ab6ab19700ff7b022ca`
- **`tt-metal` Submodule Commit**: `55c1876eb30d464006910fa5f8799795f5a17d2d`

Before building on Blackhole, verify and checkout the pinned commits:
```bash
cd /home/user/tt-wavelet
git checkout 2b384424510d3e8b95a03ab6ab19700ff7b022ca

cd /home/user/tt-wavelet/tt-metal
git checkout 55c1876eb30d464006910fa5f8799795f5a17d2d
```

### B. Environment Configuration
Export the following environment variables before building or running tests:

```bash
export TT_METAL_HOME=/home/user/tt-wavelet/tt-metal
export TT_METAL_ROOT=/home/user/tt-wavelet/tt-metal
export TT_METAL_RUNTIME_ROOT=/home/user/tt-wavelet/tt-metal
export LD_LIBRARY_PATH=$TT_METAL_HOME/build/tt-metal/tt_metal:$TT_METAL_HOME/build/tt-metal/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
```

Or source the repository helper:
```bash
source scripts/set_env.sh
```

### C. Building Standalone and TTNN Operations
Run the automated integration and build script:

```bash
./scripts/setup_ttnn_wavelet_in_ttmetal.sh
```

---

## 6. Precision Verification Protocol (FP32 Numerical Parity)

Precision validation verifies that **`ttnn-wavelet`** (`ttnn.dwt`, `ttnn.idwt`) achieves exact numerical parity with **standalone `tt-wavelet`** (C++ device executable) and **PyWavelets** (CPU reference) across all 106 wavelet schemes.

> **Mandatory Comparison Rule**: Both **`ttnn-wavelet`** and **standalone `tt-wavelet`** must be executed and compared side-by-side using `--backends ttnn standalone pywt`. `ttnn-wavelet` must match standalone `tt-wavelet` outputs within FP32 machine precision.

### A. Running Precision Validation with `--backends`
To run representative wavelets on Blackhole across selected backends:

```bash
python3 scripts/validate_all_106_schemes_precision.py \
    --backends ttnn standalone pywt \
    --schemes db1 bior1.3 bior3.5 sym3 db7 coif17 \
    --boundary-modes symmetric zero
```

To run the full 106-scheme sweep across all boundary modes:

```bash
python3 scripts/validate_all_106_schemes_precision.py --backends ttnn standalone pywt
```

### B. Acceptance Criteria
1. **Pearson Correlation Coefficient (PCC)**: `ttnn-wavelet` vs `standalone tt-wavelet` and PyWavelets must achieve **$\text{PCC} = 1.00000$** ($\ge 0.99999$).
2. **Reconstruction Error**:
   - **Compact Schemes** (`db1`, `bior1.3`, `bior3.5`, `sym3`): $\text{Rec Error} \le 1.5 \times 10^{-6}$.
   - **Medium Schemes** (`db7`, `bior3.9`): $\text{Rec Error} \le 1.0 \times 10^{-4}$.
   - **Large / Sensitive Schemes** (`coif17`, `dmey`): Tolerance scaled proportionally by `max_abs_coeff` ($\text{abs\_tol} = 5 \times 10^{-3} \times \text{max\_coeff}$).

---

## 7. Time & Performance Parity Protocol

Performance bringup requires measuring **pure Tensix hardware kernel execution time**, excluding host Python dispatch latency.

> **Mandatory Parity Rule**: The Tensix hardware execution time of **`ttnn-wavelet`** must match **standalone `tt-wavelet`** within **$\le 1\%$ delta**. Any performance drop in `ttnn-wavelet` relative to standalone `tt-wavelet` is considered a performance regression and must be investigated using NoC trace capture.

### A. Performance Benchmark Script with `--backends`
Run a benchmark sweep on Blackhole comparing PyWavelets, Standalone C++, and TTNN:

```bash
python3 scripts/benchmark_ttnn_vs_standalone_vs_pywt.py \
    --backends ttnn standalone pywt \
    --schemes db1 bior3.9 db7 coif17 \
    --length-start 10000 --length-stop 100000 --length-step 30000 \
    --output-dir benchmarks/results
```

### B. Measuring Hardware Execution Time via TT-Metal Trace
To measure pure device kernel hardware latency without Python host overhead:

```python
import torch, ttnn, time

dev = ttnn.open_device(device_id=0)
N = 100000
s_sticks = (N + 31) // 32

sig = torch.sin(torch.linspace(0, 10 * torch.pi, s_sticks * 32, dtype=torch.float32)).reshape(s_sticks, 32)
inp = ttnn.from_torch(sig, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

# Warmup JIT compilation and program cache
for _ in range(5):
    a, d = ttnn.dwt(inp, "db1", boundary_mode="symmetric")
    r = ttnn.idwt(a, d, "db1", N, boundary_mode="symmetric")
ttnn.synchronize_device(dev)

# Capture TT-Metal Trace
trace_id = ttnn.begin_trace_capture(dev)
a, d = ttnn.dwt(inp, "db1", boundary_mode="symmetric")
r = ttnn.idwt(a, d, "db1", N, boundary_mode="symmetric")
ttnn.end_trace_capture(dev, trace_id)

# Execute trace on Blackhole device
ttnn.synchronize_device(dev)
t0 = time.perf_counter()
iters = 200
for _ in range(iters):
    ttnn.execute_trace(dev, trace_id)
ttnn.synchronize_device(dev)
t1 = time.perf_counter()

device_time_ms = (t1 - t0) * 1000.0 / iters
print(f"Blackhole Hardware Trace Execution Time: {device_time_ms:.4f} ms")
ttnn.release_trace(dev, trace_id)
ttnn.close_device(dev)
```

---

## 8. Step-by-Step Blackhole Bringup Checklist

- [ ] **Step 1: Environment & Architecture Verification**
  - Run `python3 -c "import ttnn; dev = ttnn.open_device(0); print(dev.arch()); ttnn.close_device(dev)"`.
  - Confirm `tt::ARCH::Blackhole` is detected.

- [ ] **Step 2: Standalone C++ Sanity Run (1D & 2D)**
  - Run `build/lwt --benchmark --length 100000 db1`.
  - Run `build/lwt_2d --benchmark --boundary-mode symmetric db1 64 64 /tmp/signal.f32`.
  - Confirm active core count and L1 workspace allocation.

- [ ] **Step 3: TTNN Single-Op Execution (Normal & Batched)**
  - Test 1D DWT/IDWT with shape `[100000]` and batched `[4, 1, 1, 100000]`.
  - Test 2D DWT/IDWT with shape `[256, 256]` and batched `[2, 1, 256, 256]`.
  - Confirm zero JIT compilation errors and valid program caching.

- [ ] **Step 4: Precision Validation Sweep**
  - Run `python3 scripts/validate_all_106_schemes_precision.py --backends ttnn standalone pywt`.
  - Verify `benchmarks/precision_results.json` and `benchmarks/precision_results.tsv`.

- [ ] **Step 5: Hardware Time & Performance Sweep**
  - Run `python3 scripts/benchmark_ttnn_vs_standalone_vs_pywt.py --backends ttnn standalone pywt --output-dir benchmarks/results`.
  - Verify hardware execution time parity across standalone and TTNN.
  - Verify generated PNG charts in `benchmarks/results/charts/`.

- [ ] **Step 6: Large Scheme 1D & 2D LWT/ILWT Verification**
  - Run 1D and 2D DWT and IDWT for large schemes (`coif17`, `coif12`, `sym20`, `db38`, `bior6.8`, `dmey`).
  - Confirm that aggressive compile-time inlining (`ALWI` / `always_inline`) fits within Blackhole's expanded kernel ELF size without compiler errors or I-RAM overflow, achieving maximum execution throughput.

---

## 9. Troubleshooting & Diagnostics

1. **Assertion Failure on NoC Read Alignment**:
   - Check `kConfigNocAlignmentBytes` in `lwt_config.hpp`. Blackhole requires **64-byte alignment** for config buffer NoC transfers.
2. **Performance Drop on Large Signal Sizes**:
   - Verify tensor shape. Canonical shapes `[N]` use 1 DRAM page; input tensors should use stick-native shape `[S, 32]` for 128B DRAM bank striping across all DRAM controllers.
3. **Reconstruction Offset in IDWT Multi-Chunk Mode**:
   - Check `interleave.hpp`. Ensure `signal_base` is computed as `(first_stick + local_stick) * 32` and `signal_index >= output_begin && signal_index < output_end`.
4. **Kernel Binary Size Limits on Large 2D Schemes**:
   - On Wormhole, large 2D schemes (`coif17`, `dmey`) required non-inlined boundary callables to prevent I-RAM overflow. On Blackhole, leverage expanded I-RAM / ELF size for full inlining, but monitor TRISC math binary size during JIT compilation.
5. **Fallback Diagnostic Protocol**:
   - If an unknown hardware hang occurs, run watcher: `TT_METAL_WATCHER=120 python3 <script.py>`.
   - Read local ISA documentation in [`tt-isa-documentation`](file:///home/user/tt-wavelet/tt-isa-documentation).
6. **CMake Error: Target `umd::tt-umd` Not Found**:
   - If CMake fails during configure with `Target "hal_1xx" / "fabric" links to umd::tt-umd but target was not found`, `tt-metal` git submodules (`tt_metal/third_party/umd`) were unpopulated.
   - **Resolution**:
     ```bash
     cd /home/user/tt-wavelet/tt-metal
     git submodule update --init --recursive
     cd /home/user/tt-wavelet
     rm -rf build
     cmake -B build -DBUILD_TT_WAVELET=ON
     ./scripts/setup_ttnn_wavelet_in_ttmetal.sh
     ```
