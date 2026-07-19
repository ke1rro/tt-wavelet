# Production 1D LWT and ILWT

This document is the production contract for the single-level 1D lifting wavelet transform. The implementation supports all 106 generated schemes on Wormhole B0 and Blackhole while retaining FP32 storage and FP32 SFPU arithmetic.

## Public execution model

There is one production implementation. Each worker owns an independent dependency-local chunk, reconstructs the exact input intervals required by that chunk, executes every lifting route through three local `A/B/Scratch` slots, and writes terminal results to DRAM. Slot swaps change descriptors only. Intermediate routes do not make a DRAM round trip and workers do not consume another worker's intermediate state.

The host API is:

```cpp
auto forward = ttwv::create_lwt_executable<Scheme>(
    kernel_root, mesh_device, input_buffer, input_desc, boundary_mode);
ttwv::prepare_lwt(command_queue, forward);
ttwv::execute_lwt(mesh_device, command_queue, forward);

auto inverse = ttwv::create_ilwt_executable<Scheme>(
    kernel_root,
    mesh_device,
    approximation_buffer,
    detail_buffer,
    coefficient_length,
    original_length,
    boundary_mode);
ttwv::prepare_ilwt(command_queue, inverse);
ttwv::execute_ilwt(mesh_device, command_queue, inverse);
```

The CLI has no backend selector:

```bash
source ./scripts/set_env.sh
./build/lwt --boundary-mode symmetric db7 signal.txt
./build/lwt --inverse --boundary-mode symmetric db7 signal.txt
```

Supported boundary policies are `symmetric`, `zero`, `constant`, `periodic`, `antisymmetric`, `smooth`, `reflect`, and `antireflect`. `reflect` and `antireflect` require `N > 1`, matching PyWavelets DWT.

## Arithmetic order and scale path

All route coefficients are compile-time FP32 bit patterns. Predict and update steps use the horizontal SFPU stencil and preserve the existing multiply-accumulate order.

The scale audit found three formerly selectable behaviors: a standalone scale route, forward terminal scaling combined with the final predict/update, and inverse reciprocal scaling combined with the first consuming predict/update. Production now has one compile-time policy per transform:

- forward combines the terminal scale for the updated stream with the last predict/update; if the other terminal stream cannot be reached by that route, its required FP32 scale remains one explicit scale route;
- inverse combines both leading reciprocal scales lazily with the first predict/update that consumes each stream;
- runtime scale environment switches and duplicate kernel variants were removed.

The multiplication still occurs in SFPU before the corresponding stencil arithmetic. Coefficients are not folded, so the FP32 operation order is preserved.

Wormhole scale A/B measurements used 5,000,000 samples, five warmups, and 20 measured runs:

| Scheme | Separate terminal median | Inline terminal median | Decision |
| --- | ---: | ---: | --- |
| `db7` | 24.188 ms | 23.579 ms | retain inline, 2.5% lower latency |
| `bior3.9` | 14.743 ms | 14.816 ms | within run noise; retain the single policy |

The inverse reciprocal-scale path remains inline because the established Wormhole measurements improved `db7` from about 21.035 ms to 19.577 ms and `bior3.9` from about 12.695 ms to 11.375 ms.

## Architecture policy

[`policy.hpp`](../tt-wavelet/tt_wavelet/include/lifting/policy.hpp) is the authoritative host policy. It switches on `mesh_device.arch()`, whose type is the official `tt::ARCH` enum.

| Architecture | Transform | Automatic layout | Scale | Final direct interleave |
| --- | --- | --- | --- | --- |
| Wormhole B0 | LWT | route geometry heuristic | terminal inline | not applicable |
| Wormhole B0 | ILWT | row-major | inverse inline | disabled |
| Blackhole | LWT | route geometry heuristic | terminal inline | not applicable |
| Blackhole | ILWT | tile-native | inverse inline | enabled for tile-native |

`TT_WAVELET_LWT_WORKSPACE_LAYOUT=auto|row-major|tile-native` is an explicit A/B override. It does not select an architecture. Wormhole always disables direct final interleave, including under a forced tile-native layout. Blackhole enables it only when the selected layout is tile-native.

TT-Metal supplies the device JIT macros `ARCH_WORMHOLE` or `ARCH_BLACKHOLE`. The SFPI and LLK forwarding headers branch directly on those official names. No project architecture macro or architecture boolean is passed as a kernel argument. `ARCH_NAME` must remain unset on hardware; TT-Metal reserves it for simulation override.

## ILWT geometry

The inverse planner walks the forward trace backward. For each requested cropped output interval it computes the exact required even and odd coefficient intervals, reverses swaps and route offsets, executes inverse predict/update routes, then interleaves and removes the forward extension. Boundary mode identifies the coefficient convention; inverse arithmetic and crop geometry are otherwise shared.

The final updated stream can be consumed directly from its three output-CB pages when policy enables direct interleave. Otherwise both final streams are read from local workspace. Both paths preserve the same logical output.

## Correctness interpretation

Compatibility with PyWavelets and architecture equivalence are separate questions. High-order lifting factorizations can accumulate different FP32 error from PyWavelets' filter-bank ordering even when the device implementation is internally correct. Do not classify a Blackhole difference as a known baseline failure without identical-input Wormhole evidence.

Use [`capture_architecture_reference.py`](../scripts/capture_architecture_reference.py) to collect that evidence:

```bash
source .venv/bin/activate
source ./scripts/set_env.sh
python3 scripts/capture_architecture_reference.py capture \
  --architecture wormhole --wavelet bior3.9 --seed 20260719 --length 3073 \
  --boundary-mode symmetric --layout auto --output wormhole.json

# On Blackhole, reuse the saved FP32 signal and coefficient bit patterns.
python3 scripts/capture_architecture_reference.py capture \
  --architecture blackhole --inputs-from wormhole.json --layout auto \
  --output blackhole.json

python3 scripts/capture_architecture_reference.py compare \
  --allow-differences wormhole.json blackhole.json
```

The comparison reports bit mismatches, first mismatch, maximum absolute and relative error with indices, and NaN/Inf locations. Blackhole classification remains open until such a same-input capture is collected on Blackhole hardware.

## Validation commands

```bash
source .venv/bin/activate
source ./scripts/set_env.sh
python3 scripts/validate_lwt_boundaries.py
python3 scripts/validate_lwt_boundaries.py --all-schemes --lengths 33 --runtime-only
python3 scripts/validate_ilwt.py --layouts auto row-major tile-native
python3 scripts/validate_ilwt_stability.py
python3 scripts/validate_synthetic_k17.py
python3 scripts/validate_ilwt_geometry.py
python3 scripts/check_ncrisc_elf_size.py --architecture wormhole_b0
```

The device validation scripts run the NCRISC ELF gate after successful hardware cases. See [LWT_MEMORY_MODES.md](LWT_MEMORY_MODES.md) for scheduling and exact L1 accounting and [HORIZONTAL_STENCIL.md](HORIZONTAL_STENCIL.md) for SFPU details.

## Wormhole N150 acceptance results

The production refactor was validated on Wormhole B0 using the local pinned TT-Metal revision. Forward passed 312/312 representative boundary cases with a worst PyWavelets absolute error of `2.98143605e-05`, and all 848/848 scheme/mode runtime-JIT cases completed. A focused 96-case automatic/row-major/tile-native matrix had `max_abs_between_layouts = 0`. A same-input comparison against commit `8e83754c0a88ea61e88a39a58c45b11648c539c8` was bit-identical for forward and inverse across all 106 schemes. At tolerance `1e-2`, the PyWavelets result remains 70 passing and the same 36 high-order FP32 factorization failures.

ILWT passed 432/432 representative cases with `max_abs_between_layouts = 0` and worst PyWavelets absolute error `3.29005435e-05`. The all-scheme runtime sweep passed 106/106. The synthetic `K=17` acceptance scheme passed both layouts for `N=1,2,3,17,31,32,33,3071,3072,3073`; every deterministic round trip had exactly zero error and row-major/tile-native outputs were bit-identical. Host geometry validation covered all 106 schemes and 59,042 chunks.

Final device timings use 5,000,000 samples, five warmups, 20 measured runs, and the device-only timing boundary:

| Transform | Scheme | Selection | Layout | Scale | Direct interleave | Median | p10 | p90 | Baseline comparison |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| LWT | `db7` | auto | row-major | terminal inline | n/a | 23.492 ms | 23.237 ms | 23.619 ms | -0.53% vs 23.617 ms |
| LWT | `bior3.9` | auto | tile-native | terminal inline | n/a | 14.802 ms | 14.723 ms | 14.907 ms | +0.05% vs 14.795 ms |
| ILWT | `db7` | auto | row-major | inverse inline | disabled | 19.525 ms | 19.324 ms | 19.624 ms | +0.20% vs 19.486 ms |
| ILWT | `bior3.9` | auto | row-major | inverse inline | disabled | 11.363 ms | 11.296 ms | 11.454 ms | -27.52% vs 15.677 ms |
| ILWT | `bior3.9` | forced | row-major | inverse inline | disabled | 11.392 ms | 11.330 ms | 11.475 ms | +0.47% vs 11.339 ms |
| ILWT | `bior3.9` | forced | tile-native | inverse inline | disabled | 17.443 ms | 17.343 ms | 17.702 ms | +11.26% vs former 15.677 ms auto-tile policy |

All six runs reported `wormhole_b0`, 64 active cores, five warmups, and 20 measured samples. The final automatic `bior3.9` ILWT median is within 0.26% of the forced row-major run in the same build.
