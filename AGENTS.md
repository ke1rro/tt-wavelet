Use the `$hpc-modern-cpp` skill for performance-critical C++/SFPI work in this repository.

Work in `/home/user/tt-wavelet`. The product source is under `tt-wavelet/`, the
local TT-Metal checkout is under `tt-metal/`, and repository scripts and reports
are at the root. Do not create a commit unless the user explicitly asks for one.
Preserve unrelated working-tree changes.

The project supports both Wormhole and Blackhole. The current server is a
Blackhole P150b, so device runs in this checkout validate Blackhole only; do not
claim Wormhole equivalence without results from Wormhole hardware or a saved
Wormhole reference.

## Current implementation

Forward LWT has two execution modes:

* `ConeStreamed` (`--memory-mode cone`) is the default scalable backend;
* `ResidentSharded` (`--memory-mode resident`) remains a useful reference where
  the complete three-slot workspace fits in L1.

Both preserve FP32 storage and FP32 SFPU arithmetic. ConeStreamed keeps the
three logical `A/B/Scratch` slots in local L1, performs metadata-only swaps, and
writes terminal results directly to DRAM without route-by-route DRAM loopback.

Cone compute already uses native `32x16` FP32 pages:

```text
source: 4 narrow pages = 2048 FP32 positions
base:   3 narrow pages = 1536 FP32 values
output: 3 narrow pages = 1536 FP32 values
```

Persistent Cone workspace supports two physical layouts:

* `row-major`: compact and usually better for shifted low-order routes such as
  `db7`;
* `tile-native`: three narrow pages per 1536-element group, reducing an
  intermediate writer from 96 half-stick packets to three page writes;
* `auto`: chooses once on the host from route geometry.

Override the policy only for A/B validation:

```bash
export TT_WAVELET_LWT_CONE_WORKSPACE_LAYOUT=row-major
export TT_WAVELET_LWT_CONE_WORKSPACE_LAYOUT=tile-native
export TT_WAVELET_LWT_CONE_WORKSPACE_LAYOUT=auto
```

Tile-native persistence is implemented, but it is not a universal zero-copy
solution. One-element and other shifted route offsets still require reader-side
remapping, and `source_left_pad = 17 - K` is still used to align the existing
horizontal stencil to its 17-tap register layout. Treat both as current known
limitations, not as unfinished initial implementation.

The completed design investigation and Wormhole measurements are documented in:

* `docs/LWT_TILE_NATIVE_OPTIMIZATION.md`;
* `docs/LWT_MEMORY_MODES.md`;
* `docs/HORIZONTAL_STENCIL.md`.

Do not restart the broad tile-native architecture investigation unless new
measurements or a concrete correctness/performance issue justify it.

## Architecture selection

TT-Metal detects real hardware and supplies device-kernel JIT defines. Project
SFPI code has one local architecture contract in
`tt-wavelet/kernels/ckernel.h`:

```cpp
TT_WAVELET_TENSIX_ARCH_WORMHOLE
TT_WAVELET_TENSIX_ARCH_BLACKHOLE
TT_WAVELET_TENSIX_ARCH
```

`TT_WAVELET_TENSIX_ARCH` is selected from exactly one of TT-Metal's
`ARCH_WORMHOLE` or `ARCH_BLACKHOLE`. Architecture-specific SFPI code must branch
on the project value, for example:

```cpp
#if TT_WAVELET_TENSIX_ARCH == TT_WAVELET_TENSIX_ARCH_BLACKHOLE
// Blackhole implementation
#elif TT_WAVELET_TENSIX_ARCH == TT_WAVELET_TENSIX_ARCH_WORMHOLE
// Wormhole implementation
#else
#error "Unsupported Tensix architecture"
#endif
```

Do not manually define these values in CMake or normal hardware runs. Do not set
`ARCH_NAME=blackhole`; TT-Metal documents `ARCH_NAME` as a simulation override.
Unset stale values inherited from another shell:

```bash
unset ARCH_NAME
```

Do not put Wormhole and Blackhole LLK include directories into the same device
include path. The forwarding header includes only the LLK tree selected by the
JIT.

The horizontal rotate intentionally differs by architecture:

* Wormhole retains the existing `SHFLSHR1` erratum-based fast path;
* Blackhole, where that erratum is fixed, rotates both registers and injects
  the cross-register halo with a lane-zero compare and masked move.

Never compile the Wormhole erratum path for Blackhole. See
`docs/BLACKHOLE_PORTING_AND_TESTING.md` for ISA evidence and validation details.

## Blackhole status

The port has been built and exercised on this Blackhole P150b:

* representative `db1`, `db2`, `db7`, and `bior3.9` comparisons pass;
* all 106 generated schemes execute without JIT, kernel, or runtime failures;
* 68/106 pass the current `1e-2` PyWavelets threshold on the tested signal;
* 38 high-order schemes exceed that threshold.

The old Wormhole report recorded 70 passes and 36 high-order failures. This
difference has not yet been classified because exact same-input Wormhole versus
Blackhole outputs have not been collected. Do not call all 38 failures existing
baseline failures until that equivalence check is complete. Track these two
questions separately:

1. architecture equivalence against Wormhole/current backend;
2. compatibility against PyWavelets, whose filter-bank arithmetic ordering is
   different from the FP32 lifting factorization.

Remaining Blackhole acceptance work:

* a focused synthetic `K=17` kernel test;
* broader odd/even, short-length, first/last chunk, and chunk-boundary coverage;
* Wormhole/backend equivalence for the 38 high-order failures;
* Blackhole device-only performance benchmarks.

Do not present the port as fully accepted until these items are complete.

## ILWT status

1D ConeStreamed ILWT is implemented for all 106 production schemes. The two
initial reciprocal-scale routes are fused lazily into the predict/update chain
without coefficient folding: FP32 scale multiplication still happens in SFPU
before the corresponding stencil arithmetic. The last inverse route can be
interleaved/cropped directly from its three output-CB pages without first
materializing that updated stream in L1.

Measured defaults are:

* inverse scale fusion enabled;
* final-route/interleave fusion selected automatically for tile-native
  workspaces and disabled for row-major workspaces.

For A/B validation:

```bash
export TT_WAVELET_ILWT_FUSE_INVERSE_SCALE=0  # or 1
export TT_WAVELET_ILWT_FUSE_FINAL_INTERLEAVE=auto  # or 0/1
```

Blackhole validation after fusion includes 106/106 inverse scheme JIT/runtime
success, odd/even and 3072-boundary fused/unfused equivalence, both workspace
layouts, and synthetic K=17. High-order FP32 factorization error is tracked
separately from ILWT geometry regression. The remaining ILWT acceptance item is
hardware Wormhole equivalence; this server cannot provide it.

See `docs/ILWT_1D.md` for the arithmetic contract, validation matrix, and
Blackhole A/B timings. Do not claim Wormhole ILWT equivalence from compile-time
coverage or Blackhole results.

## Build and run workflow

Use the local pinned TT-Metal and SFPI toolchain. `scripts/common.sh` configures
`TT_USE_SYSTEM_SFPI=OFF` because this TT-Metal revision requires SFPI 7.17 while
the server-wide compiler may be newer.

For normal tt-wavelet iteration, rebuild only the required target:

```bash
unset ARCH_NAME
./update.sh Release lwt
source ./scripts/set_env.sh
```

`update.sh` configures the project, reapplies the local TT-Metal CMake fixes,
and rebuilds tt-wavelet. Use `build.sh` only for a first/bootstrap build or when
TT-Metal itself must be rebuilt; it builds TT-Metal plus tt-wavelet and installs
the local Python bindings.

A successful host build is insufficient for SFPI changes. Run at least one LWT
command so TT-Metal JIT-compiles and executes the kernel for the detected device.

Useful preflight checks:

```bash
tt-smi -ls
tt-smi -s
git status --short
git rev-parse HEAD
git -C tt-metal rev-parse HEAD
```

## Correctness validation

Start with Blackhole smoke coverage:

```bash
source ./scripts/set_env.sh
python3 compare.py --wavelet db1 --memory-mode cone --tolerance 1e-5
python3 compare.py --wavelet db7 --memory-mode cone --tolerance 1e-3
python3 compare.py --wavelet bior3.9 --memory-mode cone --tolerance 1e-5
```

For a layout change, run representative schemes in forced `row-major`, forced
`tile-native`, and `auto`. Compare those outputs directly, then run all schemes
for runtime stability:

```bash
python3 compare.py --all-green --memory-mode cone --tolerance 1e-2
```

Do not use only PyWavelets pass/fail counts to validate an architecture change.
Pay special attention to errors periodic at every eighth lane, 16-column face,
or tile boundary; those patterns strongly indicate SFPI shift/layout regressions.

For new SFPI stencil behavior, cover at least `K=1`, `K=2`, shipped maximum
`K=9`, and synthetic maximum `K=17`, including aligned and `+1 FP32` route
offsets. Preserve arbitrary positive/negative shifts and schemes containing
swaps.

## Performance validation

Measure device execution only, with program construction outside the timing
boundary. Do not run device benchmarks concurrently.

Minimum acceptance matrix:

```text
db7:      100k-1M sweep, 5M, and 8M or largest stable size
high-K:   representative medium and large sizes (currently bior3.9, K=9)
layouts:  row-major, tile-native, and auto where relevant
```

Example:

```bash
TT_WAVELET_LWT_CONE_WORKSPACE_LAYOUT=auto \
  ./build/lwt --benchmark --memory-mode cone \
  --repeats 10 --warmup-runs 3 --length 5000000 db7
```

Report median, minimum, p10, p90, standard deviation, active-core count, chunk
size, workspace layout, L1 usage, and exact device/TT-Metal revisions. Compare
Wormhole and Blackhole only with the same timing boundary and equivalent build
configuration.

## Invariants for future changes

Preserve all of the following:

* FP32 storage and FP32 SFPU arithmetic; do not substitute TF32, BF16, FP16, or
  FPU matmul;
* generic predict/update support for `K=1..17`, arbitrary scheme shifts, and
  route geometry;
* metadata-only swaps, the three-slot `A/B/Scratch` model, ConeStreamed
  execution, and terminal direct-to-DRAM output;
* finite-signal boundary behavior and both odd and even input lengths;
* support for both Wormhole and Blackhole without users transforming layouts or
  manually selecting an architecture;
* L1 usage that does not materially reduce useful chunk size or active-core
  count.

Keep layout and route assumptions explicit in shared descriptors/constants.
Add compile-time or host assertions for architecture, coefficient capacity,
alignment, halo, route offsets, and tile capacity rather than scattering hidden
assumptions through reader, compute, and writer kernels.
