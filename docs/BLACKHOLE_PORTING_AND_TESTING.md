# Blackhole port and validation guide

## Status

The Blackhole port has been built and exercised on a Blackhole P150b.  The
representative `db1`, `db2`, `db7`, and `bior3.9` comparisons pass.  All 106
generated schemes execute without a kernel/JIT/runtime failure; 68 pass the
existing `1e-2` PyWavelets threshold and 38 high-order schemes exceed it.  The
saved Wormhole report had 36 such failures, so exact architecture equivalence
for this set remains open.  Focused synthetic `K=17`, shape-boundary coverage,
and performance acceptance also remain open.

The first Blackhole run must include the architecture-aware changes in:

* `tt-wavelet/kernels/ckernel.h`;
* `tt-wavelet/kernels/sfpi/horizontal_stencil_sfpi.h`.

Running the previous Wormhole-only stencil unchanged on Blackhole is expected
to produce incorrect convolution results for routes that need a cross-register
halo.  This is a correctness issue, not merely a performance difference.

## What was verified

### The `SHFLSHR1` difference is real

The current Wormhole fast path performs this pair of operations:

```cpp
SFPSHFT2(a, a, SHFLROR1);
SFPSHFT2(b, b, SHFLSHR1);
```

For every 8-lane group, the intended result for `b` is a one-element right
shift whose first element comes from the end of `a`.  For example:

```text
a before: [100 101 102 103 104 105 106 107]
b before: [200 201 202 203 204 205 206 207]

b wanted: [107 200 201 202 203 204 205 206]
```

On Wormhole, `SHFLSHR1` has a documented hardware bug.  Its first lane is
unpredictable by specification, but has been observed to receive lane 7 from
the source seen by the most recent `SHFLROR1`.  The old stencil deliberately
used that observed behavior to inject `107` into `b`.

Primary local evidence:

* `tt-isa-documentation/WormholeB0/TensixTile/TensixCoprocessor/SFPSHFT2.md`
  lines 108-118 state that the mode should not be used because of the hardware
  bug;
* the same file at line 158 describes the observed `vc0[Lane + 7]` behavior and
  explicitly says it has not been thoroughly validated;
* `tt-isa-documentation/BlackholeA0/TensixTile/TensixCoprocessor/SFPSHFT2.md`
  line 8 says that Blackhole fixes this bug;
* the Blackhole functional model at lines 109-121 shifts within each 8-lane
  group and writes exactly zero into its first lane.

Therefore the same two-instruction sequence on Blackhole produces:

```text
b on Blackhole: [0 200 201 202 203 204 205 206]
```

The fix is to use contractual operations on Blackhole:

1. rotate both `a` and `b` with `SHFLROR1`;
2. compare the hardware `LTILEID` register with zero, enabling lane 0 in each
   8-lane subvector;
3. move lane 0 of `a` to lane 0 of `b`;
4. re-enable all lanes.

This yields the required `[107 200 ... 206]` without relying on the Wormhole
erratum.  The explicit `SFPNOP` after each shuffle is retained.  Blackhole has
automatic scheduling, but its ISA documentation notes that an explicit NOP
keeps the shuffle-plus-NOP pair at two cycles instead of a three-cycle
automatic stall.

### How TT-Metal selects Blackhole headers

On real hardware, TT-Metal detects the device architecture.  Its Blackhole HAL
then supplies the kernel JIT with:

```text
tt_metal/hw/ckernels/blackhole/metal/common
tt_metal/hw/ckernels/blackhole/metal/llk_io
tt_metal/hw/ckernels/blackhole/metal/llk_api
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu
tt_metal/hw/inc/internal/tt-1xx
tt_metal/hw/inc/internal/tt-1xx/blackhole
tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc
tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib
```

It also defines `ARCH_BLACKHOLE` and compiles compute kernels with
`-mcpu=tt-bh-tensix`.  See
`tt-metal/tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp`.

The project maps TT-Metal's JIT define to the single local compile-time value
`TT_WAVELET_TENSIX_ARCH` in `tt-wavelet/kernels/ckernel.h`.  SFPI branches use
that value with `TT_WAVELET_TENSIX_ARCH_BLACKHOLE` or
`TT_WAVELET_TENSIX_ARCH_WORMHOLE`; users must not define it manually.

Header ownership is consequently:

| Header | Source on Blackhole | Project action |
| --- | --- | --- |
| `ckernel.h` | `tt_llk_blackhole/common/inc` | The local forwarding wrapper selects it under `ARCH_BLACKHOLE`. |
| `ckernel_defs.h` | Blackhole JIT include set | Keep the generic include in the kernel. |
| `cmath_common.h` | Blackhole compute LLK include set | Keep the generic include in the kernel. |
| `sfpi.h` | Tensix compiler/toolchain selected by `-mcpu=tt-bh-tensix` | Do not copy a Wormhole header into the project. |

Do not add Wormhole and Blackhole LLK directories to the same kernel include
path.  They intentionally contain headers with the same names and different
architecture definitions.

`tt-wavelet/CMakeLists.txt` still contains Wormhole LLK paths for the host
target's include list.  The host sources do not include these SFPI/LLK headers,
while device kernels are compiled separately by the TT-Metal JIT.  They are
legacy build/IDE paths rather than the mechanism selecting the device
architecture.  If a future host source starts including kernel headers, remove
those paths instead of extending the list with Blackhole paths.

Do not set `ARCH_NAME=blackhole` for a normal hardware run.  TT-Metal documents
`ARCH_NAME` as a simulation-only override; the default is hardware detection.
An old `ARCH_NAME=wormhole_b0` value inherited from another shell should be
unset.

## Reproducible source versions

The static investigation used these revisions:

```text
tt-wavelet:          8e2d4b2665e68500fcea1f0ac90bf234227d259d
tt-metal:            f87c34a93ee4686c1d7f7adbd4df7ca1804d91ff
tt-isa-documentation b7738d9ac14a34a4033d60dde9463466b23082e1
```

Use the same TT-Metal revision for the first comparison.  If the Blackhole
server requires a newer TT-Metal revision, record the exact commit and retest
both correctness and timings; do not silently compare two different runtime
stacks.

The ISA documentation is useful for investigation but is not a compile-time
dependency.  It can live as a sibling checkout at
`tt-isa-documentation/`.  No header should be copied from the documentation
repository into the program.

## Blackhole server setup and build

Run all commands from the repository root.  First inspect the machine and the
working tree:

```bash
tt-smi -ls
tt-smi -s
git status --short
git rev-parse HEAD
git -C tt-metal rev-parse HEAD
unset ARCH_NAME
```

The project environment script points `TT_METAL_ROOT`, `TT_METAL_HOME`, and
`TT_METAL_RUNTIME_ROOT` to the local `tt-metal` checkout and selects Clang 20:

```bash
source ./scripts/set_env.sh
```

The build deliberately sets `TT_USE_SYSTEM_SFPI=OFF`.  The pinned TT-Metal
revision requires SFPI 7.17.0, while the P150b server currently has a newer
system SFPI.  TT-Metal downloads its checksummed 7.17.0 toolchain under the
checkout's `runtime/` directory instead of weakening the version check or
replacing the server-wide compiler.

Build only the `lwt` target:

```bash
./update.sh Release lwt
source ./scripts/set_env.sh
./build/lwt --help
```

Do **not** use `./build.sh`: it rebuilds the full TT-Metal tree with more than a
thousand targets.  `./update.sh Release lwt` configures the project and rebuilds
only the target needed here.

During the first real `lwt` execution, TT-Metal JIT-compiles the dataflow and
compute kernels for the detected Blackhole device.  A successful host build by
itself does not validate the Blackhole SFPI code; at least one device run is
required.

## Required focused shift test

Before trusting an end-to-end wavelet result, add or run a small SFPI test that
loads distinct values into every lane and checks all four 8-lane groups:

```text
a = [100..107] [110..117] [120..127] [130..137]
b = [200..207] [210..217] [220..227] [230..237]
```

The architecture-safe rotate must produce:

```text
b = [107 200..206]
    [117 210..216]
    [127 220..226]
    [137 230..236]
```

Also run the raw Blackhole `SHFLSHR1` once as an ISA sanity check.  Its first
lane in each group must be zero.  If it instead receives the previous
register's lane 7, the toolchain/device revision does not match the documented
Blackhole behavior and the result must be investigated before continuing.

Repeat the safe-rotate test across both faces of a 32x16 narrow tile and across
the source-tile boundary used by `_horizontal_stencil_plus_base_narrow()`.
This catches a test that verifies only an isolated LREG but misses DST/face
addressing.

## Correctness validation

Start with a short smoke test:

```bash
source ./scripts/set_env.sh
python3 compare.py --wavelet db7 --memory-mode cone --tolerance 1e-3
python3 compare.py --wavelet bior3.9 --memory-mode cone --tolerance 1e-5
```

`db7` uses `1e-3` here because its current FP32 factorization differs from
PyWavelets by roughly `3.4e-4` on the default short signal.  This tolerance is
not an allowance for a Blackhole regression; Blackhole must still match the
saved Wormhole/backend output much more closely.

Then cover the stencil shapes explicitly:

* `db1` or `haar`: only `K=1`, a control case that does not stress repeated
  halo injection;
* `db7`: mostly `K=1` and `K=2`, the most important low-arithmetic case;
* `bior3.9`: contains `K=9`, the largest coefficient vector in the currently
  shipped JSON schemes;
* a synthetic kernel-level `K=17` case: the SFPI template supports up to 17
  even though the current scheme set tops out at 9.

For each representative scheme cover:

* positive and negative shifts;
* schemes containing swaps;
* odd and even signal lengths;
* short signals;
* first, middle, and last chunks;
* lengths immediately below, at, and above a chunk boundary;
* `row-major`, `tile-native`, and `auto` workspace selection.

Workspace layout can be forced for A/B validation:

```bash
TT_WAVELET_LWT_CONE_WORKSPACE_LAYOUT=row-major \
  python3 compare.py --wavelet bior3.9 --memory-mode cone --tolerance 1e-5

TT_WAVELET_LWT_CONE_WORKSPACE_LAYOUT=tile-native \
  python3 compare.py --wavelet bior3.9 --memory-mode cone --tolerance 1e-5

TT_WAVELET_LWT_CONE_WORKSPACE_LAYOUT=auto \
  python3 compare.py --wavelet bior3.9 --memory-mode cone --tolerance 1e-5
```

Finally run all generated schemes for runtime stability:

```bash
python3 compare.py --all-green --memory-mode cone --tolerance 1e-2
```

This command is not expected to make all 106 schemes pass against PyWavelets.
On the current Wormhole baseline, 36 high-order factorizations exceed `1e-2`
because their long FP32 lifting chains are numerically ill-conditioned.  Treat
that as a known compatibility baseline, not automatically as a Blackhole
regression.  The stronger architecture check is:

1. save the exact Wormhole output for the same signal and scheme;
2. compare Blackhole against Wormhole or the existing correct backend;
3. separately report compatibility error against PyWavelets.

The new Blackhole path keeps the same SFPU arithmetic ordering.  For ordinary
finite values, it should reproduce the halo values used by the Wormhole path.
Any new error localized at every eighth lane, face boundary, or tile boundary
is a likely shift/layout regression.

## Performance validation

Use device benchmark mode so program construction and coefficient readback are
outside the timing loop:

```bash
source ./scripts/set_env.sh

TT_WAVELET_LWT_CONE_WORKSPACE_LAYOUT=auto \
  ./build/lwt --benchmark --memory-mode cone \
  --repeats 10 --warmup-runs 3 --length 5000000 db7

TT_WAVELET_LWT_CONE_WORKSPACE_LAYOUT=auto \
  ./build/lwt --benchmark --memory-mode cone \
  --repeats 10 --warmup-runs 3 --length 5000000 bior3.9
```

For a small sweep and a machine-readable result:

```bash
python3 compare_timings.py \
  --backend tt-wavelet \
  --tt-memory-mode cone \
  --wavelets db7 bior3.9 \
  --lengths 100000 1000000 5000000 8000000 \
  --tt-repeats 10 \
  --tt-warmup-runs 3 \
  --csv blackhole_timings.csv \
  --overwrite
```

Record at least median, minimum, p10, p90, standard deviation, active-core
count, chunk size, workspace layout, and exact device/runtime revisions.  The
Blackhole-safe rotate adds `SFPSETCC`, masked `SFPMOV`, and `SFPENCC` per halo
transfer compared with the Wormhole erratum fast path, so compare performance
by `K`: the overhead is absent for `K=1` and grows with the number of odd taps.

Do not run multiple device benchmarks concurrently; they would invalidate the
latency comparison.

## Acceptance criteria

The Blackhole port is accepted only when all of the following are true:

1. TT-Metal detects a Blackhole device and JIT-compiles with
   `ARCH_BLACKHOLE`/`-mcpu=tt-bh-tensix`.
2. The focused safe-rotate test passes for all lane groups, faces, and the
   source-tile boundary.
3. `K=1`, `K=2`, shipped maximum `K=9`, and synthetic `K=17` tests pass.
4. Row-major and tile-native outputs match for representative schemes.
5. No new error appears relative to the Wormhole/backend baseline.
6. All 106 schemes run without hangs, illegal NoC traffic, or kernel crashes.
7. Timings are collected with the same device-only boundary as the Wormhole
   baseline.

## Prompt/checklist for remaining Blackhole acceptance

The following can be pasted into a new Codex session:

> Work in the tt-wavelet repository and read `AGENTS.md` and
> `docs/BLACKHOLE_PORTING_AND_TESTING.md` first.  The machine has Blackhole
> hardware.  Do not commit and do not reset unrelated working-tree changes.
> Verify `tt-smi`, the tt-wavelet commit, and the tt-metal commit.  Do not set
> `ARCH_NAME` on hardware.  Build only with `./update.sh Release lwt`, then
> `source ./scripts/set_env.sh`; never use `./build.sh`.  Confirm that TT-Metal
> selects `ARCH_BLACKHOLE`, maps it to `TT_WAVELET_TENSIX_ARCH`, uses the
> Blackhole LLK paths, and compiles with `-mcpu=tt-bh-tensix`.  Run the focused
> SFPI rotate and synthetic `K=17` tests, broaden shape/chunk-boundary coverage,
> compare the 38 high-order outputs against the same-input Wormhole/backend
> baseline, and collect the db7/bior3.9 device-only benchmarks.  Report exact
> commands, logs, output diffs, timing statistics, device revision, files
> changed, and any unresolved architecture limitation.

## Remaining uncertainty

The ISA difference and JIT header/compiler selection are proven statically from
the checked-in sources.  Representative schemes and all 106 generated schemes
have now exercised Blackhole DST addressing, narrow-tile faces, and the complete
ConeStreamed pipeline on this P150b.  The focused rotate/`K=17` tests, broader
boundary matrix, same-input Wormhole equivalence, and Blackhole performance
measurements remain explicit acceptance work rather than claimed results.
