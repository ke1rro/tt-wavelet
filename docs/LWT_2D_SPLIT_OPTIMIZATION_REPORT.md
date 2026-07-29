# 2D LWT fused split optimization report

Date: 2026-07-28

## Result

The initial forward 2D split now has a reader-only benchmark path and a fused
tiled implementation. On the attached Blackhole P150b, the required 264 split
cases are bit-identical to the scalar reference. Across the seven requested
benchmark shapes, prepared reader-only workload latency improves by
7.65x--10.41x.

The implementation is architecture-neutral. It uses the common TT-Metal
dataflow and NoC APIs and does not add a Blackhole-specific branch to the split
hot path.

## Root cause of the old split

The scalar implementation invokes `initialize_plane()` separately for EE, EO,
OE, and OO. Each invocation clears a padded plane and gathers values through
scalar tiled indexing. The path rereads input data, performs tiny NoC reads,
and waits after each read. The measured NoC call/barrier counts confirm that
this orchestration dominates the split:

| Shape | Scalar reads/barriers | Tiled reads/barriers |
| --- | ---: | ---: |
| 64x64 | 10,000 / 10,000 | 36 / 9 |
| 128x128 | 19,360 / 19,360 | 144 / 25 |
| 256x256 | 42,640 / 42,640 | 576 / 81 |
| 512x512 | 137,776 / 137,776 | 1,728 / 221 |
| 1000x100 | 163,744 / 163,744 | 760 / 96 |
| 1000x200 | 154,112 / 154,112 | 1,900 / 224 |
| 1024x1024 | 453,024 / 453,024 | 4,608 / 550 |

## ISA and API investigation

The following checked-in documentation and APIs were inspected:

- `tt-isa-documentation/BlackholeA0/TensixTile/TensixCoprocessor/SFPTRANSP.md`
- `tt-isa-documentation/BlackholeA0/TensixTile/TensixCoprocessor/SFPSHFT2.md`
- `tt-isa-documentation/BlackholeA0/TensixTile/TensixCoprocessor/Dst.md`
- the corresponding Wormhole documents
- `tt-isa-documentation/WormholeB0/TensixTile/TensixCoprocessor/Packers/Downsampling.md`
- Wormhole unpacker format-conversion and regular-unpack documentation
- TT-Metal's compute register API, unary unpack/pack initialization, Blackhole
  LLK packer headers, and dataflow NoC APIs

Relevant findings:

1. `SFPTRANSP` transposes axes across a stack of four vector registers. It is
   not an arbitrary lane permutation or an even/odd lane compressor.
2. `SFPSHFT2` provides 8-lane rotate/shift modes and four-register copy modes.
   It can move a halo lane efficiently, but it cannot densely compress every
   other lane into a 32-lane result.
3. Blackhole has eight ordinary writable SFPU LRegs. With FP32 destination
   accumulation and the default double-buffered Dst contract, four 32x32
   tiles are available to a kernel. Four raw source tiles consume that working
   set, leaving no simultaneous space for four dense outputs.
4. SFPU conditional execution and masked moves can replace selected lanes but
   do not provide a general parity-compress primitive.
5. Wormhole documents packer downsampling, but no supported, matching
   cross-architecture TT-Metal API was found for configuring the Blackhole
   packer to densely produce all four parity outputs. The Blackhole public pack
   configuration does not expose this as a portable contract.
6. The standard unpacker APIs perform contiguous unpack, format conversion,
   broadcast, and transpose operations. No verified API selects alternating
   rows and columns and packs them densely.

### Option A: SFPU split

Rejected for this implementation. A nominal aligned 64x64 region is four raw
tiles, but wavelet padding can make the same logical window intersect a 3x3
physical tile set. Even in the four-tile case, the available FP32 Dst working
set is consumed by the inputs. Dense parity extraction would require multiple
unpack/SFPU/pack passes and a non-existent general lane-compress operation.
The expected cost and implementation risk are higher than a fused
data-movement traversal.

### Option B: unpacker/packer-driven split

Rejected for this implementation. The investigated public APIs do not
contractually support alternating row/column selection followed by dense
32x32 packing on both Wormhole and Blackhole. The Wormhole-only downsampling
documentation is not sufficient grounds for a shared production path.

### Option C: fused data-movement split

Selected. It is bit-preserving, uses supported APIs on both architectures, and
directly attacks the measured small-read/barrier problem.

## New split dataflow

For every active logical 32x32 polyphase macro-tile:

1. Compute the corresponding 64x64 raw logical window.
2. Classify it as interior only when the complete raw window lies inside the
   original logical `H x W` domain.
3. Collect the two or three physical raw tile rows and columns touched by that
   window.
4. Issue all full-tile NoC reads into a 3x3 L1 scratch grid.
5. Execute one read barrier for the batch.
6. For a complete interior output tile, precompute source even/odd column
   descriptors and tiled face offsets.
7. Traverse the 32x32 output coordinates once and write EE, EO, OE, and OO
   together as `uint32_t` bit patterns.

The full interior loop has no `symmetric_index()`, source-tile search,
division/modulo, `tile_element_offset()`, output clear, or NoC operation.

If a logical rectangle covers only part of a tile, the staged interior data is
written through the existing bounded per-plane path. This preserves padding
semantics without penalizing complete interior tiles.

## Boundary fallback

A macro-tile is a boundary tile when any element of its 64x64 raw logical
window lies outside `H x W`. Source row and column indices are reflected using
the existing symmetric mapping against the original logical dimensions.
Unique reflected physical tiles are staged with full-tile reads and one batch
barrier. The existing element-level gather is retained only after staging for
the irregular reflected mapping. Physical zero padding is never treated as
mathematical input.

## Memory use

- Raw split scratch: `3 * 3 * 4096 = 36,864` bytes per active core.
- Reader/writer synchronization page: 64 bytes.
- Split-related synchronization allocation: 36,928 bytes.
- Source/base/output/zero circular buffers: 36,864 bytes.
- Chunk, reader route, writer route, and band metadata: 512 bytes after
  padding the independently addressable pages to 128 bytes.
- Fixed L1 allocation: 74,304 bytes before workspace planes.
- Maximum planner allocation observed by the planner tests: 156,224 bytes.

The 128-byte metadata pages and 64-byte CB alignment satisfy Blackhole's
64-byte DRAM-read alignment requirement and are also valid on Wormhole.

## Files

Modified:

- `tt-wavelet/kernels/dataflow/lwt_2d_reader.cpp`
- `tt-wavelet/kernels/primitives/tile_2d_layout.hpp`
- `tt-wavelet/tt_wavelet/include/device_protocol/lwt_2d_config.hpp`
- `tt-wavelet/tt_wavelet/include/lifting/device_2d.hpp`
- `tt-wavelet/tt_wavelet/include/lifting/plan_2d.hpp`
- `tt-wavelet/tt_wavelet/src/lifting/device_2d.cpp`
- `tt-wavelet/main_2d.cpp`
- `scripts/benchmark_lwt_2d_split.py`

Added result artifacts:

- `docs/lwt_2d_split_correctness_blackhole_p150.json`
- `docs/lwt_2d_split_blackhole_p150.json`
- `docs/lwt_2d_split_blackhole_p150.csv` (ignored by the repository's CSV
  rule but retained in the workspace)

No split compute kernel or split SFPI header was added because the ISA/API
comparison selected the data-movement implementation.

## Build and validation commands

Build:

```bash
./update.sh Release lwt_2d
source ./scripts/set_env.sh
```

Planner tests:

```bash
cmake --build build --target lwt_2d_planner_tests ilwt_2d_planner_tests -j2
./build/lwt_2d_planner_tests
./build/ilwt_2d_planner_tests
```

Independent split correctness:

```bash
unset ARCH_NAME
source scripts/set_env.sh >/dev/null
python3 scripts/validate_lwt_2d_split.py \
  --wavelet db7 \
  --implementation tiled \
  --cores 64 \
  --timeout-seconds 120 \
  --fail-fast \
  --result-json docs/lwt_2d_split_correctness_blackhole_p150.json
```

Reader-only split benchmark:

```bash
unset ARCH_NAME
source scripts/set_env.sh >/dev/null
python3 scripts/benchmark_lwt_2d_split.py \
  --wavelet db7 \
  --implementations both \
  --cores 64 \
  --repeats 5 \
  --warmup-runs 2 \
  --timeout-seconds 240 \
  --csv docs/lwt_2d_split_blackhole_p150.csv \
  --result-json docs/lwt_2d_split_blackhole_p150.json
```

## Correctness result

All 264 cases passed bit-identically:

- 22 requested shapes, including all 31/32/33 and 63/64/65 boundaries and
  1000x100;
- 12 patterns: zeros, constants, row-major sequence, row ramp, column ramp,
  checkerboard, corner impulses, row/column 31 and 32 impulses, and
  deterministic random values;
- all EE, EO, OE, and OO snapshots.

Planner validation also passed for all 106 generated schemes:

```text
2D dependency-cone planner validation passed for 106 schemes
2D ILWT dependency-cone planner validation passed
```

## Split benchmark

Prepared workload wall-clock time is the primary latency result. The device
timestamp converted at an assumed 1 GHz remains a separate diagnostic because
the P150 clock is not established by that conversion.

| Shape | Scalar ms | Tiled ms | Speedup | Tiled input elements/s |
| --- | ---: | ---: | ---: | ---: |
| 64x64 | 2.306310 | 0.278428 | 8.283x | 14,711,164 |
| 128x128 | 2.316972 | 0.295016 | 7.854x | 55,535,971 |
| 256x256 | 2.398583 | 0.302320 | 7.934x | 216,776,925 |
| 512x512 | 4.321189 | 0.564614 | 7.653x | 464,288,877 |
| 1000x100 | 2.389793 | 0.281690 | 8.484x | 355,000,178 |
| 1000x200 | 2.398019 | 0.301898 | 7.943x | 662,475,406 |
| 1024x1024 | 11.381599 | 1.093910 | 10.405x | 958,557,834 |

The minimum acceptance criterion (`new < 0.5 * scalar`) and preferred target
(`new < 0.25 * scalar`) are met for every measured shape.

## Blackhole compute-port status

The established architecture-aware horizontal halo implementation is already
present in tracked code:

- TT-Metal defines exactly one of `ARCH_BLACKHOLE` and `ARCH_WORMHOLE`.
- `tt-wavelet/kernels/ckernel.h` forwards to the matching LLK tree.
- On Wormhole, `_horizontal_stencil_rotate_()` retains `SHFLROR1` followed by
  the documented erratum-dependent `SHFLSHR1`.
- On Blackhole, it uses two `SHFLROR1` operations, explicit `SFPNOP`s,
  `LTILEID`-based lane-zero selection, a masked move, and lane re-enable.

No shared SFPI, vertical stencil, `init_sfpu`, DEST synchronization, or
tile-register changes remain in this split work.

The focused halo and end-to-end tests could not be executed on this machine.
TT-Metal's unmodified `custom_sfpi_add` programming example hangs with all
three compute engines active. Earlier diagnostic runs of the horizontal and
vertical stencil controls likewise stopped before stencil execution; no
diagnostic source changes were retained. Reader-only and empty workloads
complete. Per
`BLACKHOLE_PORTING_AND_TESTING.md`, this is treated as an environment/runtime
compatibility failure rather than a split or halo-code failure.

Environment:

```text
device:             Blackhole P150b
tt-wavelet commit:  3956a9f044dd60959a651930721ce3c1db6bddaf
tt-metal commit:    f87c34a93ee4686c1d7f7adbd4df7ca1804d91ff
firmware:           19.11.0.0
runtime tested FW:  19.4.0
TT_USE_SYSTEM_SFPI: OFF
SFPI:               7.17.0[182]
SFPI compiler:      tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-g++
ARCH_NAME:          unset
```

Because the stock SFPU control fails before stencil execution, raw Blackhole
`SHFLSHR1`, the focused safe-halo test, full db1/db7/bior3.9 LWT, ILWT
round-trips, and complete db7 before/after latency were not run. The historical
approximately 17.2 ms full-db7 figure is not presented as a new measurement,
and no end-to-end speedup is claimed.

## Remaining dominant bottleneck

For the split stage itself, fixed dispatch and per-core startup dominate the
small shapes after the NoC/read-barrier reduction. For the complete transform,
the next expected bottlenecks remain route staging, route synchronization,
workspace persistence, and terminal writes, but they cannot be measured on
this machine until the stock compute-runtime incompatibility is resolved.

## Working-tree classification

Kept for the split:

- `scripts/benchmark_lwt_2d_split.py`
- `tt-wavelet/kernels/dataflow/lwt_2d_reader.cpp`
- `tt-wavelet/kernels/primitives/tile_2d_layout.hpp`
- `tt-wavelet/main_2d.cpp`
- `tt-wavelet/tt_wavelet/include/device_protocol/lwt_2d_config.hpp`
- `tt-wavelet/tt_wavelet/include/lifting/device_2d.hpp`
- `tt-wavelet/tt_wavelet/include/lifting/plan_2d.hpp`
- `tt-wavelet/tt_wavelet/src/lifting/device_2d.cpp`
- this report and the split result JSON/CSV artifacts

Kept as supplied architecture documentation:

- `docs/BLACKHOLE_PORTING_AND_TESTING.md`

Restored to tracked behavior after diagnostic experiments:

- `tt-wavelet/kernels/sfpi/lwt_sfpi_common.h`
- `tt-wavelet/kernels/sfpi/vertical_stencil_sfpi.h`
- `tt-wavelet/tests/kernels/vertical_stencil_compute.cpp`

Tracked and unchanged by this task:

- `tt-wavelet/kernels/ckernel.h`
- `tt-wavelet/kernels/sfpi/horizontal_stencil_sfpi.h`

Pre-existing/unrelated working-tree state:

- `Context.md`
- `tt-metal/tools/scaleout/CMakeLists.txt`
- `tt-metal/tt_metal/fabric/CMakeLists.txt`
- the untracked `tt-isa-documentation/` checkout

`git diff --check` reports only the two pre-existing trailing spaces in
`Context.md`. The scoped diff check for every retained implementation file and
the new report passes.
