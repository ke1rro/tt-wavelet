# Native narrow-tile workspace

The compute path uses native `32x16` FP32 pages:

```text
source: 4 pages = 2,048 FP32 positions
base:   3 pages = 1,536 FP32 values
output: 3 pages = 1,536 FP32 values
```

This shape fits seven narrow pages into the four `32x32` FP32 destination slots available under `tile_regs_acquire()` and eliminates unused half-tile transport.

## Physical mapping

For a logical index within a 1,536-element group, the persistent tile-native location is ordered by block, row, and lane:

```text
physical = group * 1536 + block * 512 + row * 16 + lane
```

where a logical row has three 16-element blocks. [`workspace_layout.hpp`](../tt-wavelet/kernels/primitives/workspace_layout.hpp) owns this mapping for both reader and writer. Row-major storage uses the logical index unchanged.

## Useful direct paths

- An aligned tile-native base group is read as three 2,048-byte pages.
- A complete tile-native intermediate output group is written as three page writes.
- The final inverse route can feed interleave directly from its three output pages on Blackhole when architecture policy enables it.

The remaining remaps are intentional:

- a one-element or otherwise shifted route offset is not page aligned;
- `source_left_pad = 17 - K` aligns the existing horizontal stencil register window;
- incomplete group tails require bounds handling;
- row-major persistence can be faster when most route offsets are shifted.

## Consolidated Wormhole paths

The final Wormhole implementation is hybrid rather than purely row-major or
purely tile-native:

- A row-major workspace can own a group-aligned tile mirror. Intermediate
  routes keep the canonical row-major data and publish a tile-native shadow
  when the output offset permits it. Later routes consume the shadow directly
  instead of gathering 16-element blocks again.
- Short route schedules with at least two groups per chunk may use aligned NoC
  staging. The production cutoff is at most five executable routes; longer
  schedules keep the scalar row-major gather because their packet setup cost
  did not amortize.
- ILWT batches 96 output sticks in its final interleave circular buffer. This
  removes the per-stick reserve/barrier/pop cadence while keeping a final write
  barrier before the buffer can be reused.

Matched experiments measured the hybrid tile-shadow path 4--11% faster at
100k elements, 9--23% faster at 1M, and 11--32% faster at 5M. Hybrid plus the
96-stick ILWT batch measured approximately 31--36% faster at 5M across the
representative schemes. For `bior3.9`, hybrid storage plus aligned NoC staging
and B96 measured 32.9% faster at 1M and 57.9% faster at 5M. These numbers are
the retained investigation results, not a claim that every scheme and signal
length has the same gain.

## Automatic layout policy

Wormhole uses only cheap execution-plan structure; there is no per-scheme
lookup table or runtime autotuner:

- Forward one-group chunks select tile-native when the plan has at least four
  executable routes. Shorter schedules retain the aligned-base geometry
  heuristic.
- Inverse one-group chunks select tile-native at seven or more executable
  routes. Shorter schedules retain hybrid row-major with B96.
- Multi-group chunks retain hybrid row-major steady-state execution.
- Tiny single-chunk and external-coefficient cases keep the established
  geometry/layout path.

The cutoffs are Wormhole-specific. Small timing matrices can be noisy enough
to reverse close layout winners; use repeated forced-layout measurements before
changing them. The `TT_WAVELET_LWT_WORKSPACE_LAYOUT` environment variable is an
A/B override, not a user-data conversion requirement.

## Blackhole policy

Blackhole retains tile-native ILWT with direct final interleave and the existing
forward route-geometry policy. The Wormhole tile mirror, aligned row-major NoC
staging, and B96 batch are deliberately disabled until they have independent
Blackhole correctness and timing evidence.

Blackhole permits larger kernel ELF binaries than Wormhole, so future work may
evaluate more aggressive inlining there. That is an architecture-specific
follow-up: the consolidated patch does not change compute/SFPU code, pragma
unroll factors, or arithmetic order on either architecture.

## Investigation classification

`KEEP`:

- hybrid row-major tile mirror on Wormhole;
- aligned NoC staging for multi-group, short-route hybrid plans;
- 96-stick Wormhole ILWT interleave batching;
- the inexpensive Wormhole one-group route-count heuristic;
- explicit L1 accounting and telemetry for the mirror and interleave batch.

`REJECTED`:

- one-stick ILWT interleave as the Wormhole production default;
- flush-only interleave synchronization;
- forcing one layout globally for every Wormhole plan.

`EXPERIMENT-ONLY` and not carried into production:

- `TTWV_EXPERIMENT_ILWT_INTERLEAVE_BATCH_STICKS`;
- `TTWV_EXPERIMENT_ILWT_INTERLEAVE_FLUSH_ONLY`;
- temporary profiler/debug switches and duplicated A/B implementations.

## L1 impact

The tile-native layout aligns each slot to a complete three-page group. At 8M `bior3.9`, the maximum logical stream is 62,985 elements and the allocated length is 64,512 elements. Across three slots that adds 18,324 padding bytes. The exact total, including every circular buffer, is 815,904 bytes/core; see [LWT_MEMORY_MODES.md](LWT_MEMORY_MODES.md).

## Correctness checks

Layout changes must be compared directly, not inferred only from PyWavelets tolerance counts:

```bash
python3 scripts/validate_lwt_boundaries.py --layouts auto row-major tile-native
python3 scripts/validate_ilwt.py --layouts auto row-major tile-native
```

Investigate errors periodic at eight lanes, a 16-column face, or a 1,536-element group first; these patterns indicate register rotation or physical-layout mapping defects.
