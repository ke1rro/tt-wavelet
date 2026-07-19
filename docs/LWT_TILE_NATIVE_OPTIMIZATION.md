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

## Measured policy

Wormhole measurements show that row-major is the correct automatic ILWT layout. In the final 5M, five-warmup, 20-run measurement, `bior3.9` medians were 11.363 ms automatic row-major, 11.392 ms forced row-major, and 17.443 ms forced tile-native. The former automatic tile-native baseline was 15.677 ms. Wormhole also regresses when direct final interleave is used, so production disables it.

Blackhole measurements established tile-native ILWT with direct final interleave as its production policy. Forward retains the route-geometry heuristic on both architectures because it selects row-major for shifted `db7` and tile-native for aligned `bior3.9`.

Use the layout environment variable only to reproduce A/B results; user input never needs conversion.

## L1 impact

The tile-native layout aligns each slot to a complete three-page group. At 8M `bior3.9`, the maximum logical stream is 62,985 elements and the allocated length is 64,512 elements. Across three slots that adds 18,324 padding bytes. The exact total, including every circular buffer, is 815,904 bytes/core; see [LWT_MEMORY_MODES.md](LWT_MEMORY_MODES.md).

## Correctness checks

Layout changes must be compared directly, not inferred only from PyWavelets tolerance counts:

```bash
python3 scripts/validate_lwt_boundaries.py --layouts auto row-major tile-native
python3 scripts/validate_ilwt.py --layouts auto row-major tile-native
```

Investigate errors periodic at eight lanes, a 16-column face, or a 1,536-element group first; these patterns indicate register rotation or physical-layout mapping defects.
