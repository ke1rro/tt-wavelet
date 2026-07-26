## Main diagnosis

The current implementation is **not compute-bound**. It is dominated by:

1. scalar data rearrangement on the RISC-V data-movement cores;
2. thousands of tiny NoC transfers and barriers;
3. route-level synchronization;
4. excessive one-output-tile-per-core parallelism;
5. fixed dispatch and control overhead.

The benchmark strongly supports this conclusion:

```text
100,000 elements -> 17.162 ms
210,000 elements -> 17.160 ms
```

The transform time is essentially constant while the input size more than doubles. That means the SFPU arithmetic is not determining runtime. You are observing a **latency floor**, not a throughput limit.

PyWavelets takes roughly `2.7–5.7 ms`, while the device stays around `17.2 ms`, so the current path needs approximately a **3–6× latency reduction** to become competitive at these sizes.

---

# 1. The largest bottleneck: scalar tile construction in the reader

The critical functions are:

```cpp
initialize_plane(...)
fill_base_or_scale_tile(...)
fill_stencil_source_tile(...)
```

They construct tiles using nested scalar loops:

```cpp
for (uint32_t row = 0; row < 32; ++row) {
    for (uint32_t column = 0; column < 32; ++column) {
        tile[tile_element_offset(row, column)] =
            read_plane(...);
    }
}
```

For every output tile of a predict/update route, the reader constructs:

* source tile 0;
* source tile 1;
* base tile.

That is approximately:

```text
3 × 1024 = 3072
```

scalar element copies **per output tile per lifting route**, before the SFPU receives anything.

For `db7`, this happens repeatedly across:

* two vertical transforms;
* two horizontal transforms;
* every predict/update route.

The SFPU then performs a relatively small amount of vector arithmetic, while the data-movement RISC-V has already executed thousands of scalar address calculations and L1 loads/stores. The supplied implementation confirms that all route tiles are currently assembled element-by-element. 

### Required optimization

Create two separate route paths.

#### Fast aligned path

When source, base, and output rectangles are tile-aligned:

```text
plane tile -> CB full-tile copy
```

Use full-tile local NoC transfers or direct L1 copies:

```cpp
noc_async_read(
    get_noc_addr(local_plane_tile_addr),
    get_write_ptr(cb_source),
    kTileBytes);
```

or the appropriate local one-packet/full-tile primitive.

No scalar `read_plane()` calls, no per-element indexing, no tile clearing.

#### Generic misaligned path

Retain the current gather implementation only for routes whose phase or rectangle cannot be represented as direct tiles.

The planner should report:

```text
aligned_route_tiles
misaligned_route_tiles
```

For performance schemes, the aligned proportion should be very high.

---

# 2. `split2d` is currently extremely expensive

The current `initialize_plane()` is called four times:

```text
initialize EE
initialize EO
initialize OE
initialize OO
```

Each call:

1. clears the complete tile-aligned plane using scalar stores;
2. rereads the corresponding input samples;
3. performs alternating-value gathers;
4. issues many small NoC reads;
5. waits on a barrier after each small read.

For an interior row, it reads short segments containing at most a small fraction of a tile. For edge elements, the code can issue:

```text
4-byte NoC read
barrier
4-byte NoC read
barrier
...
```

That is a correctness implementation, not a performance implementation. 

## Replace it with macro-tile `split2d`

A `64×64` raw input region consists of four `32×32` raw tiles and produces:

```text
EE: 32×32
EO: 32×32
OE: 32×32
OO: 32×32
```

The optimized interior path should be:

```text
read four raw tiles once
    -> unpack into DEST
    -> SFPU parity deinterleave
    -> pack four full polyphase tiles
```

Conceptually:

```cpp
split_2x2_raw_tiles_to_four_polyphase_tiles(
    raw00,
    raw01,
    raw10,
    raw11,
    ee,
    eo,
    oe,
    oo);
```

This gives three major improvements:

* each input element is read once rather than revisited by four plane initializers;
* deinterleaving runs on SFPU vectors instead of scalar RISC-V loops;
* reads and writes become complete tiles rather than tiny fragments.

Keep a special symmetric edge kernel for boundary macro-tiles. Interior tiles should never execute `symmetric_index()` or scalar gathering.

---

# 3. The final writer is also scalar and barrier-heavy

The final `write_band()` function writes rows in small face-limited segments:

```cpp
noc_async_write(..., count * sizeof(float));
noc_async_write_barrier();
```

The barrier is inside the row-segment loop. Therefore one output tile may require many small writes and many barriers.

The output buffers are already padded tile-layout tensors. That means the hot path can write complete output tiles, including physical padding:

```text
local band tile -> DRAM band tile
```

There is no need to crop individual rows on the device.

## Replace final band writes with full-tile writes

For every owned output band tile:

```cpp
noc_async_write_tile(
    destination_tile_id,
    output_accessor,
    local_plane_tile_addr);
```

or:

```cpp
noc_async_write(
    local_plane_tile_addr,
    output.get_noc_addr(destination_tile),
    kTileBytes);
```

Issue several asynchronous writes, then one barrier after a batch:

```cpp
for (...) {
    noc_async_write(...);
}
noc_async_write_barrier();
```

Do not execute one barrier per row or face segment.

Since physical output storage is tile-padded, the logical crop should remain a host/TTNN tensor-shape concern, not a device scalar-copy concern.

---

# 4. Intermediate route output is copied word-by-word

The writer currently persists each compute output tile into the local workspace through:

```cpp
for (uint32_t word = 0; word < kTileBytes / sizeof(uint32_t); ++word) {
    destination[word] = source[word];
}
```

For FP32 `32×32` tiles, that is:

```text
1024 scalar 32-bit copies per route tile
```

This is executed after every non-metadata route. 

## Immediate fix

Use the same local-NoC full-tile path already explored in the 1D implementation:

```cpp
noc_async_write(
    get_read_ptr(cb_output),
    get_noc_addr(local_output_tile_addr),
    kTileBytes);
```

Batch or flush appropriately before publishing route completion.

## Better long-term fix

Pack directly into workspace-backed CB storage.

Possible architecture:

```text
compute packs output
    -> CB whose backing address is the output plane tile
```

Then the writer does not copy the tile into the plane at all. It only coordinates ownership and, for terminal outputs, initiates DRAM writes.

This is more difficult because output plane slots vary by route, but it removes a complete local-copy stage.

---

# 5. Every route loads configuration from DRAM twice

For every route:

* the reader loads a route config page;
* the writer loads the same route config page separately;
* each load performs a NoC read;
* each load executes a read barrier;
* the page is copied word-by-word into a local array.

The route topology is mostly compile-time for a generated scheme:

* route type;
* axis;
* coefficient count;
* source/base role;
* swap behavior;
* scale behavior.

Only chunk-specific rectangles and physical offsets vary.

## Better protocol

Move invariant data to compile time:

```cpp
using Step = SchemeStep<Scheme, StepIndex>;
constexpr StepType type = Step::type;
constexpr uint32_t k = Step::k;
```

Move chunk geometry into one packed runtime descriptor loaded once per chunk:

```cpp
struct ChunkRuntimeDescriptor {
    Rect initial_planes[4];
    RouteGeometry routes[kRouteCount];
    BandGeometry bands[4];
};
```

Then:

```text
one config load per chunk
```

instead of:

```text
reader config load per route
writer config load per route
```

At minimum, preload all route pages into local L1 before execution begins.

---

# 6. Route synchronization serializes the complete program

After every route, the reader waits for the writer:

```cpp
cb_wait_front(cb_sync, 1);
cb_pop_front(cb_sync, 1);
```

The writer only signals after all route output tiles have been persisted to the local plane.

The dependency requires route (n) to finish before route (n+1) consumes its output, but the current synchronization is more expensive than necessary because:

* the writer performs scalar copies;
* the reader cannot begin any work for the next route;
* there is no tile-level pipeline;
* config loading is also serialized.

## First improvement

Keep one route-level barrier, but eliminate all barriers inside tile copy loops.

The desired route structure is:

```text
load all source/base tiles asynchronously
barrier once
compute all route tiles
write all output tiles asynchronously
barrier once
publish route completion
```

Not:

```text
small read
barrier
small read
barrier
compute one tile
small write
barrier
...
```

## Later improvement

Pipeline route tiles where dependency geometry allows it:

```text
route n+1 reads completed tile block 0
route n writes tile block 1
compute processes another block
```

This requires tile-level producer-consumer counters or separate ping-pong plane regions, so it should follow the simpler full-tile optimization.

---

# 7. The planner is over-parallelizing

Your telemetry shows:

```text
1000×100 -> 32 chunks, 32 cores
1000×120 -> 48 chunks, 48 cores
1000×190 -> 64 chunks, 64 cores
```

This strongly suggests that each core owns approximately one final band-tile rectangle.

The planner selection logic prioritizes:

1. maximum active core count;
2. then dependency overhead;
3. then chunk area.

That is the wrong objective for a latency-heavy route pipeline. The planner currently prefers more cores even if this produces tiny chunks with poor amortization and duplicated halos. 

One core handling one output tile still has to pay for:

* chunk config;
* four polyphase initializations;
* the complete `db7` route sequence;
* route config loads;
* route synchronization;
* four final band writes.

Therefore increasing from 32 to 64 cores does not reduce latency. Every core still executes almost the same control-flow sequence.

## Change planner objective

Do not maximize core count first.

Use a cost model such as:

```text
estimated_cost =
    launch_and_core_start_cost
  + route_count * route_barrier_cost
  + config_page_loads * config_load_cost
  + scalar_gather_elements * scalar_copy_cost
  + source_tile_count * tile_read_cost
  + output_tile_count * tile_write_cost
  + duplicated_halo_elements * halo_cost
  + sfpu_operations * sfpu_cost
```

Prefer larger chunks until each core has enough work to amortize route overhead.

A reasonable first search is:

```text
1×1 band tiles per chunk
1×2
2×1
2×2
2×4
4×2
4×4
```

For these shapes, explicitly benchmark:

```text
--cores 4
--cores 8
--cores 16
--cores 32
--cores 48
--cores 64
```

I would expect `8–16` cores with larger chunks to be competitive with or faster than the current `32–64` core configuration, even before kernel rewrites.

---

# 8. Explicit scale routes should be fused

The 1D implementation already has inline terminal-scale logic. The 2D compute path currently runs scales as independent routes:

```text
read source tile
copy to DEST
scale
pack
write workspace
synchronize
```

This adds an entire plane pass for each scale route.

For four 1D transforms in a 2D decomposition, explicit terminal scales can create substantial extra traffic.

## Reuse the 1D scale policy

Fuse a terminal scale into the final predict/update output:

```cpp
stencil(...);

if constexpr (InlineTerminalScale) {
    scale_tile(output_dst, scale_bits);
}
```

For a stream not updated by the final route, either:

* scale it during terminal band write;
* or retain one explicit scale path if FP32 operation order requires it.

This removes:

* one source read;
* one compute tile copy;
* one pack;
* one local workspace write;
* one route barrier

per fused scale route.

---

# 9. Coefficients should be compile-time

The compute source includes the complete generated `Scheme`, but still loads coefficients from runtime arguments:

```cpp
coefficients[coefficient] =
    get_arg_val<uint32_t>(coefficient_arg_base + coefficient);
```

This occurs each time `run_stencil()` is entered.

It is not the main bottleneck, but it is unnecessary. The 1D kernel uses:

```cpp
Step::coeff_bits
```

directly.

Use:

```cpp
run_stencil<Step::k, Vertical>(
    tile_count,
    ...,
    Step::coeff_bits);
```

This improves specialization and allows the compiler to propagate constants into the SFPI body.

---

# 10. Measure the fixed dispatch floor

`execute_lwt_2d()` performs:

```cpp
EnqueueMeshWorkload(...)
Finish(...)
```

inside the measured interval. 

That is correct for device execution latency, but you need to determine how much of the `17.17 ms` is:

* host enqueue/dispatch;
* kernel startup;
* actual reader execution;
* SFPU compute;
* writer execution;
* final synchronization.

## Required microbenchmarks

### A. Empty prepared workload

Same number of cores, kernels immediately return:

```text
T_empty
```

### B. Split-only workload

Run only local EE/EO/OE/OO construction:

```text
T_split
```

### C. Route-copy workload

Perform source/base/output movement without SFPU arithmetic:

```text
T_movement
```

### D. Compute-only workload

Feed preconstructed CB tiles and run stencil kernels:

```text
T_compute
```

### E. Final write-only workload

Write four local planes to DRAM:

```text
T_write
```

Then estimate:

```text
split cost    = T_split - T_empty
movement cost = T_movement - T_empty
compute cost  = T_compute - T_empty
write cost    = T_write - T_empty
```

Do not subtract `T_empty` from published transform results, but use it to decide whether kernel optimization can realistically beat PyWavelets at these sizes.

Also compare:

```text
db1
db7
bior3.9
synthetic_k17
```

If all remain near `17 ms`, the fixed protocol/dispatch path dominates. If time increases significantly with route count, route staging and synchronization dominate.

---

# Recommended optimization order

## Phase 1: inexpensive changes

1. Benchmark core counts `1, 4, 8, 16, 32, 64`.
2. Change planner priority from maximum cores to estimated latency.
3. Move coefficient bits to compile time.
4. Fuse terminal scales.
5. Load config once per chunk.
6. Replace route output word-copy with full-tile local NoC copy.
7. Replace final band row fragments with full-tile DRAM writes.

These changes preserve the current architecture.

## Phase 2: essential dataflow rewrite

8. Add aligned route fast path using full-tile source/base transfers.
9. Keep scalar gather only for misaligned routes.
10. Implement tile-native `64×64 -> EE/EO/OE/OO` split.
11. Separate interior and symmetric-edge kernels.
12. Batch NoC operations and use one barrier per batch.

This is the phase most likely to produce the required multi-fold speedup.

## Phase 3: deeper fusion

13. Pack compute output directly into workspace-backed buffers.
14. Introduce route tile pipelining.
15. Fuse adjacent route processing where the required tile working set fits DEST/L1.
16. Autotune chunk geometry per scheme and shape.

---

# What performance should you expect?

The current version is an MVP whose dominant work is orchestration and scalar data motion. Its `17 ms` latency is not representative of the potential SFPU stencil performance.

For your current sizes, a realistic first target is:

```text
current:          ~17.2 ms
after fast copies,
scale fusion,
config reduction: 6–10 ms

after tile-native
split and aligned
route staging:     2–5 ms
```

The lower end is not guaranteed, especially on small images where dispatch remains important. But without replacing scalar tile construction and tiny NoC transfers, beating PyWavelets is unlikely.

## Bottom line

The primary bottleneck is **not the horizontal or vertical SFPU stencil**.

It is this path:

```text
scalar split2d
-> scalar construction of three CB tiles per route tile
-> scalar CB-to-workspace copy
-> route synchronization
-> scalar fragmented final writes
```

Your best architectural target is:

```text
full raw tiles
-> SFPU split2d
-> direct full-tile source/base staging
-> SFPU stencil
-> direct full-tile workspace persistence
-> direct full-tile band writes
```

That transition changes the implementation from a correctness-oriented MVP into an actual tile-native Tenstorrent kernel.

# Optimize the 2D LWT split2d path

The current single-level forward 2D LWT implementation is correct, but its
performance is significantly slower than PyWavelets. The broader performance
requirements and optimization ideas are already included elsewhere in this
prompt.

For this task, focus specifically on the initial input-to-polyphase split
performed by the 2D reader kernel.

## Current problem

The current implementation in:

```text
tt-wavelet/kernels/dataflow/lwt_2d_reader.cpp
```

constructs the initial EE, EO, OE, and OO planes through four separate calls to
`initialize_plane(...)`.

The existing path:

* clears each complete tile-aligned plane through scalar RISC-V stores;
* initializes EE, EO, OE, and OO independently;
* rereads or regathers input data separately for each plane;
* performs scalar per-element tiled indexing;
* uses many small NoC reads;
* executes frequent `noc_async_read_barrier()` calls;
* retains a scalar element-by-element boundary path.

This is appropriate for correctness, but it is not a tile-native performance
implementation.

The objective is to design and implement a substantially faster `split2d`
operation that transforms a tiled raw input region directly into the four
polyphase planes:

```text
EE[r, c] = input[2r,     2c]
EO[r, c] = input[2r,     2c + 1]
OE[r, c] = input[2r + 1, 2c]
OO[r, c] = input[2r + 1, 2c + 1]
```

The implementation must preserve symmetric boundary behavior based on the
original logical input dimensions.

---

# Mandatory investigation before implementation

Before choosing an implementation, inspect:

```text
tt-isa-documentation/
```

and any relevant TT-Metal, SFPI, SFPU, unpacker, packer, tile transpose,
permutation, shuffle, row-shift, face manipulation, and register movement
documentation available in the repository.

Determine whether the SFPU can efficiently perform the required parity
deinterleave.

Do not assume that SFPU is automatically the correct solution.

Specifically investigate whether the available ISA and SFPI abstractions can
efficiently perform operations equivalent to:

```text
take even rows
take odd rows
take even columns
take odd columns
repack selected lanes into dense 32x32 output tiles
```

Relevant operations may include, depending on what actually exists:

* lane shuffle or permutation;
* vector shift;
* register transpose;
* face transpose;
* row or column broadcast;
* destination-register remapping;
* unpacker address manipulation;
* tile transpose;
* packer configuration;
* SFPU conditional moves;
* direct manipulation of tile faces;
* custom unpack or pack layouts.

Only use operations that are actually supported by the checked-in
documentation and current TT-Metal API.

Document the exact relevant ISA/SFPI operations you found and explain why they
are or are not suitable.

---

# Required design comparison

Evaluate at least the following implementation strategies.

## Option A: SFPU split kernel

Investigate a compute kernel that consumes a `64x64` logical raw region,
represented by four `32x32` raw tiles:

```text
raw00 raw01
raw10 raw11
```

and produces four dense `32x32` tiles:

```text
EE EO
OE OO
```

Conceptually:

```text
four raw input tiles
    -> unpack into DEST
    -> row/column parity deinterleave
    -> pack EE, EO, OE, OO
```

Determine:

* whether four raw FP32 tiles can be held or processed with the available DEST
  register capacity;
* whether the split must be implemented in multiple passes;
* how faces and lanes map after parity extraction;
* whether output can be packed directly into the four local planes;
* how many unpack, SFPU, and pack operations are required;
* whether the operation is likely to be compute-bound or data-movement-bound.

## Option B: unpacker/packer-driven split

Investigate whether parity selection can be implemented more efficiently by
configuring the unpacker or packer rather than running general SFPU arithmetic.

Possible ideas to validate against actual hardware capabilities include:

* unpacking only selected rows or faces;
* remapping source addresses;
* packing selected lanes into dense output tiles;
* using tile or face transpose operations;
* using two narrow-tile passes;
* creating temporary `32x16` tiles from alternating columns.

Do not claim support for these mechanisms unless verified in the repository
documentation or APIs.

## Option C: optimized data-movement split

If SFPU or packer-based splitting is not efficient or practical, implement a
high-performance data-movement kernel instead of preserving the current scalar
path.

The optimized data-movement design should:

* load large contiguous input segments;
* avoid one-value or very small NoC reads;
* read each raw input tile at most once where possible;
* avoid initializing EE, EO, OE, and OO independently;
* process the four polyphase outputs in one fused traversal;
* batch NoC reads;
* use one barrier after multiple asynchronous reads;
* use direct L1 copies where possible;
* avoid repeated calls to `tiled_element_offset()` inside the innermost loop;
* use precomputed tile/face offsets;
* separate the dense interior path from boundary handling.

A hybrid solution is allowed:

```text
interior macro-tiles -> fast SFPU/unpacker/data-movement split
edge macro-tiles     -> conservative symmetric gather path
```

This is likely preferable to forcing one generic implementation onto every
tile.

---

# Target architecture

The preferred interior unit is one `64x64` raw macro-tile:

```text
raw tile (0,0) | raw tile (0,1)
---------------+---------------
raw tile (1,0) | raw tile (1,1)
```

It produces:

```text
EE 32x32
EO 32x32
OE 32x32
OO 32x32
```

The implementation should read the four raw tiles once and produce all four
polyphase tiles before moving to the next macro-tile.

The fast path must not:

* call `symmetric_index()` for interior elements;
* issue per-element NoC reads;
* execute a barrier after each small read;
* clear full output planes element-by-element;
* initialize each polyphase plane in a separate complete traversal.

Physical tile padding and mathematical boundary handling must remain separate.

The logical input shape remains:

```text
H x W
```

The physical storage shape is:

```text
round_up(H, 32) x round_up(W, 32)
```

Symmetric mapping must always use the logical dimensions `H` and `W`. Physical
zero-padding values must never be interpreted as samples of the mathematical
signal.

---

# Boundary strategy

Implement two paths.

## Interior path

A macro-tile is interior only when every raw sample required by its EE, EO, OE,
and OO outputs lies inside the logical input domain.

The interior path should use only direct tile reads and fast deinterleave
operations.

## Boundary path

For left, right, top, bottom, and corner macro-tiles:

* use symmetric mapping in original logical coordinates;
* preserve exact current correctness;
* avoid penalizing the interior path;
* batch reads whenever reflected addresses remain contiguous;
* retain a scalar fallback only for genuinely irregular reflected fragments.

The planner or reader should classify macro-tiles as:

```text
interior
top edge
bottom edge
left edge
right edge
corner
```

A simpler interior-versus-boundary classification is acceptable for the first
version.

---

# Required implementation work

Implement the selected approach in the repository.

The expected changes may include:

```text
tt-wavelet/kernels/dataflow/lwt_2d_reader.cpp
tt-wavelet/kernels/compute/lwt_2d_split_compute.cpp
tt-wavelet/kernels/sfpi/lwt_2d_split_sfpi.h
tt-wavelet/kernels/primitives/tile_2d_layout.hpp
tt_wavelet/include/device_protocol/lwt_2d_config.hpp
tt_wavelet/include/lifting/device_2d.hpp
```

The exact file structure may differ if a cleaner design is found.

Do not merely provide pseudocode or update documentation.

The resulting code must compile and execute on the supported Tenstorrent
architecture.

---

# Preserve the existing interface where practical

The current downstream lifting scheduler expects four initial local planes:

```text
EE
EO
OE
OO
```

Preserve this contract initially unless changing it provides a clear and
measured performance benefit.

The initial optimization should not require rewriting the entire vertical and
horizontal route scheduler.

The new split stage may use:

* dedicated input circular buffers;
* dedicated output circular buffers;
* temporary L1 scratch;
* a dedicated compute kernel;
* compile-time tiling metadata;
* per-core macro-tile descriptors.

Account for all additional L1 memory explicitly.

---

# Correctness requirements

The optimized split must be validated independently before running the complete
LWT.

Add a dedicated split test that compares the device split against a scalar
reference for:

```text
1x1
1x7
7x1
2x2
2x3
3x2
15x17
31x31
31x32
32x31
32x32
32x33
33x32
33x33
63x63
63x64
64x63
64x64
64x65
65x64
65x65
1000x100
```

Test inputs should include:

* zeros;
* constants;
* increasing row-major sequence;
* row ramp;
* column ramp;
* checkerboard;
* corner impulses;
* impulses on row 31, row 32, column 31, and column 32;
* bounded deterministic random values.

For every case, validate all four planes:

```text
EE
EO
OE
OO
```

The split should preferably be bit-identical to the scalar FP32 reference,
because it is primarily a data rearrangement operation.

Any difference caused by arithmetic should be explained. Tilization or lane
reordering alone must not introduce numerical error.

Then run full 2D LWT correctness tests for at least:

```text
db1
db7
bior3.9
synthetic_k17
```

The full transform must remain within the existing FP32 tolerance policy.

---

# Benchmark requirements

Add split-stage microbenchmarks so its cost can be measured separately from the
complete LWT.

Measure at least:

```text
current scalar initialize_plane split
new optimized split
```

for:

```text
64x64
128x128
256x256
512x512
1000x100
1000x200
1024x1024
```

Report:

* total split latency;
* input elements per second;
* raw input bytes read;
* local output bytes written;
* number of NoC read calls;
* number of NoC read barriers;
* interior macro-tile count;
* boundary macro-tile count;
* active core count;
* per-core macro-tile count.

Also benchmark complete `db7` 2D LWT using the same input shapes as the existing
PyWavelets comparison.

The complete transform benchmark must reuse a prepared executable and enabled
program cache. Input upload and output readback must remain outside the timed
region, matching the existing benchmark methodology.

---

# Performance acceptance criteria

The optimization is considered successful only if it provides a measurable
improvement.

Minimum acceptance:

```text
new split latency < 0.5 * current split latency
```

for sufficiently large interior-dominated inputs.

Preferred target:

```text
new split latency < 0.25 * current split latency
```

The complete `db7` transform should show a clear improvement over the current
approximately 17 ms latency plateau.

Do not claim success based only on fewer source lines or theoretical operation
count.

Report measured before-and-after numbers.

---

# Implementation priority

Use this order:

```text
1. Inspect tt-isa-documentation and relevant SFPI/TT-Metal code.
2. Document available parity-deinterleave mechanisms.
3. Build a standalone prototype for one 64x64 raw macro-tile.
4. Compare SFPU, unpacker/packer, and data-movement approaches.
5. Select the approach based on measured performance and implementation risk.
6. Add an independent split correctness test.
7. Implement the interior fast path.
8. Add the symmetric boundary fallback.
9. Integrate the split with the existing five-plane 2D LWT scheduler.
10. Run full 2D correctness tests.
11. Run split and end-to-end benchmarks.
12. Report the remaining bottlenecks.
```

---

# Required final report

After implementation, provide:

```text
1. Root cause of the old split performance.
2. Relevant SFPU/ISA operations found in tt-isa-documentation.
3. Whether SFPU was selected and why.
4. Rejected approaches and their limitations.
5. Exact files added and modified.
6. Description of the new interior split dataflow.
7. Description of the boundary fallback.
8. L1 and circular-buffer memory usage.
9. Exact build commands.
10. Exact correctness test commands.
11. Exact benchmark commands.
12. Before-and-after split latency.
13. Before-and-after complete db7 latency.
14. Remaining dominant bottleneck.
```

Do not stop after analysis. Implement and benchmark the best practical approach.

Do not modify unrelated working 1D LWT or ILWT behavior.

Do not weaken existing correctness thresholds to obtain a performance pass.
