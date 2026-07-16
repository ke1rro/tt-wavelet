Use the `$hpc-modern-cpp` skill for this task.

Work in the `tt-wavelet` repository and investigate whether the current FP32 SFPI horizontal lifting stencil can be redesigned to operate natively on Tensix tile/face layout, or otherwise avoid the current route-by-route row-major ↔ tile repacking and the `17 - K` padding requirement.

This is a research-and-implementation task. Do not begin by assuming that the previously suggested hybrid/tile-native workspace is the correct solution. First inspect the implementation, Wormhole ISA documentation, TT-Metal code, and all supported wavelet schemes. Compare multiple technically plausible designs, select the best one based on evidence, and then implement and benchmark it.

Do not create a commit.

## Repository and documentation sources

Primary project:

```text
/home/user/tt-wavelet/tt-wavelet
```

ISA documentation:

```text
/home/user/tt-isa-documentation
```

TT-Metal source:

* Use `TT_METAL_ROOT` if set.
* Otherwise locate the local `tt-metal` checkout used by this project.

Also inspect project documentation and scheme definitions, especially:

```text
docs/
wavelets/
```

Relevant project files include, but are not limited to:

```text
kernels/sfpi/horizontal_stencil_sfpi.h
kernels/dataflow/lwt_cone_reader.cpp
kernels/dataflow/lwt_cone_writer.cpp
tt_wavelet/include/lifting/
tt_wavelet/src/lifting/
```

Relevant local documentation likely includes:

```text
tiles.rst
compute_engines_and_dataflow_within_tensix.rst
SFPI.md
03_SFPU_REGISTER_MODEL.md
04_SFPI_PROGRAMMING_MODEL.md
tensor_layouts.md
HORIZONTAL_STENCIL.md
SHIFT.md
METALIUM_GUIDE.md
```

Search TT-Metal for production examples involving:

* tilize/untilize;
* tile-native sharded buffers;
* SFPU kernels that shift or combine lanes across faces or tiles;
* sliding-window and convolution readers;
* halo handling;
* pack/unpack configuration;
* direct L1 tile reads and writes;
* DST register addressing;
* `SFPSHFT2`, shuffle, rotate, lane masking, and related Wormhole instructions;
* kernels that retain intermediate data in tile layout across multiple operations.

TT-Metal convolution code is a useful architectural reference, but do not assume its high-level convolution operator should be reused directly.

## Current implementation and suspected bottleneck

The compute kernel already executes in tile/DST layout and performs FP32 arithmetic on SFPU.

The current intermediate workspace is compact row-major L1:

```text
row-major L1
    → reader scalar packing
    → tile CB
    → DST registers and SFPU
    → tile CB
    → writer
    → row-major L1
```

For each predict/update group producing 1536 useful output values, the current data movement is approximately:

```text
source packing:      2048 FP32 values
base packing:        1536 FP32 values
output materialize:  1536 FP32 values
total:               5120 FP32 values
```

The source region contains only about 1552 unique values, but the reader packs 2048 because overlapping stencil rows repeatedly copy about 496 values.

The writer has already been improved: approximately 1536 scalar copies were replaced with aligned 64-byte NoC writes. The remaining major cost appears to be the reader-side scalar packing.

For `db7`, most lifting steps have `K = 1` or `K = 2`, so the SFPU arithmetic is small and packing can cost more than the stencil itself.

However, a simple switch to `TILE_LAYOUT` may not solve the problem:

* routes have different `source_offset` and `base_offset`;
* offsets may differ by one FP32 value;
* the source representation currently requires halo and four half-blocks per row;
* compact output contains only three half-blocks per row;
* the next route may require a different alignment;
* the current SFPI algorithm relies on a particular input arrangement and `17 - K` left padding;
* tile faces, subvector boundaries, even/odd columns, and cross-tile halo behavior must remain correct.

The core research question is therefore:

> Can the horizontal SFPI lifting stencil and its persistent workspace be redesigned so that intermediate values remain in a native or near-native tile representation across lifting routes, while preserving FP32 SFPU precision and avoiding most or all route-level scalar repacking?

A second, related question is:

> Can the stencil itself be reformulated so that `17 - K` padding is unnecessary, or replaced by a much cheaper representation such as register-level halo injection, shifted tile views, masks, or limited boundary handling?

## Hard constraints

Preserve these invariants:

* FP32 storage and FP32 SFPU arithmetic.
* Do not replace SFPU FP32 computation with TF32, BF16, FP16, or FPU matmul.
* Preserve generic predict/update support.
* Preserve all supported coefficient lengths, including `K = 1` through the maximum currently supported value, expected to be 17.
* Preserve arbitrary scheme shifts and route geometry.
* Preserve metadata-only swap semantics.
* Preserve ConeStreamed execution and terminal direct-to-DRAM output.
* Do not reintroduce route-by-route DRAM loopback.
* Do not hard-code `db7` or any individual wavelet into the compute path.
* Do not require users to provide manually transformed layouts.
* Boundary behavior and finite-signal correctness must remain consistent with the current backend.
* Avoid increasing L1 usage so much that useful chunk size or active-core count materially decreases.
* Do not optimize only the SFPI instruction sequence while leaving a larger layout conversion bottleneck untouched.

## Required initial investigation

Before implementing a design, reconstruct the exact current physical dataflow.

Document, with index and address examples:

1. How one logical row of source, base, and output is represented in:

   * compact row-major workspace;
   * circular buffers;
   * tile faces;
   * DST registers;
   * SFPU lanes.

2. Why the current SFPI stencil expects two source tiles and two base/output tiles.

3. Why the source uses four half-blocks per row while the output contains three.

4. The exact meaning of the current `17 - K` padding:

   * which logical tap it aligns;
   * which tile column or lane it aligns with;
   * whether it is mathematically necessary or only an implementation artifact;
   * whether it is needed for all `K` or only particular parity/shift combinations.

5. How `_horizontal_stencil_rotate_()` moves values:

   * within a subvector;
   * across subvectors;
   * across faces;
   * across tiles;
   * which transitions it cannot express.

6. The precise mapping implemented by:

```cpp
_get_dst_base()
_horizontal_stencil_rotate_()
_horizontal_stencil_plus_base_block()
_horizontal_stencil_plus_base_face()
_horizontal_stencil_plus_base()
```

7. For every supported wavelet scheme, collect:

   * number of predict/update steps;
   * distribution of `K`;
   * shift range;
   * source/base offset alignment;
   * frequency of one-element misalignment;
   * swap count;
   * maximum halo;
   * routes that violate any proposed tile-native alignment rule.

Do not infer these properties from `db7` alone. Scan all schemes under `wavelets/`.

## ISA research questions

Use the Wormhole ISA documentation to answer, with citations to local files or instruction definitions:

1. Can SFPU load directly from a shifted logical view of a tile without physically repacking the data?

2. Can `SFPSHFT2` or another instruction construct a non-wrapping shift using values from:

   * another LREG;
   * another face;
   * another tile;
   * a preloaded halo register?

3. Can the first or last lane of each subvector be replaced efficiently using masks or conditional moves?

4. Can pack/unpack configuration expose a shifted or overlapping tile view without scalar data movement?

5. Can DST addressing or address modifiers represent the required `+1 FP32` route offsets?

6. Can a tile-native persistent workspace be read and written without tilize/untilize between routes?

7. Can source and base be stored in the same physical tile representation while still allowing the three-slot `A/B/Scratch` model?

8. Can overlapping halo columns be stored once per tile row and reused by multiple routes?

9. Can a route produce an output layout that is immediately valid as the input layout of the next route, even when the next route has a different shift?

10. Are there undocumented or subtle limitations involving:

    * face boundaries;
    * 16-lane subvectors;
    * tile column parity;
    * register allocation;
    * LREG14 configuration;
    * DST tile count;
    * synchronization between unpack, math, and pack?

Do not claim that an instruction supports a required cross-boundary behavior unless it is verified from documentation or a minimal hardware experiment.

## Candidate architecture space

Investigate at least the following classes of solutions, but do not assume any one of them is correct.

### 1. Fully tile-native persistent workspace

Store all intermediate `A/B/Scratch` streams in a face-compatible tile layout and avoid row-major materialization between lifting routes.

Determine whether arbitrary route offsets can be represented as metadata, shifted tile origins, or tile-local operations.

### 2. Tile-native body with explicit halo/prologue

Keep aligned interior data tile-native and construct only the small unaligned halo or prologue required by a route.

Determine whether this actually reduces total traffic for `K = 1`, `K = 2`, and high-`K` schemes.

### 3. SFPI stencil redesign without `17 - K` padding

Reformulate the register-level convolution so taps begin at their natural position.

Possible mechanisms to investigate include, but are not limited to:

* preloaded halo registers;
* masked lane replacement;
* pairwise tile concatenation;
* explicit first-lane handling;
* different even/odd register decomposition;
* processing taps in the opposite direction;
* changing which source half-block is considered the center;
* maintaining shifted state between taps;
* using a small prologue/epilogue rather than padding every row.

Do not implement one of these merely because it sounds plausible. Verify instruction semantics and estimate instruction cost.

### 4. Redundant persistent tile representation

Store a limited amount of duplicated overlap in each tile so common route offsets require no repacking.

Quantify:

* L1 overhead;
* extra writer traffic;
* reduction in reader work;
* whether redundancy must be regenerated after every route;
* whether different shifts require multiple redundant views.

### 5. Multiple physical views or phase layouts

Maintain two or more phase-aligned tile views, for example alignment 0 and alignment +1, if that is cheaper than rebuilding a shifted view for each route.

Determine whether the memory and update cost is justified by the actual scheme distribution.

### 6. One-time tilization at cone entry and one-time untilization at exit

Convert the loaded dependency cone to a native tile representation once, execute the complete lifting scheme in that representation, and convert only the final interior to the output format.

Determine whether intermediate route geometry can remain valid without additional repacking.

### 7. Alternative SFPI data decomposition

Investigate whether the current even/odd lane decomposition is itself creating the layout requirement.

Consider whether another mapping of logical consecutive values to:

* LREGs;
* lanes;
* faces;
* DST tiles

would make shifts and tap application simpler while preserving vectorized FP32 execution.

### 8. Current compact representation with a substantially better packer

If a persistent tile-native representation is impossible or too expensive, determine the best achievable reader design:

* aligned block NoC reads for dense regions;
* reuse of overlapping source data;
* persistent local cache across groups;
* reduced duplicated packing;
* specialized fast paths for common `K` and offset parity;
* separate interior and boundary paths.

This is the fallback, not the assumed answer.

## Required comparison

For each viable design, provide an evidence table containing:

* physical layout;
* whether `17 - K` padding remains;
* support for arbitrary `K`;
* support for arbitrary shift and route offset;
* FP32 preservation;
* source elements read per 1536-output group;
* base elements read per group;
* output elements written per group;
* duplicated values;
* SFPU instruction overhead;
* NoC operations;
* L1 bytes per slot;
* additional halo storage;
* expected active-core impact;
* implementation complexity;
* correctness risk;
* compatibility with forward 1D, inverse 1D, and future 2D LWT.

Explicitly identify whether each design solves:

```text
row-major → tile repacking
tile → row-major materialization
17 - K padding
one-FP32 route misalignment
cross-face halo
cross-tile halo
```

## Microbenchmarks

Before rewriting the complete backend, build focused hardware microbenchmarks where needed.

At minimum isolate:

1. Current reader packing only.
2. Current SFPI stencil only.
3. Current writer only.
4. Tile-native source/base load into DST.
5. A one-element shifted tile view.
6. Cross-face or cross-tile halo injection.
7. A chain of several predict/update routes without row-major materialization.
8. `K = 1`, `K = 2`, and `K = 17`.
9. Aligned and `+1 FP32` offset cases.

Measure device execution only. Exclude program construction and unrelated host work from the timing boundary.

Report median, minimum, p10, p90, and standard deviation over enough repetitions to distinguish small differences.

## Design decision

After the investigation, choose the architecture that provides the best overall reduction in data movement without sacrificing generality or FP32 correctness.

The selected design must be justified by:

* ISA evidence;
* measured microbenchmarks;
* scheme-distribution analysis;
* exact physical layout mapping;
* projected end-to-end traffic;
* expected effect on ConeStreamed performance.

Do not select a design solely because it is easiest to implement.

If no fully tile-native design is feasible, clearly explain the exact hardware or layout constraint that prevents it and implement the best evidence-backed hybrid or packing design.

## Implementation requirements

After selecting the design:

1. Implement the smallest coherent architecture that demonstrates the selected persistent layout or stencil model.
2. Remove obsolete packing paths that the new design replaces.
3. Keep boundary handling separate from the fast interior path where beneficial.
4. Keep layout and route geometry explicit in strong C++ types or descriptors.
5. Do not scatter hidden layout assumptions through reader, compute, and writer code.
6. Add assertions for alignment, tile capacity, halo size, route offsets, and supported instruction assumptions.
7. Reuse the three-slot `A/B/Scratch` execution model where it remains appropriate.
8. Preserve terminal direct-to-DRAM output.
9. Ensure the design can later be reused for inverse 1D and separable 2D LWT, but do not add speculative abstractions that are not needed by the chosen implementation.

## Correctness validation

Run at least:

* `db7`;
* one wavelet dominated by `K = 1`;
* one wavelet dominated by `K = 2`;
* one wavelet containing the maximum supported `K`;
* schemes with positive and negative shifts;
* schemes with swaps;
* odd and even signal lengths;
* short signals;
* chunk-boundary cases;
* first and last chunks;
* all 106 supported schemes for runtime stability.

Compare the new backend directly against the current correct ConeStreamed or Resident reference using the same arithmetic ordering where possible.

Track separately:

1. architecture equivalence against the existing backend;
2. compatibility against PyWavelets.

Do not attribute existing high-order factorization error to the new layout unless the new implementation increases it.

FP32 error must not regress materially relative to the current SFPU implementation.

## Performance validation

Benchmark at least:

```text
db7:
100k–1M sweep
5M
8M or the largest stable size

high-K wavelet:
representative medium and large sizes
```

Use the same device-only timing boundary for old and new implementations.

Report:

* total time;
* reader time if measurable;
* compute time if measurable;
* writer time if measurable;
* source/base/output traffic;
* scalar operations eliminated;
* speedup by signal size;
* speedup by `K`;
* L1 usage;
* active-core count;
* chunk size;
* any scheduler thresholds.

The primary performance objective is not a specific percentage. It is to prove that the selected design materially reduces repeated layout conversion and improves end-to-end execution for low-arithmetic schemes such as `db7`, without causing major regressions for high-`K` schemes.

## Required deliverables

Produce:

1. A research note under `docs/` describing:

   * current physical layout;
   * ISA findings;
   * scheme statistics;
   * candidate comparison;
   * selected architecture;
   * rejected alternatives and reasons.

2. The implementation of the selected design.

3. Focused correctness tests.

4. Microbenchmark and end-to-end benchmark results.

5. A concise final summary containing:

   * what was proven;
   * whether native tile persistence is feasible;
   * whether `17 - K` padding was eliminated;
   * what data movement was removed;
   * measured performance changes;
   * remaining bottlenecks;
   * exact files changed.

Do not return only a speculative plan. Inspect the sources, validate assumptions against Wormhole hardware, implement the evidence-backed solution, and report any unresolved hardware limitation honestly.
