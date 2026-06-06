Firstly check docs and limitations of the project

-[text](AGENTS.md)
-[text](docs/DEFINITIONS.md)
-[text](docs/HORIZONTAL_STENCIL.md)
-[text](docs/LWT.md)
-[text](docs/PROJECT_LIMITATIONs.md)
-[text](docs/VERTICAL_STENCIL.md)


Be logical validate the idea be precise write greate performance code and check try to refer to docs.

the Code of chunked kernel in [chunked kernel](chunk_kernel.txt)

# Conceptual Task: Use Dependency Cones to Make LWT Faster



## Goal

We want to improve the performance of the current Lifting Wavelet Transform backend.

The important goal is not to blindly rewrite the code in one specific way. The important goal is to use the dependency structure of LWT to reduce unnecessary intermediate materialization and beat the current `lwt` timing while preserving correctness.

The implementation should remain buildable and testable:

```bash
./update.sh
```

should pass.

The correctness comparison should pass:

```bash
python3 compare.py
```

or the existing equivalent comparison command in this repository.

Benchmark/timing tests should still work, and the optimized path should be compared against the current LWT backend.

---

## Core Idea: LWT as a Layered DAG

Think of the Lifting Wavelet Transform as a layered directed acyclic graph.

Each value in the even/odd streams at each lifting step is a graph node:

```text
E_i^(r) = even stream value at logical index i after step r
O_i^(r) = odd stream value at logical index i after step r
```

Each lifting step creates edges from layer `r` to layer `r + 1`.

For example, a predict step:

```text
O_new[i] = O_old[i] + stencil(E_old)[i]
```

means that `O_i^(r+1)` depends on:

```text
O_i^(r)
and several nearby E_j^(r) values
```

Similarly, an update step:

```text
E_new[i] = E_old[i] + stencil(O_old)[i]
```

means that `E_i^(r+1)` depends on:

```text
E_i^(r)
and several nearby O_j^(r) values
```

So the full transform is a layered DAG:

```text
initial E/O after padding and polyphase split
        ↓
after step 0
        ↓
after step 1
        ↓
after step 2
        ↓
final E/O
```

The existing route-based backend effectively computes this graph layer-by-layer:

```text
compute full layer 1
materialize full layer 1

compute full layer 2
materialize full layer 2

compute full layer 3
materialize full layer 3
...
```

This is correct, but it may create too much memory traffic because every intermediate layer or route can be written and reread.

---

## Dependency Cone Concept

Instead of always computing every intermediate layer globally, consider a final output chunk:

```text
final output chunk = [a, b]
```

The question is:

```text
Which initial input values are actually required to compute this final chunk?
```

Because LWT steps are local stencils, the answer is not the whole signal. It is the final chunk plus a finite halo region.

This required region is called a dependency cone.

Visually:

```text
initial layer:     [........input + halo........]
                         \              /
intermediate:          [....needed....]
                           \          /
final layer:               [a....b]
```

The cone expands backward through the steps because each stencil needs neighboring values.

The key optimization opportunity is:

```text
For each output chunk, compute only the subgraph needed for that chunk.
Avoid materializing global intermediate streams when possible.
```

---

## Important: This Is Not Explicit Path Traversal

Do not think of this as enumerating every graph path.

A naive graph algorithm would recursively visit every ancestor of every output node:

```text
for each output node:
    visit all parent nodes
    visit all grandparents
    ...
```

That would be too expensive and unnecessary.

LWT has a regular stencil structure. If the target output set is a contiguous interval, then the dependencies at every previous layer are also contiguous intervals.

So instead of storing individual graph nodes, represent the needed region at each layer by two intervals:

```text
D_E^(r) = required even interval before step r
D_O^(r) = required odd interval before step r
```

This compresses graph reachability into interval arithmetic.

The dependency-cone computation is therefore:

```text
backward interval propagation
```

not explicit graph traversal.

---

## Interval Propagation Rules

Let each step have:

```text
K = number of coefficients
shift = lifting step shift
```

The current backend convention appears to use this source span:

```text
source_span_lo = -shift - K + 1
source_span_hi = -shift
```

That convention should be preserved unless there is a very deliberate reason to change it. Be careful not to accidentally switch to a different convention like `i + shift + j`.

For an interval:

```text
I = [a, b]
```

and a source span:

```text
S = [u, v]
```

define interval shift/addition:

```text
I + S = [a + u, b + v]
```

### Predict

Forward meaning:

```text
O_new = O_old + stencil(E_old)
E_new = E_old
```

Backward dependency rule:

```text
D_O_before = D_O_after

D_E_before = union(
    D_E_after,
    D_O_after + [source_span_lo, source_span_hi]
)
```

Meaning:

```text
To compute the required new odd interval,
we need the same old odd interval,
plus the even interval touched by the stencil.
```

### Update

Forward meaning:

```text
E_new = E_old + stencil(O_old)
O_new = O_old
```

Backward dependency rule:

```text
D_E_before = D_E_after

D_O_before = union(
    D_O_after,
    D_E_after + [source_span_lo, source_span_hi]
)
```

Meaning:

```text
To compute the required new even interval,
we need the same old even interval,
plus the odd interval touched by the stencil.
```

### Swap

Forward meaning:

```text
E_new = O_old
O_new = E_old
```

Backward dependency rule:

```text
D_E_before = D_O_after
D_O_before = D_E_after
```

### Scale

Forward meaning:

```text
E_new = alpha * E_old
O_new = beta  * O_old
```

Scale does not introduce spatial dependencies.

Backward rule:

```text
D_E_before = D_E_after
D_O_before = D_O_after
```

Scale can often be fused into final output if it is terminal.

---

## Example

Suppose the final chunk is:

```text
[100, 199]
```

and the scheme has two steps:

```text
step 0: predict, shift = -1, K = 3
step 1: update,  shift = 0,  K = 2
```

Start from final output:

```text
D_E^2 = [100, 199]
D_O^2 = [100, 199]
```

Backward through update:

```text
shift = 0
K = 2

source_span_lo = -0 - 2 + 1 = -1
source_span_hi = -0 = 0
```

Update computes even from odd, so odd dependencies expand:

```text
needed odd source = [100, 199] + [-1, 0]
                  = [99, 199]
```

Therefore:

```text
D_E^1 = [100, 199]
D_O^1 = union([100, 199], [99, 199])
      = [99, 199]
```

Backward through predict:

```text
shift = -1
K = 3

source_span_lo = -(-1) - 3 + 1 = -1
source_span_hi = -(-1) = 1
```

Predict computes odd from even, so even dependencies expand:

```text
needed even source = [99, 199] + [-1, 1]
                   = [98, 200]
```

Therefore:

```text
D_O^0 = [99, 199]
D_E^0 = union([100, 199], [98, 200])
      = [98, 200]
```

So to compute final output `[100, 199]`, the implementation only needs initial intervals:

```text
initial even: [98, 200]
initial odd:  [99, 199]
```

The extra region around `[100, 199]` is the halo.

---

## Why This Matters for Performance

The current route-oriented backend can repeatedly do:

```text
read full stream
compute one route
write full stream
```

for every lifting step.

The dependency-cone idea suggests a different execution schedule:

```text
for each final output chunk:
    determine required input intervals
    read input + halo
    run all needed lifting steps locally
    write only final output chunk
```

This can reduce intermediate memory traffic significantly.

The mathematical transform is unchanged. Only the schedule and materialization boundaries change.

Current schedule:

```text
materialize after every step
```

Dependency-cone schedule:

```text
materialize only input and final output when possible
```

---

## Desired Outcome

The final implementation should try to beat the current LWT backend timing.

The implementation is free to choose the best architecture, but it should respect the dependency-cone insight:

```text
Do not unnecessarily compute or materialize global intermediate streams
when only a local output chunk is needed.
```

The implementation should preserve correctness against the existing backend and PyWavelets/reference comparison.

Required validation:

```bash
./update.sh
python3 compare.py
```

or repository-equivalent commands.

Required benchmark comparison:

```text
old/current LWT backend
new optimized backend
```

Important metrics to report:

```text
scheme name
signal length
backend name
chunk size or relevant tuning parameter
mean time
min time
repeats
warmup runs
```

---

## Useful Benchmarks

Try several representative schemes:

```text
haar
db1
db4
db7
bior2.2
bior3.3
bior4.4
coif1
sym8
```

Try several lengths:

```text
17
31
32
33
257
4095
4096
4097
100000
1000000
1000010
```

Small lengths catch boundary/indexing bugs.

Large lengths show performance behavior.

---

## Warnings and Pitfalls

### 1. Do not confuse dependency planning with actual computation

Backward interval propagation is a planning analysis.

The numerical transform still runs forward:

```text
initial E/O
  -> predict/update
  -> predict/update
  -> final E/O
```

### 2. Do not enumerate graph paths

The dependency cone should be represented by intervals.

Path traversal is unnecessary and would be too expensive.

### 3. Be careful with shift convention

The current backend convention appears to be:

```text
source_span_lo = -shift - K + 1
source_span_hi = -shift
```

Using a different convention can produce correct-looking but wrong outputs.

### 4. Be careful with coefficient order

The horizontal stencil convention may require reversed coefficients relative to the JSON order.

Validate coefficient order against the existing backend.

### 5. Do not write halo output

A chunk may compute halo internally, but it should write only the owned final output interval.

### 6. Be careful with swap before scale

Some schemes may have:

```text
swap
scale-even
scale-odd
```

Scale applies to the logical streams after swap.

### 7. Terminal scale should not force extra full-stream passes

If scale steps are terminal, they can usually be fused into final output.

Avoid adding separate full-stream scale routes unless absolutely necessary.

### 8. Avoid scalar bottlenecks

If an implementation reduces DRAM traffic but moves all arithmetic to slow scalar loops, it may become slower.

The optimized path should preserve or improve math throughput.

### 9. Avoid tiny chunks unless they are empirically faster

Small chunks increase overhead.

Tune chunk size empirically.

### 10. Keep the old backend available

The old backend is needed for correctness comparison and timing baseline.

Do not delete the known-good path.

---

## Success Definition

This task is successful if:

1. The repository builds.
2. `./update.sh` passes.
3. Correctness comparison passes.
4. The old backend remains available for A/B comparison.
5. The new backend uses the dependency-cone idea to avoid unnecessary global intermediate materialization.
6. Benchmarks show the new backend can beat the current LWT timing on at least meaningful large-signal cases.
7. Any remaining limitations are clearly documented.

## Preferred Execution Architecture: Reader / Compute / Writer

The dependency-cone idea describes **what data is actually needed** for each output chunk.

However, to make this fast on Tenstorrent hardware, the implementation should also use the normal TT-Metal execution split:

```text
reader  ->  compute  ->  writer
```

The conceptual responsibilities are:

```text
reader:
    data movement only
    read original input + halo
    apply boundary mapping
    build local even/odd working representation

compute:
    numerical lifting work
    run predict/update steps
    use SFPU / compute engine for stencil arithmetic
    keep intermediate streams local when possible

writer:
    data movement only
    write final owned output interval
    do not write halo output
```

The important performance idea is:

```text
Do not move LWT arithmetic into scalar dataflow loops.
The reader and writer should move data.
The compute kernel should do the stencil math.
```

A slow implementation can reduce DRAM traffic but still lose performance if it performs all predict/update work as scalar `float` loops in a data-movement kernel.

The optimized direction should preserve the dependency-cone insight while also preserving math throughput:

```text
dependency cone:
    reduces unnecessary global intermediate materialization

reader/compute/writer split:
    keeps data movement and math on the right engines

SFPU compute:
    preserves high-throughput stencil arithmetic
```

---

## Why the Compute Kernel Matters

Each predict/update step is a local linear stencil.

Predict:

```text
odd = odd + P(even)
```

Update:

```text
even = even + U(odd)
```

Both have the same primitive form:

```text
target = base + stencil(source)
```

where:

```text
source = opposite parity stream
base   = stream being updated
target = updated stream
```

This is exactly the kind of work that should happen in the compute kernel, not in a data-movement kernel.

The compute path should use the existing SFPU-oriented horizontal stencil primitive where possible.

Conceptually:

```text
source tiles + base tiles
    -> SFPU horizontal stencil
    -> output tiles
```

rather than:

```text
for each scalar index:
    sum = 0
    for each coefficient:
        sum += coeff * source[index + offset]
    base[index] += sum
```

The scalar version is useful as a reference, but the optimized backend should aim to use the compute engine for the actual convolution/stencil work.

---

## Mapping LWT Steps to SFPU Stencil Compute

The central compute primitive is:

```text
output = base + horizontal_stencil(source)
```

This maps naturally to lifting steps:

```text
Predict:
    source = even
    base   = odd
    output = updated odd

Update:
    source = odd
    base   = even
    output = updated even
```

So the compute kernel can think in terms of a generic step:

```text
run_stencil_step(source_stream, base_stream, output_stream, coefficients, shift)
```

The exact implementation can vary, but the important idea is:

```text
Use the existing SFPU/tile horizontal stencil machinery for the convolution.
Do not recompute the convolution using scalar dataflow loops.
```

The current horizontal-stencil convention may differ from the abstract math notation. In particular, the existing backend appears to use a convention equivalent to:

```text
source_span_lo = -shift - K + 1
source_span_hi = -shift
```

and may require reversed coefficient order relative to the JSON scheme.

Therefore:

```text
Preserve the existing backend's shift and coefficient-order convention.
Validate against the current route backend or scalar reference.
```

This is one of the easiest places to accidentally produce numerically wrong results.

---

## Combining Dependency Cones with SFPU Compute

The dependency cone tells us the local input region required for a final chunk.

For each output chunk:

```text
1. Host/planner determines the required initial even/odd intervals.
2. Reader loads the required original input + halo.
3. Reader builds local even/odd tile pages.
4. Compute runs all lifting steps forward using SFPU stencil operations.
5. Writer writes only the owned final output interval.
```

The important separation is:

```text
Backward interval propagation:
    planning problem

Forward predict/update execution:
    numerical compute problem
```

The compute kernel should not perform graph/path traversal.

Ideally, dependency-cone planning is done before or outside the inner numeric compute loop. The compute kernel should receive enough metadata to process its local chunk efficiently.

---

## Materialization Boundaries

The current route-oriented backend may materialize after each route:

```text
input
  -> route 0 output
  -> route 1 output
  -> route 2 output
  -> ...
  -> final output
```

The dependency-cone optimized direction tries to move materialization boundaries closer to:

```text
input
  -> local chunk working set
  -> final output
```

In other words:

```text
Intermediate lifting states should stay local when possible.
They should not be repeatedly written to and reread from global memory.
```

This is the main architectural reason dependency cones may improve performance.

---

## Reader Responsibilities in the Dependency-Cone Backend

The reader should focus on data movement and local layout construction.

Conceptually:

```text
reader input:
    original signal buffer
    chunk configuration
    required input interval / halo
    padding/boundary configuration

reader output:
    local even tiles
    local odd tiles
```

The reader should:

```text
read input using aligned stick/page reads when possible
reuse existing stick-cache ideas if useful
apply symmetric boundary mapping
perform polyphase split
place values into local even/odd tile representation
```

Avoid an implementation that performs many random one-float NoC reads. That may destroy the benefit of the dependency-cone schedule.

The reader does not need to understand the full lifting computation. It only needs to prepare the local initial streams for the compute kernel.

---

## Compute Responsibilities in the Dependency-Cone Backend

The compute kernel should own the numerical lifting work.

Conceptually:

```text
compute input:
    local even/odd tiles from reader
    scheme steps
    coefficients
    shifts
    local stream geometry

compute output:
    final active even/odd local tiles
```

For each predict/update step, the compute kernel should run:

```text
target = base + stencil(source)
```

using the SFPU/tile horizontal-stencil implementation.

The compute kernel should try to keep intermediate even/odd streams in local L1/CB storage.

A useful mental model:

```text
even_current
even_free

odd_current
odd_free
```

For predict:

```text
odd_free = odd_current + stencil(even_current)
odd_current becomes odd_free
even_current remains unchanged
```

For update:

```text
even_free = even_current + stencil(odd_current)
even_current becomes even_free
odd_current remains unchanged
```

This avoids unnecessary copies of the stream that was not updated.

The implementation does not have to follow this exact structure, but it should avoid introducing expensive local copies or global materialization unless required for correctness.

---

## Writer Responsibilities in the Dependency-Cone Backend

The writer should only write final output.

Conceptually:

```text
writer input:
    final active even/odd local tiles
    output interval ownership
    final scale metadata

writer output:
    global approximation/detail buffers
```

The writer should:

```text
write only the owned final output interval
clip partial chunks correctly
avoid writing halo output
preserve canonical output ordering
```

If scale steps are terminal, scale can be fused into the final output stage.

Possible choices:

```text
compute applies final scale before producing final tiles
or
writer applies final scale while writing
```

The important point is:

```text
terminal scale should not force extra full-stream passes
```

---

## SFPU Use for Current Convolution / Stencil

The existing convolution-like operation in LWT is the shifted horizontal stencil.

The optimized backend should try to reuse the current SFPU stencil machinery instead of replacing it with scalar loops.

The desired compute-level operation is:

```text
load source tile(s)
load base tile(s)
run SFPU horizontal stencil:
    output = base + stencil(source)
store output tile(s)
```

This is especially important because the dependency-cone schedule changes memory behavior, but performance will only improve if math throughput remains high.

Bad direction:

```text
less DRAM traffic
but scalar arithmetic in dataflow kernel
```

Better direction:

```text
less DRAM traffic
and stencil arithmetic still runs on SFPU / compute engine
```

The implementation is free to decide the exact tiling, buffering, and scheduling strategy, but it should preserve this principle.

---

## Desired Optimized Shape

A good optimized design should look conceptually like:

```text
for each core/chunk:
    reader:
        read input interval + halo
        build local E/O tiles

    compute:
        for each lifting step:
            if predict:
                odd = odd + stencil(even) using SFPU
            if update:
                even = even + stencil(odd) using SFPU
            if swap:
                swap logical stream roles
            if terminal scale:
                defer or fuse into final output

    writer:
        write final owned output interval only
```

This is still the same mathematical LWT.

Only the schedule and materialization strategy change.

---

## Additional Warnings

### Avoid scalar bottlenecks

A dependency-cone implementation can still be slower if the arithmetic moves from SFPU/tile compute into scalar dataflow loops.

Reducing memory traffic is not enough by itself.

The optimized path should preserve or improve math throughput.

### Avoid one-kernel monoliths for the final optimized path

A single dataflow kernel that reads, computes, and writes is simple, but it is unlikely to be the best performance target.

It is fine as a reference, but the optimized path should prefer a reader/compute/writer split.

### Keep coefficient and shift conventions consistent

The existing horizontal-stencil path may use reversed coefficient order and a specific shift convention.

Validate against the existing backend carefully.

### Keep the old backend available

The known-good route backend is still valuable as:

```text
correctness oracle
timing baseline
regression fallback
```

Do not remove it while experimenting.

### Document limitations

If the optimized backend initially supports only some schemes, chunk sizes, or aligned lengths, document that clearly.

But the long-term direction should be:

```text
dependency-cone chunking
reader/compute/writer split
SFPU stencil compute
terminal scale fusion
```