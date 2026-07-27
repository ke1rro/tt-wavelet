# 2D LWT bounded route-staging report

Date: 2026-07-27  
Device: Wormhole N150  
Primary case: forward `db7`, symmetric boundary, FP32, `1000x100`

Labels used below:

- **Fact**: follows directly from code, ISA/API contracts, or exact counts.
- **Measured**: observed on the N150 used for this task.
- **Derived**: calculated from facts or measurements.
- **Hypothesis**: a proposed explanation or next optimization that still needs an A/B measurement.

## 1. Executive conclusion

**Measured.** The production exact-domain transform improved from
`11.132515 ms` to `2.239485 ms` mean for `db7 1000x100`, a `4.867x`
speedup and a `79.45%` latency reduction.

**Measured.** Critical-path source/base staging fell from approximately
`11,340,166` cycles to `1,652,236` cycles. The new value is `14.57%` of
the old value, an `85.43%` reduction, and is below the requested 3-million
cycle target.

The selected production solution is:

```text
exact dependency domain
-> persistent same-core zero tile
-> one request classification
-> batched zero/full-tile reads
-> bounded face-row copies for the valid intersection
-> unchanged SFPU lifting route
```

The tile-closed dependency policy was implemented and validated, but it is not
the default. For this case it expanded initial work by `7.025x`, route work by
`2.658x`, L1 use by `2.417x`, and latency from `2.239 ms` to `10.446 ms`.

**Measured.** The corrected warmed PyWavelets FP32 result is `1.776859 ms`
mean, not the earlier approximately `2.84 ms` figure obtained from a float64
comparison. TT device-only is therefore `1.260x` slower on `1000x100`;
transfer-inclusive TT is `1.396x` slower. TT wins on the larger tested cases:
`1.696x` at `512x512` and `3.968x` at `1024x1024`.

## 2. Scope and changed files

The implementation changed:

```text
compare_timings.py
scripts/validate_lwt_2d_device.py
tt-wavelet/kernels/dataflow/lwt_2d_reader.cpp
tt-wavelet/main_2d.cpp
tt-wavelet/tt_wavelet/include/device_protocol/lwt_2d_config.hpp
tt-wavelet/tt_wavelet/include/lifting/device_2d.hpp
tt-wavelet/tt_wavelet/include/lifting/execution_plan.hpp
tt-wavelet/tt_wavelet/include/lifting/plan_2d.hpp
tt-wavelet/tt_wavelet/src/lifting/device_2d.cpp
docs/LWT_2D_BOUNDED_STAGING_REPORT.md
```

The already optimized split, route persistence, tiled terminal writer,
preloaded route configuration, scale fusion, and exact local-NoC transfer
remain the production choices.

`backpropagate_requirements()` gained an optional closure extent, defaulting to
zero. The 1D planner therefore retains its old behavior. The 1D target builds,
and the vertical K=1/2/5/9/13/14/17 and dense horizontal K=1/2/9/17 device
stencil tests pass bit-identically.

The timing script now:

- constructs the 2D PyWavelets input as `numpy.float32`;
- supports discarded PyWavelets warmups;
- exposes `--tt-route-domain exact|tile-closed`.

## 3. Geometry, root cause, and tile closure

### 3.1 Why partial and empty requests occurred

**Fact.** A route output is computed as a complete `32x32` tile, but the stored
dependency cone is an exact logical rectangle. A requested source/base tile is
partial whenever that complete tile crosses the stored rectangle boundary. It
is empty when the complete request does not intersect the stored rectangle.

The old generic path cleared 1024 words and performed `contains()`,
tile-address calculation, L1 load, and L1 store per lane. On `db7 1000x100`,
that path handled 91.5% of staging requests.

### 3.2 Exact fallback valid-area histogram

The histogram excludes route-output records and includes the 5,668 partial or
empty source/base requests.

| Valid FP32 lanes in requested tile | Requests |
|---:|---:|
| 0 | 870 |
| 960 | 548 |
| 544 | 420 |
| 192 | 363 |
| 992 | 261 |
| 224 | 258 |
| 512 | 180 |
| 180 | 152 |
| 160 | 120 |
| 150 | 120 |
| 30 | 90 |
| 120 | 90 |
| 90 | 90 |
| 60 | 90 |
| 186 | 78 |
| 42 | 75 |
| 36 | 75 |
| 128 | 75 |
| 96 | 75 |
| 64 | 75 |
| Other 77 areas | 1,563 |

**Derived.** Many requests are nearly full: 809 requests contain either 960
or 992 useful lanes. This supports copying a bounded rectangle rather than
executing a 1024-lane scalar gather.

### 3.3 Tile-closed propagation algorithm

`--route-domain tile-closed` performs this operation independently on the
vertical and horizontal finite lifting streams:

1. Convert exact final ownership to final even/odd stream intervals.
2. Round each non-empty interval begin down and end up to a multiple of 32.
3. Clip the rounded interval to the mathematically defined finite stream
   length.
4. Traverse the lifting routes backward.
5. For predict/update, translate the required output by the existing source
   and base offsets, include the `K-1` source extent, and take the hull with
   the unchanged stream.
6. For scale and swap, apply their existing inverse dependency transition.
7. Close each predecessor even/odd interval to 32 and repeat.
8. Form EE/EO/OE/OO rectangles from the resulting y/x interval products.
9. Build internal routes from the closed cones, but retain the exact cones for
   final LL/LH/HL/HH source ownership and terminal writes.
10. Reject any candidate whose resulting L1 allocation exceeds the device
    budget. Exact-domain production planning retains its benchmarked
    conservative 768 KiB search budget; the larger experimental tile-closed
    domain is checked against `MeshDevice::l1_size_per_core()` so the A/B
    experiment is rejected only at the architecture's reported limit.

The implementation keeps intermediate coordinates as checked finite stream
indices. Translation arithmetic is performed with signed offsets before
validation, preventing unsigned dependency wraparound. Closure is clipped to
the finite route stream because values beyond that stream are not part of the
defined lifting state. Enlarged initial intervals are still initialized by the
existing split using symmetric mapping against the original logical `H` and
`W`, never physical zero padding.

### 3.4 Why enlarged work preserves exact final outputs

**Fact.** At every reverse transition, the closed requirement is a superset of
the exact requirement. Predict/update computes each output lane from its base
lane and that lane's fixed source stencil; extra output lanes do not feed a
reduction into exact output lanes. The exact final source rectangles remain
separate and only those rectangles are written to the four result bands.

**Measured.** Exact versus tile-closed produced bit-identical LL/LH/HL/HH
bands in all 60 random-input A/B cases: four schemes by 15 required shapes,
or 240/240 bit-identical bands. This includes `db7 1000x100`, `513x769`, and
`1024x1024`.

### 3.5 One db7 route example

For chunk 0, vertical route 0:

| Domain | Output rectangle `[y0,y1)x[x0,x1)` | Active base rectangle | Active source rectangle |
|---|---|---|---|
| exact | `[0,38)x[1,39)` | `[0,38)x[1,39)` | `[1,39)x[1,39)` |
| tile-closed | `[0,224)x[0,63)` | `[0,224)x[0,63)` | `[0,256)x[0,63)` |

This example shows why repeated closure is expensive for a high-order scheme:
the first vertical route inherits the union of tile-closed requirements from
all later routes.

## 4. Coverage and work expansion

### 4.1 Request-class coverage

| Policy/path | Exact | One-axis shifted | Two-axis shifted | Partial | Empty/zero | Requests |
|---|---:|---:|---:|---:|---:|---:|
| old exact + scalar fallback | 242 | 286 | 0 | 4,798 | 870 | 6,196 |
| exact + bounded assembler | 242 | 286 | 0 | 4,798 | 870 | 6,196 |
| tile-closed + bounded assembler | 1,464 | 1,462 | 0 | 6,426 | 264 | 9,616 |

The bounded assembler intentionally does not alter the logical classification.
It changes the physical implementation:

```text
before: 4,798 partial + 870 empty -> scalar 1024-lane fallback
after:  4,798 partial             -> zero-tile read + bounded face-row copies
        870 empty                 -> zero-tile read only
        scalar generic requests   -> 0
```

**Measured.** Tile closure increased exact/shifted requests, but generated
3,420 additional staging requests and more partial requests in absolute terms.
It therefore missed the coverage objective and was rejected on complete
latency.

### 4.2 Overcompute by phase

| Phase | Exact elements | Tile-closed elements | Ratio |
|---|---:|---:|---:|
| initial EE/EO/OE/OO | 163,744 | 1,150,372 | 7.025430x |
| vertical route outputs | 665,040 | 2,538,000 | 3.816312x |
| horizontal route outputs | 552,552 | 698,280 | 1.263736x |
| all route outputs | 1,217,592 | 3,236,280 | 2.657935x |
| exact final bands | 113,344 | 113,344 | 1.000000x |

Route output tiles increased from 2,172 to 3,312 (`1.525x`).

Split macro-tiles increased from 96 to 321 (`3.344x`). Raw input read traffic
increased from 3,112,960 to 10,485,760 bytes (`3.368x`), and local split output
traffic increased from 654,976 to 4,601,488 bytes (`7.026x`).

## 5. Bounded staging dataflow

### 5.1 One-time classification

`classify_route_tile()` now returns:

```text
exact
one-axis shifted
two-axis shifted
partial
empty
```

It computes containment/intersection once per request. The partial copy path
does not call `contains()` or `tiled_element_offset()` per element.

### 5.2 Persistent zero tile

A new one-page `c_9` circular buffer reserves 4,096 bytes per worker core. The
reader initializes it to FP32 zero. It is refreshed once after each chunk's
split stage; this prevents stale words when one worker processes multiple
chunks while split scratch and CB state are reused.

For partial and empty requests:

```text
reserve destination CB page
-> issue same-core full-tile NoC read from persistent zero
-> issue source0/source1/base reads without an intervening barrier
-> one read barrier for the batch
-> partial only: overlay valid face-row segments
-> publish CB pages
```

The empty path executes no 1024-word clear for each request. The one per-chunk
zero refresh is amortized across the chunk's route requests.

### 5.3 Face-row copy strategy

`assemble_bounded_tile()` computes the requested/stored intersection once,
then derives:

```text
valid source y/x origin
valid destination y/x origin
valid height and width
```

For each valid row it splits only when either source or destination reaches a
16-lane face boundary. Each inner operation is a contiguous direct L1 word
copy. Divisions/modulo are not performed per copied element.

Full contained two-axis-shifted requests use the same bounded assembler.
Existing one-axis-shifted requests retain the specialized dense face-row
assembler. Exact tiles retain the full-tile local NoC path.

The old scalar gather remains selectable through `--route-staging scalar` for
A/B work, but it is no longer reached by optimized production staging.

### 5.4 Compute-only instrumentation

`--microbenchmark compute` is now a compile-time instrumentation variant. It
feeds the persistent zero tile directly into the actual unpack, SFPU stencil,
terminal scale, pack, and output-CB path. It bypasses workspace plane assembly
but preserves route tile counts and writer coordination. The define is absent
from production kernels.

For `db7 1000x100`:

```text
fixed-input staging, max core:        67,059 cycles
compute-pipeline wait, max core:     159,564 cycles
complete compute microbenchmark:      0.804 ms mean
```

The 159,564-cycle number remains an upper bound on compute because it is
observed through writer CB wait, but it is no longer dominated by the bounded
plane assembler. It demonstrates that the former 11.2-million-cycle
`compute/pipeline` value was not pure SFPU arithmetic.

## 6. L1 and NCRISC resource use

### 6.1 L1

| Resource | Exact | Tile-closed |
|---|---:|---:|
| five-plane workspace | 81,920 B | 303,104 B |
| circular buffers, including zero tile | 36,864 B | 36,864 B |
| metadata pages | 384 B | 384 B |
| sync + split scratch | 36,896 B | 36,896 B |
| total | 156,064 B | 377,248 B |
| N150 planner capacity | 1,499,136 B | 1,499,136 B |
| headroom | 1,343,072 B | 1,121,888 B |

The bounded solution adds exactly one 4,096-byte tile relative to the previous
151,968-byte exact plan.

### 6.2 Reader instruction size

Sizes were collected in isolated JIT caches with
`riscv-tt-elf-size`:

| Reader build | NCRISC `.text` |
|---|---:|
| production | 11,464 B |
| transport metrics | 12,672 B |
| direct staging validation | 12,796 B |

All are below the 16 KiB NCRISC instruction region. Shared `noinline` bounded,
finish, and compute-benchmark helpers avoid route/K template multiplication.

## 7. PyWavelets memory-access study

The inspected implementation performs the multidimensional transform
axis-by-axis. Its C axis helper materializes a contiguous temporary input and
output when the selected axis is strided, then runs a dense one-dimensional
filter-bank routine and copies the result back.

Useful concepts transferred:

```text
classify layout once
-> materialize a contiguous working unit
-> run a branch-light dense inner transform
-> keep boundary handling out of the dense loop
```

Those concepts correspond to the TT bounded tile assembler and specialized
SFPU route.

Not transferred:

- convolution filter-bank arithmetic;
- heap allocation per axis signal;
- host-strided arrays;
- PyWavelets' convolution boundary loops.

The TT implementation remains lifting-based, tile-local, and L1/CB driven.

Primary source files inspected:

- https://github.com/PyWavelets/pywt/blob/main/pywt/_multidim.py
- https://github.com/PyWavelets/pywt/blob/main/pywt/_extensions/_dwt.pyx
- https://github.com/PyWavelets/pywt/blob/main/pywt/_extensions/c/wt.template.c
- https://github.com/PyWavelets/pywt/blob/main/pywt/_extensions/c/convolution.template.c

## 8. Numerical behavior and validation

### 8.1 Direct movement validation

**Measured.**

```text
staging corpus:        60/60 cases passed
optimized CB tiles:    164,126
staging mismatches:    0 FP32 words
persisted tiles:       64,268
persistence mismatch: 0 FP32 words
```

Validation mismatch totals are separately reported for exact, one-axis,
two-axis, partial, and empty classes; all were zero.

### 8.2 Split validation

**Measured.**

```text
EE/EO/OE/OO split validation: 264/264 passed
```

This includes thin inputs, 31/32/33 and 63/64/65 boundaries, impulses on
tile-boundary rows/columns, deterministic random input, and `1000x100`.

### 8.3 Complete transform validation

**Measured.** 480/480 complete cases passed the unchanged `1e-4` hard
threshold:

```text
schemes:  db1, db7, bior3.9, synthetic-k17
shapes:   15 required edge/interior/multichunk shapes
patterns: zero, constant, row ramp, column ramp, checkerboard,
          corner impulse, tile-boundary impulse, deterministic random
```

Every single-core result matched its multicore result bit-for-bit. No NaN or
Inf was observed. The maximum oracle error was `5.966425e-5` for db7 random
input, within the existing threshold.

The independent route-domain run covered `db1`, `db7`, `bior3.9`, and
`synthetic-k17` at all 15 shapes with deterministic random input:

```text
exact versus tile-closed cases: 60/60 passed
bit-identical result bands:     240/240
```

## 9. Performance results

### 9.1 Stage timing before and after

Instrumented `db7 1000x100`, max active-core totals:

| Phase | Before | After |
|---|---:|---:|
| source/base staging | 11,340,166 cycles | 1,652,236 cycles |
| compute/pipeline wait | 11,204,801 cycles | 1,649,174 cycles |
| persistence | approximately 21,000 cycles | 20,030 cycles |
| route synchronization | approximately 62,000 cycles | 63,014 cycles |
| terminal writes | approximately 163,000 cycles | 162,596 cycles |
| max core | 11,898,118 cycles | 2,179,661 cycles |

The reader and writer overlap, so phase totals must not be added. The
compute-only result in section 5.4 bounds actual compute much lower than the
full-path compute/pipeline wait.

### 9.2 Complete latency before and after

| Case | Before mean | After mean | After minimum | Change |
|---|---:|---:|---:|---:|
| db7 1000x100 | 11.132515 ms | 2.239485 ms | 2.235004 ms | 4.867x faster |
| db7 1024x1024 | 30.019498 ms | 10.747271 ms | 10.742021 ms | 64.2% lower |

The `1024x1024` result is well below the preferred 20 ms target and does not
regress.

### 9.3 Corrected FP32 TT versus PyWavelets

Inputs are the same float32 ramp, `db7`, symmetric mode. Both paths use two
discarded warmups and ten measured repetitions for the primary numbers.

| Backend, 1000x100 | Mean | Minimum |
|---|---:|---:|
| TT prepared device-only | 2.239485 ms | 2.235004 ms |
| TT including H2D + four D2H reads | 2.480762 ms | 2.473373 ms |
| PyWavelets float32 | 1.776859 ms | 1.756555 ms |

**Derived.**

```text
PyWavelets / TT device-only: 0.793x
TT device-only / PyWavelets: 1.260x
PyWavelets / TT end-to-end:  0.716x
TT end-to-end / PyWavelets:  1.396x
```

The skinny-case stretch target was not met after correcting the CPU dtype.

Five-repeat shape matrix:

| Shape | TT mean/min | PyWavelets FP32 mean/min | Py / TT | Cores | Chunk tiles |
|---|---:|---:|---:|---:|---:|
| 64x64 | 2.211/2.209 ms | 0.144/0.136 ms | 0.065x | 4 | 1x1 |
| 128x128 | 2.250/2.248 ms | 0.836/0.773 ms | 0.371x | 9 | 1x1 |
| 256x256 | 2.255/2.250 ms | 1.968/1.926 ms | 0.872x | 25 | 1x1 |
| 512x512 | 3.963/3.959 ms | 6.721/6.702 ms | 1.696x | 45 | 1x2 |
| 1000x100 | 2.239/2.234 ms | 1.943/1.738 ms | 0.868x | 32 | 1x1 |
| 1000x200 | 2.256/2.253 ms | 3.263/3.203 ms | 1.446x | 64 | 1x1 |
| 1024x1024 | 10.748/10.740 ms | 42.253/42.165 ms | 3.931x | 54 | 2x3 |

### 9.4 Width sweep

The warmed FP32 sweep covers `1000x100` through `1000x1000` in steps of 10.
TT was faster in 86 of 91 shapes. It lost at widths 100, 110, 120, 130, and
250. The isolated width-250 loss is a chunk-geometry transition.

| Width beginning | Previous geometry | New geometry | TT change |
|---:|---|---|---:|
| 120 | 32 cores, 1x1 | 48 cores, 1x1 | -0.022 ms |
| 190 | 48 cores, 1x1 | 64 cores, 1x1 | +0.003 ms |
| 250 | 64 cores, 1x1 | 48 cores, 1x2 | +1.696 ms |
| 380 | 48 cores, 1x2 | 64 cores, 1x2 | -0.155 ms |
| 510 | 64 cores, 1x2 | 48 cores, 1x3 | +1.838 ms |
| 570 | 48 cores, 1x3 | 64 cores, 1x3 | -0.130 ms |
| 760 | 64 cores, 1x3 | 64 cores, 1x4 | +1.796 ms |

These steps are planner/chunk work-count changes, not smooth bandwidth scaling.

## 10. Reproduction commands

### Build

```bash
ninja -C build lwt_2d lwt_2d_planner_tests lwt_2d_reference lwt
ninja -C build vertical_stencil_k17_test horizontal_dense_stencil_test
```

### Planner and direct staging

```bash
build/lwt_2d_planner_tests

python3 scripts/validate_lwt_2d_staging.py \
  --result-json /tmp/lwt2d-staging.json \
  --fail-fast
```

### Split

```bash
python3 scripts/validate_lwt_2d_split.py \
  --implementation tiled \
  --result-json /tmp/lwt2d-split.json \
  --fail-fast
```

### Complete four-scheme corpus

```bash
for pattern in zero constant row-ramp column-ramp checkerboard \
               corner-impulse tile-boundary-impulse random; do
  python3 scripts/validate_lwt_2d_device.py \
    --schemes db1,db7,bior3.9,synthetic-k17 \
    --shapes 1x1,2x3,15x17,31x31,32x32,32x33,33x32,33x33,63x65,64x64,65x63,1000x100,1000x200,513x769,1024x1024 \
    --input-type "$pattern" \
    --result-json "/tmp/lwt2d-${pattern}.json" \
    --fail-fast
done
```

### Exact versus tile-closed

```bash
python3 scripts/validate_lwt_2d_device.py \
  --schemes db1,db7,bior3.9,synthetic-k17 \
  --shapes 1x1,2x3,15x17,31x31,32x32,32x33,33x32,33x33,63x65,64x64,65x63,1000x100,1000x200,513x769,1024x1024 \
  --input-type random \
  --route-domain exact \
  --validate-route-domain-ab \
  --fail-fast
```

### Stage metrics and compute-only bound

```bash
source scripts/set_env.sh

build/lwt_2d \
  --benchmark --binary-input --cores 64 \
  --repeats 5 --warmup-runs 1 \
  --transport-metrics \
  db7 1000 100 /dev/zero

build/lwt_2d \
  --microbenchmark compute --binary-input --cores 64 \
  --repeats 5 --warmup-runs 1 \
  db7 1000 100 /dev/zero
```

### Corrected FP32 matrix

```bash
python3 compare_timings.py \
  --backend both \
  --shapes 64x64 128x128 256x256 512x512 1000x100 1000x200 1024x1024 \
  --wavelets db7 \
  --tt-repeats 5 \
  --tt-warmup-runs 1 \
  --pywt-repeats 5 \
  --pywt-warmup-runs 1 \
  --csv /tmp/lwt2d-fp32-matrix.csv \
  --overwrite
```

### Width sweep

```bash
mapfile -t shapes < <(
  for width in $(seq 100 10 1000); do
    printf '1000x%s\n' "$width"
  done
)

python3 compare_timings.py \
  --backend both \
  --shapes "${shapes[@]}" \
  --wavelets db7 \
  --tt-repeats 3 \
  --tt-warmup-runs 1 \
  --pywt-repeats 3 \
  --pywt-warmup-runs 1 \
  --csv /tmp/lwt2d-fp32-width-sweep.csv \
  --overwrite
```

## 11. Remaining bottleneck, risks, and next step

**Measured.** Exact bounded route staging remains the dominant critical-path
phase at 1.652 million cycles. The fixed-input compute-pipeline upper bound is
0.160 million cycles, split is 0.424 million cycles, terminal writes are 0.163
million cycles, and route synchronization is 0.063 million cycles.

**Hypothesis.** The next useful optimization is adjacent-output staging reuse:
form the union of source physical tiles for two neighboring outputs, load each
unique tile once, and produce two CB-page sets. It should be tested separately
for vertical/horizontal routes and skinny/square shapes. It was not added in
this change because it alters CB batching and route ownership substantially;
the bounded assembler delivered the primary 4.867x result without that risk.

Planner discontinuities at widths 250, 510, and 760 are also material. A
continuous cost model or per-shape autotuning between neighboring chunk
geometries can avoid approximately 1.7-1.8 ms steps.

The tile-closed option remains useful as a correctness/geometry experiment,
but should not be selected for db7 skinny matrices unless a future planner
predicts a net latency reduction. Its present repeated closure amplifies early
vertical work too aggressively.

The main remaining risk is benchmark variance on small CPU transforms. Primary
claims therefore use warmed mean-to-mean and minimum-to-minimum values, and do
not compare TT minimum against PyWavelets mean.
