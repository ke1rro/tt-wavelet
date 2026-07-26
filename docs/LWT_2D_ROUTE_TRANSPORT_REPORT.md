# 2D LWT route-transport optimization report

Measurement date: 2026-07-26 UTC  
Device: Wormhole N150, one chip  
Timed region: prepared `MeshWorkload` enqueue plus device completion; input
upload, output readback, program construction, and compilation are excluded.

Claims are labelled as **Fact**, **Derived**, **Measured**, or **Hypothesis**.

## 1. Executive verdict

**Measured — success.** The optimized complete db7 transform is 27.0-27.8%
faster on the plateau-sized square cases and 27.1-27.3% faster on the two
skinny cases. The final 1000x100 production check is:

| Statistic | Time |
|---|---:|
| mean | 11.132515 ms |
| minimum | 11.128537 ms |
| median | 11.130626 ms |
| p10 | 11.128873 ms |
| p90 | 11.136459 ms |
| standard deviation | 0.004735 ms |

The original measured value was 15.109254 ms, close to the supplied
approximately 15.101 ms baseline. This is a 26.3% reduction for the final
nine-repeat check. The broad five-repeat matrix measured 15.291 ms to
11.120 ms, a 27.3% reduction.

**Measured — all acceptance targets passed.**

- Eligible exact staging used 0.005892 times the scalar staging cycles.
- Full-tile persistence used 0.012836 times the scalar persistence cycles.
- Full-tile terminal writes used 0.147 times the fragmented cycles on
  1024x1024.
- Every required plateau case improved by more than 20%.
- db7 1024x1024 improved by 69.5%, rather than regressing.
- Split validation remains 264/264.
- The direct transport validator checked 44,010 optimized CB pages and 64,268
  persisted tiles with zero mismatched FP32 words.

**Fact — execution architecture.** One prepared TT program contains one reader
kernel, one SFPU compute kernel, and one writer kernel. The reader, compute,
and writer loop over every vertical and horizontal route on device. There is
not a host program launch per route and there is no DRAM round trip for
vertical intermediate bands. A timed repeat performs one workload enqueue and
one finish.

**Fact — split versus compute.** The unchanged tiled `split2d` is a dataflow
reader implementation: it reads complete raw tiles through NoC into local
scratch and deinterleaves parity values into the four local planes. It is not
an SFPU split. Predict/update and scale arithmetic execute in the SFPU compute
kernel. The measured split maximum was 492,181 cycles, approximately 0.492 ms
at 1 GHz, so route execution remains the optimization target.

## 2. Scope, assumptions, and controls

**Fact.** The production defaults are now:

```text
split:              tiled
route staging:      optimized
route persistence: full-tile
terminal writes:    tiled
scale:              fused
route config:       preloaded
exact transfer:     local-noc
planner:            latency
```

**Fact.** Every old implementation remains selectable:

```text
--route-staging scalar|optimized
--route-persistence scalar|full-tile
--terminal-writes fragmented|tiled
--scale-policy explicit|fused
--route-config per-route|preloaded
--exact-transfer local-noc|l1-copy
--planner max-cores|latency
```

**Fact.** Timestamp instrumentation is compiled only when transport metrics
are requested. It records route axis, step, K, output tiles, configuration,
staging, compute/pipeline, persistence, synchronization, terminal writes, and
whole-reader/writer lifetime. Host aggregation reports maximum and mean active
core cycles, route tiles, and maximum route tiles per core.

**Fact.** The executable exposes these phase views:

```text
--microbenchmark empty|split|staging|compute|persistence|terminal|full
```

`empty` is a no-op prepared TT program. The other named modes run the complete
prepared program with only the requested phase emphasized in the printed
instrumentation. They are phase views, not separate stage-only numerical
programs. The empty five-run result was 0.098766 ms mean and 0.047569 ms
minimum; host/device launch variance is therefore significant at sub-0.1 ms.

## 3. Mathematical and precision contract

**Fact.** The implementation retains the vertical-first separable lifting
order:

```text
EE/OE -> Le/He
EO/OO -> Lo/Ho
Le/Lo -> LL/LH
He/Ho -> HL/HH
```

**Fact.** Route arithmetic remains FP32, with `fp32_dest_acc_en`,
`UnpackToDestFp32`, scheme-order tap accumulation, and the existing horizontal
and vertical stencil contracts. The generated scheme supplies
`Step::coeff_bits` at compile time; static coefficients are no longer runtime
arguments.

**Fact.** Terminal scale fusion performs the scale in DEST immediately after
the last predict/update that produces the stream. It does not fold scale into
stencil coefficients or reorder the lifting operations.

| Scheme | Route records | Explicit executable | Fused executable | Scale routes removed |
|---|---:|---:|---:|---:|
| db1 | 20 | 16 | 12 | 4 |
| db7 | 44 | 40 | 36 | 4 |
| bior3.9 | 24 | 20 | 16 | 4 |
| synthetic-k17 | 12 | 12 | 8 | 4 |

**Measured.** The required random corpus stayed below the existing 1e-4 hard
threshold. Maximum errors were 0 for db1, `5.96642494e-05` for db7,
`1.43051147e-06` for bior3.9, and `7.15255737e-07` for synthetic-k17. No NaN or
Inf occurred.

## 4. Dependency ownership and route staging

**Fact.** Each chunk owns an exact dependency cone, all persistent planes are
same-core L1 shards, and no worker reads another worker's intermediate plane.

The optimized reader classifies each requested source or base page:

- Exact full tile: one 4096-byte same-core local NoC read into the CB page.
- One-axis shifted: face-row contiguous L1 segments are copied from adjacent
  persistent tiles. There is no per-element tiled-index calculation.
- Generic fallback: the original scalar gather remains for partial, zero-fill,
  and irregular cases.

Exact source0, source1, and base reads are issued together and completed by one
barrier. Generic correctness is unchanged.

### K=1 through K=17 source packing

**Fact.** For 32 outputs and K taps, the useful logical source interval has
`32 + K - 1` values. Vertical packing starts at the route source origin.
Horizontal packing preserves the SFPU helper's leading `17-K` lanes:

| K | Useful logical values | Horizontal leading lanes | Horizontal CB0 request | Vertical CB0 request | Fast physical tiles per CB |
|---:|---:|---:|---|---|---|
| 1 | 32 | 16 | `s + o - 16` | `s + o` | exact 1, shifted 2 |
| 2 | 33 | 15 | `s + o - 15` | `s + o` | exact 1, shifted 2 |
| 3 | 34 | 14 | `s + o - 14` | `s + o` | exact 1, shifted 2 |
| 4 | 35 | 13 | `s + o - 13` | `s + o` | exact 1, shifted 2 |
| 5 | 36 | 12 | `s + o - 12` | `s + o` | exact 1, shifted 2 |
| 6 | 37 | 11 | `s + o - 11` | `s + o` | exact 1, shifted 2 |
| 7 | 38 | 10 | `s + o - 10` | `s + o` | exact 1, shifted 2 |
| 8 | 39 | 9 | `s + o - 9` | `s + o` | exact 1, shifted 2 |
| 9 | 40 | 8 | `s + o - 8` | `s + o` | exact 1, shifted 2 |
| 10 | 41 | 7 | `s + o - 7` | `s + o` | exact 1, shifted 2 |
| 11 | 42 | 6 | `s + o - 6` | `s + o` | exact 1, shifted 2 |
| 12 | 43 | 5 | `s + o - 5` | `s + o` | exact 1, shifted 2 |
| 13 | 44 | 4 | `s + o - 4` | `s + o` | exact 1, shifted 2 |
| 14 | 45 | 3 | `s + o - 3` | `s + o` | exact 1, shifted 2 |
| 15 | 46 | 2 | `s + o - 2` | `s + o` | exact 1, shifted 2 |
| 16 | 47 | 1 | `s + o - 1` | `s + o` | exact 1, shifted 2 |
| 17 | 48 | 0 | `s + o` | `s + o` | exact 1, shifted 2 |

Here `s` is the route source begin and `o` is the output-tile displacement
translated into source coordinates. CB1 starts 32 logical positions after the
listed CB0 request. Two-axis or partial geometry remains generic and may touch
up to four physical tiles.

## 5. Alignment histograms

**Measured.** These are staging records only; terminal output records are
excluded. `E` is exact, `S` one-axis shifted, `P` partial logical edge, and `Z`
out-of-stored-range zero fill. Two-axis shifted was zero for all cases because
separable routes shift only their active axis.

| Scheme | Shape | E | S | P | Z | Fast E+S | Fallback P+Z |
|---|---|---:|---:|---:|---:|---:|---:|
| db1 | 64x64 | 7 | 4 | 18 | 6 | 31.4% | 68.6% |
| db1 | 256x256 | 112 | 64 | 288 | 96 | 31.4% | 68.6% |
| db1 | 1000x100 | 105 | 60 | 683 | 160 | 16.4% | 83.6% |
| db1 | 1000x200 | 315 | 180 | 1217 | 416 | 23.3% | 76.7% |
| db1 | 1024x1024 | 2368 | 2944 | 1152 | 1152 | 69.7% | 30.3% |
| db7 | 64x64 | 16 | 19 | 499 | 146 | 5.1% | 94.9% |
| db7 | 256x256 | 256 | 304 | 3988 | 968 | 10.2% | 89.8% |
| db7 | 1000x100 | 242 | 286 | 4798 | 870 | 8.5% | 91.5% |
| db7 | 1000x200 | 726 | 858 | 10310 | 2414 | 11.1% | 88.9% |
| db7 | 1024x1024 | 7152 | 11312 | 19974 | 4108 | 43.4% | 56.6% |
| bior3.9 | 64x64 | 6 | 5 | 241 | 45 | 3.7% | 96.3% |
| bior3.9 | 256x256 | 96 | 80 | 1996 | 324 | 7.1% | 92.9% |
| bior3.9 | 1000x100 | 188 | 141 | 2983 | 448 | 8.8% | 91.2% |
| bior3.9 | 1000x200 | 282 | 234 | 5436 | 784 | 7.7% | 92.3% |
| bior3.9 | 1024x1024 | 2848 | 3760 | 10648 | 1298 | 35.6% | 64.4% |
| synthetic-k17 | 64x64 | 0 | 0 | 143 | 13 | 0.0% | 100.0% |
| synthetic-k17 | 256x256 | 0 | 0 | 1184 | 148 | 0.0% | 100.0% |
| synthetic-k17 | 1000x100 | 0 | 0 | 2272 | 368 | 0.0% | 100.0% |
| synthetic-k17 | 1000x200 | 0 | 0 | 2615 | 285 | 0.0% | 100.0% |
| synthetic-k17 | 1024x1024 | 1032 | 328 | 6930 | 610 | 15.3% | 84.7% |

**Derived.** Fast-path coverage is shape- and scheme-dependent. The complete
speedup cannot be attributed only to exact tiles: full-tile persistence,
terminal batching, scale fusion, configuration preload, and better large-shape
chunking are also material.

## 6. Writer, configuration, and planner schedule

**Fact — route persistence.** The writer replaces 1,024 scalar word stores with
one 4096-byte same-core local NoC write. It batches two output tiles when the
two-page output CB is physically contiguous, waits once, and releases pages
only after the transfer barrier.

**Fact — terminal writes.** Complete owned terminal tiles use 4096-byte DRAM
writes, batched up to 16 before a barrier. Partial/shared edge tiles retain the
fragmented face-row writer, so workers never overwrite another chunk's valid
values.

**Fact — configuration.** Reader and writer preload all 96-byte route pages
into separate halves of the existing 36 KiB split scratch after that scratch
is no longer live. For db7 this changes 88 per-route blocking read barriers per
chunk to two preload barriers and removes the repeated 24-word stack copy.

**Measured and important qualification.** The number of descriptor pages read
from DRAM is not reduced: reader and writer still each read 44 pages for db7.
Therefore “configuration loads removed” is zero pages; “blocking barriers
removed” is 86 per chunk. A shared single preload was tested but deadlocked
under the current reader/writer synchronization and was rejected. Safe
duplicate preload improved db7 1000x100 from 11.621954 ms to 11.129060 ms in
the isolated A/B.

**Fact — planner.** The latency planner estimates fixed core/route cost,
alignment-specific staging, axis/K compute, persistence, terminal writes,
configuration, halo duplication, and chunks per core. It reduces occupancy
only when the predicted win exceeds a conservative 10%.

**Measured.** On db7 1024x1024, max-cores selected 64 active cores, 68 chunks,
and 1x5 band-tile chunks for 50.692191 ms after other optimizations. The
latency planner selected 54 active cores, 54 chunks, and 2x3 chunks for
30.019498 ms. On 1000x100 both policies settle at 32 1x1 chunks.

The early 1000x100 core sweep measured:

| Requested cores | Mean |
|---:|---:|
| 1 | 157.674 ms |
| 2 | 80.292 ms |
| 4 | 41.498 ms |
| 8 | 22.078 ms |
| 16 | 12.380 ms |
| 32 | 11.740 ms |
| 48 | 11.738 ms |
| 64 | 11.739 ms |

Only 32 chunks exist for that candidate, so requests above 32 do not create
additional active work.

## 7. Performance evidence

### Stage-level db7 1000x100

**Measured.** Timestamp instrumentation adds overhead, so use this table for
relative phase attribution, not as the production wall-clock value. Phases
overlap and are not additive.

| Maximum active-core phase | Scalar/explicit/per-route | Optimized/fused/preloaded | Change |
|---|---:|---:|---:|
| complete core | 14,879,216 cycles | 11,898,640 cycles | -20.0% |
| configuration | 64,311 | 4,556 | -92.9% |
| source/base staging | 13,503,300 | 11,340,263 | -16.0% |
| compute/pipeline | 12,969,242 | 11,205,044 | -13.6% |
| output persistence | 1,660,707 | 20,880 | -98.7% |
| route synchronization wait | 668,959 | 61,986 | -90.7% |
| terminal writes | 177,377 | 162,533 | -8.4% |

The instrumented complete invocation moved from 15.143846 ms to 12.131214 ms.
The non-instrumented production result is lower.

### Same-core transfer mechanisms

**Measured.** Exact tile staging on db7 1024x1024:

| Mechanism | Complete transform |
|---|---:|
| local NoC read | 30.068046 ms |
| scalar/unrolled L1 block copy | 33.597607 ms |

Local NoC is 10.5% lower and is the selected default. CB aliasing, globally
allocated CBs, and direct pack-to-plane were not selected for the first
production version because they would change ownership and synchronization
contracts.

### Complete transform matrix

**Measured.** Values are five-run means in milliseconds. Baseline means
scalar staging, scalar persistence, fragmented terminal writes, explicit
scale, per-route configuration, and max-core planning.

| Scheme | Shape | Baseline | Optimized | Improvement |
|---|---|---:|---:|---:|
| db1 | 64x64 | 3.242 | 1.680 | 48.2% |
| db1 | 128x128 | 3.198 | 1.723 | 46.1% |
| db1 | 256x256 | 3.235 | 1.724 | 46.7% |
| db1 | 512x512 | 3.250 | 1.746 | 46.3% |
| db1 | 1000x100 | 3.206 | 1.953 | 39.1% |
| db1 | 1000x200 | 3.236 | 1.987 | 38.6% |
| db1 | 1024x1024 | 13.506 | 4.227 | 68.7% |
| db7 | 64x64 | 15.399 | 11.122 | 27.8% |
| db7 | 128x128 | 15.275 | 11.155 | 27.0% |
| db7 | 256x256 | 15.324 | 11.161 | 27.2% |
| db7 | 512x512 | 30.609 | 14.737 | 51.9% |
| db7 | 1000x100 | 15.291 | 11.120 | 27.3% |
| db7 | 1000x200 | 15.309 | 11.156 | 27.1% |
| db7 | 1024x1024 | 98.314 | 30.019 | 69.5% |
| bior3.9 | 64x64 | 7.733 | 5.768 | 25.4% |
| bior3.9 | 128x128 | 7.725 | 5.774 | 25.3% |
| bior3.9 | 256x256 | 7.765 | 5.801 | 25.3% |
| bior3.9 | 512x512 | 15.496 | 8.502 | 45.1% |
| bior3.9 | 1000x100 | 7.747 | 5.799 | 25.1% |
| bior3.9 | 1000x200 | 7.759 | 5.858 | 24.5% |
| bior3.9 | 1024x1024 | 50.830 | 16.859 | 66.8% |
| synthetic-k17 | 64x64 | 5.095 | 3.787 | 25.7% |
| synthetic-k17 | 128x128 | 5.091 | 3.784 | 25.7% |
| synthetic-k17 | 256x256 | 5.113 | 3.786 | 26.0% |
| synthetic-k17 | 512x512 | 10.143 | 6.028 | 40.6% |
| synthetic-k17 | 1000x100 | 5.087 | 3.770 | 25.9% |
| synthetic-k17 | 1000x200 | 10.059 | 6.286 | 37.5% |
| synthetic-k17 | 1024x1024 | 32.372 | 12.039 | 62.8% |

Complete db7 optimized statistics:

| Shape | Mean | Min | Median | P10 | P90 | Stddev | Cores | Chunks | Geometry |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64x64 | 11.122262 | 11.111298 | 11.112228 | 11.111346 | 11.138659 | 0.013090 | 4 | 4 | 1x1 |
| 128x128 | 11.155493 | 11.153488 | 11.154647 | 11.153672 | 11.158259 | 0.002564 | 9 | 9 | 1x1 |
| 256x256 | 11.160865 | 11.154297 | 11.156078 | 11.154569 | 11.170584 | 0.007868 | 25 | 25 | 1x1 |
| 512x512 | 14.736965 | 14.732687 | 14.735477 | 14.733507 | 14.741577 | 0.003907 | 45 | 45 | 1x2 |
| 1000x100 | 11.120365 | 11.116477 | 11.120997 | 11.116681 | 11.123918 | 0.003215 | 32 | 32 | 1x1 |
| 1000x200 | 11.155771 | 11.152077 | 11.154048 | 11.152673 | 11.160361 | 0.004021 | 64 | 64 | 1x1 |
| 1024x1024 | 30.019498 | 30.003073 | 30.004953 | 30.003185 | 30.044704 | 0.019938 | 54 | 54 | 2x3 |

db7 uses 44 route records, 36 executable routes after fusion. Route tile and
fast/fallback counts are available in the transport metrics CSV; the
1000x100 instrumented run had 2,172 route output tiles and a maximum of 90
route tiles on one core.

## 8. Memory and code-size status

**Fact.** Four FP32 tile CBs—source0, source1, base, output—are double-buffered:
eight 4096-byte pages, or 32,768 bytes. Metadata is 384 bytes. Synchronization
plus 3x3 raw split scratch is 36,896 bytes. Configuration preload reuses that
scratch and adds no L1 allocation. Optional metric buffers are in DRAM and are
absent from production variants.

| Shape/scheme | L1 total | L1 capacity | Headroom |
|---|---:|---:|---:|
| db7 1000x100 | 151,968 B | 1,499,136 B | 1,347,168 B |
| db7 1024x1024 | 315,808 B | 1,499,136 B | 1,183,328 B |

**Measured.** Successful JIT ELF sizes:

| Variant | RISC | Text | BSS | Status |
|---|---|---:|---:|---|
| final production reader | NCRISC | 12,948 B | 8 B | fits |
| direct-validation reader | NCRISC | 15,512 B | 24 B | fits 16 KiB region |
| final production writer | BRISC | 3,704 B | 0 B | fits |
| direct-validation writer | BRISC | 4,252 B | 0 B | fits |
| worst observed successful compute | TRISC1 | 35,228 B | 0 B | JIT/load passed |

The first combined timestamp-plus-direct-validation reader overflowed NCRISC
by 328 bytes. The final direct checker therefore uses a two-counter lightweight
metric page and does not compile timestamp instrumentation into that variant.

## 9. Edge cases and validation evidence

**Measured.**

- 60/60 direct route-transport cases passed.
- 44,010 optimized source/base CB pages matched the scalar gather bit-for-bit
  before SFPU consumption.
- 64,268 full-tile persistence operations matched the output CB bit-for-bit
  after the local NoC barrier and before CB release.
- 60/60 random complete-transform cases passed on the exact required
  scheme/shape corpus.
- Fragmented and tiled terminal bands were bit-identical in all 60 cases.
- Single-core and multi-core bands were bit-identical in all 60 cases.
- Seven deterministic pattern runs each passed 60/60 cases: zero, constant,
  row ramp, column ramp, checkerboard, corner impulse, and tile-boundary
  impulse.
- Combined with random, the complete-transform required validation total was
  480/480 cases. The earlier extended random corpus added another 92/92 cases.
- Tiled split validation passed 264/264.
- A 1D db7 length-100000 timing smoke passed.
- `ctest -N` reports no registered CTest cases in this build.

**Fact.** Generic staging remains the correctness path for partial and
zero-filled dependency tiles. Terminal edge writes remain fragmented.
Symmetric boundary mapping still occurs in original image coordinates during
the split.

## 10. Reproduction commands and files

### Build

```bash
ninja -C build lwt lwt_2d lwt_2d_reference
python3 -m py_compile \
  compare_timings.py \
  scripts/validate_lwt_2d_device.py \
  scripts/validate_lwt_2d_staging.py
```

### Direct staging and persistence validation

```bash
python3 scripts/validate_lwt_2d_staging.py \
  --result-json /tmp/lwt2d_required_transport_validation_final.json \
  --fail-fast
```

Result:

```text
60/60 passed
44010 optimized CB tiles, 0 staging mismatches
64268 persisted tiles, 0 persistence mismatches
```

### Complete transform and terminal A/B

```bash
python3 scripts/validate_lwt_2d_device.py \
  --schemes db1,db7,bior3.9,synthetic-k17 \
  --shapes 1x1,2x3,15x17,31x31,32x32,32x33,33x32,33x33,63x65,64x64,65x63,1000x100,1000x200,513x769,1024x1024 \
  --input-type random \
  --validate-terminal-ab \
  --result-json /tmp/lwt2d_required_terminal_ab_validation.json \
  --fail-fast
```

Result: 60/60 passed, including oracle threshold, finite-value,
single/multi-core identity, and fragmented/tiled identity.

The seven deterministic pattern runs used the same command without
`--validate-terminal-ab`, replacing `random` with each of:

```text
zero
constant
row-ramp
column-ramp
checkerboard
corner-impulse
tile-boundary-impulse
```

### Split and 1D regression

```bash
python3 scripts/validate_lwt_2d_split.py \
  --implementation tiled \
  --result-json /tmp/lwt2d_split_tiled_validation.json \
  --fail-fast

python3 compare_timings.py \
  --transform lwt \
  --backend tt-wavelet \
  --wavelets db7 \
  --length-start 100000 \
  --length-stop 100000 \
  --length-step 10000 \
  --tt-repeats 1 \
  --tt-warmup-runs 1 \
  --csv /tmp/lwt1d_regression_smoke.csv \
  --overwrite
```

### Baseline and optimized benchmark matrices

```bash
python3 compare_timings.py \
  --transform lwt \
  --backend tt-wavelet \
  --wavelets db1 db7 bior3.9 synthetic-k17 \
  --shapes 64x64 128x128 256x256 512x512 1000x100 1000x200 1024x1024 \
  --tt-repeats 5 \
  --tt-warmup-runs 1 \
  --tt-route-staging scalar \
  --tt-route-persistence scalar \
  --tt-terminal-writes fragmented \
  --tt-scale-policy explicit \
  --tt-route-config per-route \
  --tt-planner max-cores \
  --csv /tmp/lwt2d_all_schemes_baseline.csv \
  --overwrite

python3 compare_timings.py \
  --transform lwt \
  --backend tt-wavelet \
  --wavelets db1 db7 bior3.9 synthetic-k17 \
  --shapes 64x64 128x128 256x256 512x512 1000x100 1000x200 1024x1024 \
  --tt-repeats 5 \
  --tt-warmup-runs 1 \
  --csv /tmp/lwt2d_all_schemes_optimized.csv \
  --overwrite
```

### Final production check

```bash
python3 compare_timings.py \
  --transform lwt \
  --backend tt-wavelet \
  --wavelets db7 \
  --shapes 1000x100 \
  --tt-repeats 9 \
  --tt-warmup-runs 3 \
  --tt-cores 64 \
  --csv /tmp/lwt2d_final_production_postformat2.csv \
  --overwrite
```

### Telemetry and alignment

```bash
python3 compare_timings.py \
  --transform lwt \
  --backend tt-wavelet \
  --wavelets db7 \
  --shapes 1000x100 \
  --tt-repeats 1 \
  --tt-warmup-runs 1 \
  --tt-transport-metrics \
  --csv /tmp/lwt2d_final_metrics_check.csv \
  --overwrite

python3 compare_timings.py \
  --transform lwt \
  --backend tt-wavelet \
  --wavelets db1 db7 bior3.9 synthetic-k17 \
  --shapes 64x64 256x256 1000x100 1000x200 1024x1024 \
  --tt-repeats 1 \
  --tt-warmup-runs 1 \
  --tt-alignment-csv-dir /tmp/lwt2d_alignment \
  --csv /tmp/lwt2d_alignment_runs.csv \
  --overwrite
```

### Files added or modified for this task

```text
compare_timings.py
scripts/validate_lwt_2d_device.py
scripts/validate_lwt_2d_staging.py
tt-wavelet/kernels/compute/lwt_2d_compute.cpp
tt-wavelet/kernels/dataflow/lwt_2d_empty.cpp
tt-wavelet/kernels/dataflow/lwt_2d_reader.cpp
tt-wavelet/kernels/dataflow/lwt_2d_writer.cpp
tt-wavelet/main_2d.cpp
tt-wavelet/tt_wavelet/include/device_protocol/lwt_2d_config.hpp
tt-wavelet/tt_wavelet/include/lifting/device_2d.hpp
tt-wavelet/tt_wavelet/include/lifting/plan_2d.hpp
tt-wavelet/tt_wavelet/src/lifting/device_2d.cpp
docs/LWT_2D_ROUTE_TRANSPORT_REPORT.md
```

No unrelated 1D LWT or ILWT implementation was changed by this task.

## 11. Recommended implementation and remaining bottleneck

**Fact.** The checked-in production choice is:

```text
full-tile local NoC for exact staging
face-row L1 segments for one-axis shifts
scalar generic gather for partial/zero-fill geometry
SFPU predict/update and fused terminal scale
two-tile batched local NoC persistence
16-tile batched DRAM terminal writes
duplicate per-RISC descriptor preload in reused split scratch
latency-oriented 2D chunk selection
```

**Measured.** The remaining dominant critical path is reader staging overlapped
with SFPU compute/pipeline: 11.340 million and 11.205 million maximum-core
cycles respectively on instrumented db7 1000x100. Persistence and route wait
are no longer dominant.

**Derived.** The main reason is coverage, not the exact-tile mechanism:
91.5% of db7 1000x100 staging records remain partial or zero-fill fallback.

**Recommended next step.** Add bounded partial/zero-fill assembly that copies
complete faces and face rows, then clears only the uncovered lanes. Specialize
it by axis and route shift modulo 32, but retain the scalar gather as an oracle.
Re-profile after that change. Tile-batch route pipelining is a second step only
if staging and SFPU remain balanced after fallback reduction.

**Hypothesis.** Converting the dominant partial fallback to face/row bulk
assembly is more likely to move the 11 ms plateau toward the requested
6-10 ms range than further split work or more core occupancy.
