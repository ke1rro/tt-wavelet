# TTNN Wavelet Operations

This directory implements native, single-level FP32 lifting wavelet transforms:

- `ttnn.dwt` and `ttnn.idwt` for one-dimensional signals;
- `ttnn.dwt_2d` and `ttnn.idwt_2d` for two-dimensional signals.

The TTNN port preserves the standalone `tt-wavelet` lifting arithmetic,
dependency-cone planners, chunk scheduling, boundary extension, SFPI compute
primitives, and fused reader/compute/writer execution. The public namespace is
`ttnn`; implementation types live in `ttnn::operations::wavelet`, and registered
device primitives live in `ttnn::prim`.

## Python API

```python
approximation, detail = ttnn.dwt(
    input,
    "bior1.3",
    boundary_mode="symmetric",
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_tensors=None,
)

reconstructed = ttnn.idwt(
    approximation,
    detail,
    "bior1.3",
    original_length,
    boundary_mode="symmetric",
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_tensor=None,
)

ll, lh, hl, hh = ttnn.dwt_2d(
    input_2d,
    "bior1.3",
    boundary_mode="symmetric",
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_tensors=None,
)

reconstructed_2d = ttnn.idwt_2d(
    ll,
    lh,
    hl,
    hh,
    "bior1.3",
    (original_height, original_width),
    boundary_mode="symmetric",
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_tensor=None,
)
```

The 2D band order `(LL, LH, HL, HH)` is named by vertical and horizontal
result. It maps to PyWavelets `(cA, cV, cH, cD)`; PyWavelets' nested `dwt2`
return is therefore `ll, (hl, lh, hh)` when variable names follow its
documentation.

`original_length` and `output_shape` are required because coefficient shapes do
not uniquely encode the original odd/even dimensions. Preallocated outputs must
have the exact inferred tensor specification, be on the same device, and not
alias inputs or sibling outputs.

## Supported contract

| Operation | Rank and tensor layout | Memory | Output |
| --- | --- | --- | --- |
| `dwt` | `[W]` or `[B,1,1,W]`, row-major, FLOAT32 | interleaved DRAM or L1 input, one physical device | two rank-preserving row-major interleaved DRAM coefficient tensors |
| `idwt` | two equal `[Wc]` or `[B,1,1,Wc]` row-major FLOAT32 tensors | independently interleaved DRAM or L1 inputs, one physical device | one rank-preserving row-major interleaved DRAM signal tensor |
| `dwt_2d` | `[H,W]` or `[B,1,H,W]`, standard 32x32 tile layout, FLOAT32 | interleaved DRAM or L1 input, one physical device | four rank-preserving tile-layout interleaved DRAM band tensors |
| `idwt_2d` | four equal `[Hc,Wc]` or `[B,1,Hc,Wc]` standard-tile FLOAT32 tensors | independently interleaved DRAM or L1 inputs, one physical device | one rank-preserving tile-layout interleaved DRAM image tensor |

All 106 discrete PyWavelets scheme names are registered. The generated registry
and headers under `generated/schemes/` are the source of truth for scheme IDs,
tap counts, lifting steps, coefficients, and forward/inverse compute types.
Device kernels are compiled on first use for the selected scheme; calling one
scheme does not eagerly compile the other 105.

All eight standalone boundary modes are supported: `zero`, `constant`,
`symmetric`, `reflect`, `periodic`, `smooth`, `antisymmetric`, and
`antireflect`. Reflect and antireflect require every transformed dimension to
contain more than one logical sample.

Rank-4 inputs require `C == 1`; 1D rank-4 inputs additionally require `H == 1`.
The operation intentionally rejects host tensors, multi-device tensors,
sharded memory layouts (including sharded L1), BFLOAT16, arbitrary channels,
and nonstandard 2D tiles. Inputs larger than one core's active L1 working set are
chunked by the existing dependency-cone planner. The full input may reside in
interleaved DRAM or interleaved L1; external L1 storage is still staged through
the reader into dependency-local static-CB workspace and is not used as that
workspace. Inputs that fit neither device DRAM nor aggregate interleaved L1
require application-level streaming and are not an implicit part of this
contract.

## Device execution and cache behavior

Each registered primitive builds one fused Metalium workload with the original
reader, SFPI compute, and writer kernels. Tensor addresses and per-core chunk
ranges are runtime arguments. Scheme, boundary specialization, tensor specs
(including DRAM versus L1 placement), workspace geometry, and compute
configuration participate in program specialization through the ordinary TTNN
device-operation cache. Tensor addresses remain runtime arguments, so new
buffers with unchanged specs reuse the cached program.

Per-core intermediate planes are static circular-buffer storage. The 1D
program reserves one contiguous region containing three stick-addressed slots;
the 2D program reserves one region containing five tile-addressed planes.
Readers and writers still receive the same ready absolute addresses used by the
standalone implementation. This avoids retaining allocator-owned L1
`MeshBuffer`s in every cached workload without adding address calculations to
the device hot path. The workspace reservation follows the latency-sensitive
source, base, output, synchronization, metadata, and scratch CBs, preserving
their low-L1 placement. Static CBs grow from the allocator's unreserved L1 base;
ordinary L1 tensors occupy the opposite frontier. Planning subtracts the live
allocator frontier and fixed CB overhead, and Metalium validates the final CB
region against live allocator occupancy before launch. A configuration that
cannot safely hold the external L1 tensor and required workspace fails instead
of overlapping them.

The generated compute headers remain compile-time specializations so tap loops,
step structure, coefficients, and SFPI unroll factors are visible to the device
compiler. Boundary handling stays in the reader. Forward/inverse and 1D/2D use
separate kernels, as in the standalone implementation.

## Numerical and performance validation

FP32 comparison uses PyWavelets as the semantic reference plus direct
LWT-to-ILWT round trips. The fast suite covers lengths 20, 31, 32, and 33,
external inverse coefficients shorter than one 32-value stick, odd and even 2D
shapes, every boundary mode, preallocation, cache behavior, and validation
failures. Focused placement tests compare DRAM and interleaved-L1 results
exactly for odd, boundary-touching, multi-chunk 1D and 2D cases; they also cover
all-L1 and mixed-placement inverse inputs, cache separation, address override,
and sharded-input rejection. Slow tests JIT and execute forward and inverse
transforms for all 106 schemes in both dimensions.

High-order schemes such as `db38`, `coif17`, and `dmey` can amplify FP32 error
substantially. This is inherited from the standalone arithmetic and is not
masked with relaxed implementation-specific math. Scheme-wide smoke tests
therefore require finite results; representative lower-order tests enforce
PyWavelets and round-trip tolerances.

Measured on the available Wormhole N150, warm 5,000,000-element 1D results were:

| Transform | Standalone median | TTNN median | Difference |
| --- | ---: | ---: | ---: |
| `bior3.9` LWT | 13.401 ms | 14.444 ms | +7.8% |
| `db1` LWT | 11.676 ms | 13.171 ms | +12.8% |
| `bior3.9` ILWT | 10.196 ms | 9.813 ms | -3.8% |

The LWT difference is attributable to the first-version exact-rank-1 TTNN
row-major DRAM page shape: it exposes one wide page instead of the standalone
128-byte bank-striped sticks. The TTNN wrapper, static workspace ownership, and
runtime address override do not add device-kernel work. A same-run 1024x1024
`bior1.3` 2D inverse A/B comparison changed from 27.391 ms to 27.398 ms
(+0.024%) after compacting inverse scale specialization. The final static-CB
workspace build measured 27.665 ms with batched warm dispatch (+0.98% against
that earlier run); no reader, writer, or compute hot-path instructions were
added by the workspace ownership change.

Wormhole execution is hardware-tested. Blackhole follows the common public
compute/dataflow APIs and architecture policy but still requires hardware CI
before claiming measured parity.

## Testing

From a configured build tree and environment:

```bash
cmake --build build --target ttnncpp

pytest tests/ttnn/unit_tests/operations/wavelet/test_wavelet.py
pytest --runslow \
    tests/ttnn/unit_tests/operations/wavelet/test_wavelet_all_schemes.py
```

The standalone executables remain the migration oracle and should continue to
be built and compared until TTNN has architecture and performance parity.
