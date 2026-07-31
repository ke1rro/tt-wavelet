# TTNN Integration Design for tt-wavelet

Status: proposed design, not an implementation
Research date: 2026-07-30
Standalone repository revision: `94ee3c0da63d7bf4de3ee9fca27c96b2a92e2648`
Pinned tt-metal revision: `bb9bb157cda03f4780084cf1beed242ddadb9031`

This document uses four evidence labels:

- **Fact** — directly established by the pinned source trees.
- **Inference** — a conclusion derived from facts, but not itself an existing contract.
- **Recommendation** — the proposed integration choice.
- **Open question** — a decision requiring the project owner or an upstream TTNN owner.

## 1. Executive Summary

**Fact.** `tt-wavelet` is a standalone Metalium implementation of four single-level FP32 transforms: 1D LWT, 1D ILWT, 2D LWT, and 2D ILWT. It contains a generated closed registry of 106 schemes, eight boundary modes, explicit dependency-cone planners, architecture policies for Wormhole B0 and Blackhole, and fused reader/compute/writer programs. It is not currently a TTNN operation and has no nanobind Python surface. The build creates four command-line executables (`tt-wavelet/CMakeLists.txt:34-50`).

**Recommendation.** Integrate the transforms as four top-level semantic APIs:

```text
ttnn::lwt
ttnn::ilwt
ttnn::lwt_2d
ttnn::ilwt_2d
```

Each public function should normalize a wavelet name, shapes, boundary mode, and output memory request, then launch exactly one internal device operation in `ttnn::prim`. The primitive must preserve the existing fused planners and device programs; decomposing the transform into generic TTNN slicing, padding, gather, and elementwise operations would add materialization and dispatch boundaries and would discard the present dependency-cone and L1 ownership design.

**Recommendation.** The first upstream-quality support envelope is deliberately narrow:

| Contract | First integration |
|---|---|
| Level count | exactly one |
| Dtype | FP32 |
| Device scope | one physical device |
| Schemes | all 106 validated generated names |
| Boundary modes | all eight existing modes |
| 1D rank/layout | rank 1, row-major |
| 2D rank/layout | rank 2, tile layout |
| Memory | interleaved DRAM input and outputs |
| Forward outputs | fixed tuple of two or four tensors |
| Inverse inputs | separate coefficient tensors |
| Odd-size inverse | explicit original/output shape |
| Architectures | Wormhole B0 and Blackhole |

Sharded tensors, leading batch dimensions, multilevel decomposition, spatial multi-chip execution, and host/out-of-core streaming should be separate measured follow-ups.

**Recommendation.** Keep compile-time scheme specialization and compile it on demand. Use a validated runtime name-to-`SchemeId` registry, bounded cache keys, and the current generated headers. Do not create 106 public operations and do not eagerly compile the Cartesian product of schemes, dimensions, directions, modes, layouts, and architectures.

**Fact.** The current TTNN device-operation framework has explicit cache-miss validation, output-specification, output-allocation, factory selection, and cached workload dispatch (`tt-metal/ttnn/api/ttnn/device_operation.hpp:254-329`, `tt-metal/ttnn/api/ttnn/device_operation.hpp:413-502`). Its descriptor binding implementation is in transition: the public experimental patching header says new code must not depend on the temporary shim (`tt-metal/tt_metal/api/tt-metalium/experimental/program_descriptor_patching.hpp:7-19`), while the mesh adapter also describes a Metal 2.0 `ProgramSpec` path (`tt-metal/ttnn/api/ttnn/mesh_device_operation_adapter.hpp:760-780`).

**Open question.** Before the device-operation PR is written, the TTNN framework owner must select the accepted factory surface for a new complex operation at this exact revision: current `ProgramDescriptor`, `WorkloadDescriptor`, or the emerging `ProgramSpec`. The design below keeps the operation contract independent of that mechanical choice.

**Fact.** Current TTNN warns that Wormhole B0 can produce worse results with HiFi4 plus FP32 destination accumulation (`tt-metal/ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.cpp:43-67`). This is a release blocker for numerical sign-off, not permission to change the lifting arithmetic silently.

**Recommendation.** Preserve shared logical kernels and explicit architecture policy. Wormhole must retain compact 2D boundary code and a `0x4000`-byte NCRISC executable-segment gate. Blackhole must retain the non-compact, tile-native performance path. Fidelity changes require a before/after 106-scheme accuracy study and a separate owner decision.

## 2. Repository and Version Baseline

### 2.1 Exact revisions

**Fact.**

| Item | Revision or state |
|---|---|
| Parent branch | `dev-cleanup` |
| Parent HEAD | `94ee3c0da63d7bf4de3ee9fca27c96b2a92e2648` |
| Old tt-metal pointer | `f87c34a93ee4686c1d7f7adbd4df7ca1804d91ff` |
| Freshly fetched `origin/main` | `bb9bb157cda03f4780084cf1beed242ddadb9031` |
| New tt-metal state | detached HEAD, clean |
| Merge base | `191d0eed35ebfd9120841020e65bbac0ee14def5` |
| Old-only commits | 13 |
| `origin/main`-only commits | 7,440 |

The old submodule identified as `v0.66.0-rc8`; the pinned revision identifies as `v0.76.0-dev20260730-42-gbb9bb157cda`. This is a major API migration, not a routine fast-forward.

**Fact.** The parent worktree already contained user changes to `.gitignore` and `AGENTS.md`. They are outside this design task and must remain untouched. The task changes are limited to the `tt-metal` gitlink and this document.

### 2.2 Baseline risk

**Inference.** A build that succeeded against the old submodule does not establish source compatibility with the new one. The 7,440-main-only divergence includes changes to operation registration, device-operation adapters, descriptor caching, tensor APIs, firmware, SFPI, and architecture policy.

**Recommendation.** Treat `bb9bb15` as the source-of-truth integration baseline, but capture numerical and performance results from the standalone code before moving source into TTNN. The minimum lock should contain:

1. exact generated registry content and generator inputs;
2. representative and exhaustive correctness results, separated by what actually passed;
3. JIT/ELF artifacts for the largest schemes;
4. Wormhole NCRISC size-gate output;
5. Blackhole and Wormhole device-only latency baselines;
6. the current planner telemetry and L1 accounting;
7. known failures.

**Fact.** A previous extra coif17 numerical comparison produced very large errors. The existing workflow treats coif17 as a maximum-step JIT execution check, not an established numerical-correctness case.

**Recommendation.** Record coif17 as a known unresolved baseline limitation. It may pass compilation/execution while still failing correctness; no integration report may convert that JIT pass into a numerical claim.

## 3. Current tt-wavelet Architecture

### 3.1 Public host surface and generated schemes

**Fact.** The public standalone namespace is `ttwv`. Forward and inverse 1D host APIs build typed executable objects, prepare metadata, and enqueue them (`tt-wavelet/tt_wavelet/include/lifting/device.hpp:66-127`, `tt-wavelet/tt_wavelet/include/lifting/device.hpp:129-164`). The 2D equivalents own a `MeshWorkload` and explicitly require original output height and width for inverse reconstruction (`tt-wavelet/tt_wavelet/include/lifting/device_2d.hpp:69-125`, `tt-wavelet/tt_wavelet/include/lifting/device_2d.hpp:127-172`).

**Fact.** The generated registry defines a closed `SchemeId`, metadata, and a 106-entry `kSchemeInfos` array (`tt-wavelet/tt_wavelet/include/schemes/generated/registry.hpp:120-238`, `tt-wavelet/tt_wavelet/include/schemes/generated/registry.hpp:238-345`). It maps public strings to IDs (`tt-wavelet/tt_wavelet/include/schemes/generated/registry.hpp:347-365` and following). Examples of the specialization range are db7 with 11 steps and coif17 with 55 steps (`tt-wavelet/tt_wavelet/include/schemes/generated/registry.hpp:254-262`, `tt-wavelet/tt_wavelet/include/schemes/generated/registry.hpp:271-306`).

**Fact.** Every lifting step records its type, shift, coefficient count, and coefficient bit patterns at compile time (`tt-wavelet/tt_wavelet/include/lifting/static_scheme.hpp:12-44`). Step types are predict, update, scale-even, scale-odd, and swap (`tt-wavelet/tt_wavelet/include/lifting/step.hpp:7-13`).

For a predict/update route, the semantic operation is:

```text
target'[i] = target[i + base_offset]
           + Σ(j=0..K-1) coefficient[j] * source[i + source_offset + j]
```

The generated step shift is incorporated into planned source/base rectangles. Inverse schemes reverse the forward steps and use negated lifting coefficients or reciprocal scales; db7 shows both generated sequences (`tt-wavelet/tt_wavelet/include/schemes/generated/db7.hpp:23-87`, `tt-wavelet/tt_wavelet/include/schemes/generated/db7.hpp:89-164`).

### 3.2 Boundary and shape semantics

**Fact.** The supported boundary modes are zero, constant, symmetric, reflect, periodic, smooth, antisymmetric, and antireflect (`tt-wavelet/tt_wavelet/include/common/boundary.hpp:15-42`). Boundary mapping uses logical dimensions, not physical tile padding.

**Fact.** 2D tensors distinguish logical and storage shapes. Storage is the minimal 32-by-32 tile-aligned expansion (`tt-wavelet/tt_wavelet/include/common/tiling_2d.hpp:17-40`, `tt-wavelet/tt_wavelet/include/common/tiling_2d.hpp:57-76`). Validation requires padding before split and rejects a non-minimal physical expansion (`tt-wavelet/tt_wavelet/include/common/tiling_2d.hpp:78-109`). Host padding writes zeros only into physical padding (`tt-wavelet/tt_wavelet/include/common/tiling_2d.hpp:111-145`).

**Recommendation.** TTNN `TensorSpec` must carry the logical shape while the selected layout supplies its padded shape. Mathematical extension must continue to use logical extents. A padded zero is storage, never a wavelet sample.

### 3.3 Planners and dataflow

```text
host Tensor / Metal buffer
        |
        v
scheme-specific dependency-cone planner
        |
        +--> per-core chunk descriptors
        +--> route descriptors
        +--> band descriptors
        +--> exact L1/CB/resource estimate
        |
        v
reader NCRISC --> source/base CBs --> SFPU lifting compute
       ^                                  |
       |                                  v
       +------ local L1 plane slots <-- writer BRISC
                                           |
                                           v
                                  final DRAM tensor(s)
```

**Fact.** The 1D planner constructs preprocess padding, storage routes, terminal scale routing, and output length (`tt-wavelet/tt_wavelet/include/lifting/plan.hpp:20-95`, `tt-wavelet/tt_wavelet/include/lifting/plan.hpp:224-290`). The inverse planner propagates requirements backward and then emits inverse routes (`tt-wavelet/tt_wavelet/include/lifting/inverse_plan.hpp:108-198`, `tt-wavelet/tt_wavelet/include/lifting/inverse_plan.hpp:200-253`).

**Fact.** The 2D planner represents rectangles, routes, bands, per-core resources, chunks, and execution plans (`tt-wavelet/tt_wavelet/include/lifting/plan_2d.hpp:75-162`). It applies vertical routes to both column-parity pairs, then horizontal routes, and assigns LL/LH/HL/HH (`tt-wavelet/tt_wavelet/include/lifting/plan_2d.hpp:447-489`). It computes exact dependency cones and resource use before searching L1-fitting chunk candidates (`tt-wavelet/tt_wavelet/include/lifting/plan_2d.hpp:512-680`, `tt-wavelet/tt_wavelet/include/lifting/plan_2d.hpp:851-1055`).

The 2D split contract is:

```text
EE[r,c] = input_extended[2r,   2c]
EO[r,c] = input_extended[2r,   2c+1]
OE[r,c] = input_extended[2r+1, 2c]
OO[r,c] = input_extended[2r+1, 2c+1]
```

**Fact.** The current reader stages complete source tiles for a 64-by-64 logical macro-region and waits once per batch (`tt-wavelet/kernels/dataflow/lwt_2d_reader.cpp:178-225`). A complete interior macro-tile traverses EE/EO/OE/OO together with precomputed face offsets and no boundary mapping (`tt-wavelet/kernels/dataflow/lwt_2d_reader.cpp:508-588`). Incomplete or boundary tiles use a separate extension-aware fallback (`tt-wavelet/kernels/dataflow/lwt_2d_reader.cpp:590-677`). This fused split is current behavior and must not regress to four independent `initialize_plane` traversals.

**Fact.** The compute kernel uses compile-time scheme steps, fuses terminal scales when eligible, and runs vertical or horizontal dense stencils (`tt-wavelet/kernels/compute/lwt_2d_compute.cpp:19-116`, `tt-wavelet/kernels/compute/lwt_2d_compute.cpp:119-211`). It is intentionally fused at the route/tile level.

**Fact.** Intermediate route output is persisted with full-tile local NoC writes, and aligned terminal bands use batched full-tile writes; only misaligned fragments use the scalar scratch path (`tt-wavelet/kernels/dataflow/lwt_2d_writer.cpp:46-75`, `tt-wavelet/kernels/dataflow/lwt_2d_writer.cpp:135-207`).

### 3.4 Buffers, execution, and caching

**Fact.** Standalone 1D execution owns three workspace slots, final outputs, route and chunk configuration, and telemetry (`tt-wavelet/tt_wavelet/include/lifting/device.hpp:23-70`). Standalone 2D owns five L1 planes, four output MeshBuffers, three metadata buffers, core ownership, and telemetry (`tt-wavelet/tt_wavelet/include/lifting/device_2d.hpp:27-73`).

**Fact.** The 2D protocol accounts for 32-by-32 FP32 tiles, five planes, four bands, and boundary-dependent split scratch. Symmetric needs at most 3-by-3 staged tiles; the general modes reserve up to 5-by-5 (`tt-wavelet/tt_wavelet/include/device_protocol/lwt_2d_config.hpp:11-35`). Chunk, route, and band pages are each 128 bytes and 64-byte aligned (`tt-wavelet/tt_wavelet/include/device_protocol/lwt_2d_config.hpp:37-101`).

**Fact.** Current buffers are Metalium `MeshBuffer` objects rather than TTNN tensors. Programs are manually created, runtime arguments are set per core, and a program or workload is enqueued. The benchmark executables manually enable the Metal program cache and reuse prepared executables (`tt-wavelet/main.cpp:537-567`).

**Inference.** The current manual executable reuse is analogous to, but not integrated with, the TTNN operation program cache. A TTNN port must make compile-affecting values explicit operation attributes and address-only values explicit bindings/runtime arguments.

### 3.5 Architecture policy

**Fact.** Wormhole defaults to row-major inverse workspace, non-direct final interleave, and compact 2D reader code. Blackhole defaults to tile-native inverse workspace, direct final interleave, and non-compact reader code (`tt-wavelet/tt_wavelet/include/lifting/policy.hpp:12-45`).

**Fact.** Forward 2D compact boundary code is enabled when architecture policy requests it, route count is at least 52, or the boundary mode is antireflect (`tt-wavelet/tt_wavelet/src/lifting/device_2d.cpp:550-567`). In the reader, compact mode changes large polyphase helpers to noinline and uses runtime parity inside one helper rather than four parity-specialized copies (`tt-wavelet/kernels/dataflow/lwt_2d_reader.cpp:360-487`). It does not mark the fused full interior split loop noinline.

**Fact.** The NCRISC gate checks every successful matching Wormhole ELF, applies a `0x4000` executable-load-segment limit, reports headroom, and reports the maximum `.text` and load-segment sizes (`scripts/check_ncrisc_elf_size.py:11-20`, `scripts/check_ncrisc_elf_size.py:75-125`, `scripts/check_ncrisc_elf_size.py:180-201`).

### 3.6 Current test, benchmark, and Python exposure

**Fact.** The current source tree has no checked-in `tt-wavelet/tests` directory and its CMake file registers only the four executables, not unit-test or Python-extension targets (`tt-wavelet/CMakeLists.txt:1-50`). Validation is primarily orchestration scripts. The 2D extension validator compares device bands and roundtrip outputs with PyWavelets and records tolerance/error fields (`scripts/validate_lwt_2d_extension_modes.py:21-101`). The 1D layout benchmark states that it measures the C++ device-only boundary and distinguishes strict reference cases from known FP32 limitations (`scripts/benchmark_lwt_ilwt_layouts.py:1-16`).

**Fact.** There is no nanobind/pybind target in the standalone CMake surface. Python scripts drive the command-line executables (`scripts/validate_lwt_2d_extension_modes.py:24-29`, `scripts/validate_lwt_2d_extension_modes.py:160-181`); that is validation orchestration, not a Python tensor API.

**Fact.** Current 1D and 2D input/workspace/output objects are replicated `MeshBuffer`s even when their local storage is DRAM or L1 (`tt-wavelet/tt_wavelet/src/lifting/device.cpp:160-222`, `tt-wavelet/tt_wavelet/src/lifting/device_2d.cpp:142-179`). Therefore current full-mesh stamping represents replication, not a spatially partitioned wavelet.

**Recommendation.** The migration lock must count actual executed cases rather than count scripts. It must report forward, inverse, roundtrip, architecture, shape, mode, numerical, JIT, memory, and timing coverage separately.

## 4. Survey of TTNN Operation Patterns

### 4.1 Operation registration and ownership

**Fact.** The pinned tree has 32 top-level operation-family directories. Stable migrated families are registered through per-family CMake targets and an umbrella target (`tt-metal/ttnn/cpp/ttnn/operations/CMakeLists.txt:15-53`). Experimental families use the same ownership model in their own CMake file (`tt-metal/ttnn/cpp/ttnn/operations/experimental/CMakeLists.txt:18-75`).

**Recommendation.** Wavelet should be one owned family:

```text
ttnn/cpp/ttnn/operations/wavelet/
```

with its own `CMakeLists.txt` or `sources.cmake`, public headers, device operations, program factories, kernels, and nanobind registration. Generated scheme artifacts should remain within that family and have an explicit code owner.

### 4.2 Device-operation contract

**Fact.** Current device operations conventionally define:

```text
operation_attributes_t
tensor_args_t
spec_return_value_t
tensor_return_value_t
program_factory_t
select_program_factory
validate_on_program_cache_miss
compute_output_specs
create_output_tensors
```

Conv2d is a representative single-output primitive (`tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_device_operation.hpp:25-45`). TopK is the most relevant fixed multi-output example: it returns a tuple of two `TensorSpec`s and two tensors and supports optional preallocated outputs (`tt-metal/ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.hpp:16-58`).

**Fact.** `device_operation::launch` validates that all inputs are allocated device tensors, creates outputs, derives the MeshDevice, updates output topology, and launches through the mesh adapter (`tt-metal/ttnn/api/ttnn/device_operation.hpp:442-502`).

**Recommendation.** Define four primitives rather than a runtime dimension/direction switch. Fixed return types are clearer, prevent invalid combinations, and allow exact output specs:

```text
ttnn::prim::LwtDeviceOperation       -> tuple<Tensor, Tensor>
ttnn::prim::IlwtDeviceOperation      -> Tensor
ttnn::prim::Lwt2DDeviceOperation     -> tuple<Tensor, Tensor, Tensor, Tensor>
ttnn::prim::Ilwt2DDeviceOperation    -> Tensor
```

Shared planners, scheme dispatch, validators, and kernel sources should remain common implementation utilities.

### 4.3 Output specification and multi-output behavior

**Fact.** TopK validates its shape/dtype/resources independently of allocation (`tt-metal/ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp:115-247`), creates distinct output specs, honors preallocated output specs, and allocates each output on the input device (`tt-metal/ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp:249-295`).

**Recommendation.** Wavelet forward primitives should copy this fixed-tuple pattern. Each band needs its own exact logical shape and physical layout; a list or a single concatenated tensor would weaken static arity, complicate inverse validation, and obscure LL/LH/HL/HH identity.

### 4.4 Data movement, layout conversion, and sharding

**Fact.** Pad selects factories by row-major/tile and sharded/interleaved state, validates device allocation, rank, layout, dtype, padding rules, sharding alignment, and output memory (`tt-metal/ttnn/cpp/ttnn/operations/data_movement/pad/device/pad_device_operation.cpp:70-211`). It separates logical and padded output shape in `TensorSpec` (`tt-metal/ttnn/cpp/ttnn/operations/data_movement/pad/device/pad_device_operation.cpp:213-233`).

**Fact.** `to_layout` is a public composite that may reshape padding and then choose tilize/untilize device paths (`tt-metal/ttnn/cpp/ttnn/operations/core/to_layout/to_layout_op.cpp:25-103`, `tt-metal/ttnn/cpp/ttnn/operations/core/to_layout/to_layout_op.cpp:106-220`). Reshard is a dedicated data-movement primitive under `data_movement/sharded/reshard`.

**Recommendation.** The wavelet primitive must reject unsupported layout/memory combinations rather than hide conversions in the cached device operation. A public convenience wrapper may perform an explicit `to_layout` outside the primitive in a later PR, where tracing and cost remain visible.

### 4.5 Reduction, normalization, and state

**Fact.** Reduction and normalization families demonstrate shape-changing outputs, optional statistics, and multiple factory selections. TopK demonstrates fixed tuple outputs. Normalization operations demonstrate that a composite semantic API can own one fused device primitive rather than expose every internal stage.

**Recommendation.** Wavelet is semantically closer to normalization than to a sequence of public data-movement operations: its split, predict/update routes, scale fusion, and band write are implementation stages and must not become public tensors.

### 4.6 CCL, distributed tensors, and mesh workloads

**Fact.** CCL operations expose top-level functions and device operations that can compute custom output topologies. All-gather is an example (`tt-metal/ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_device_operation.hpp:16-35`). Native `CachedMeshWorkload` is preferred over adapting single-device programs when an operation truly owns mesh behavior (`tt-metal/ttnn/api/ttnn/device_operation.hpp:34-44`).

**Fact.** A cached mesh workload owns a `MeshWorkload` and shared variables; the adapted form stamps single-device programs across ranges (`tt-metal/tt_metal/api/tt-metalium/program_cache.hpp:61-82`). `WorkloadDescriptor` can own per-range program descriptors and workload-lifetime buffers/semaphores (`tt-metal/tt_metal/api/tt-metalium/workload_descriptor.hpp:16-68`).

**Recommendation.** The first wavelet primitive must be genuinely single-device and reject spatially sharded multi-device tensors. A future spatial multi-chip operation should be a native mesh workload with explicit output topology and fabric exchanges, not an automatic stamping of the current program.

### 4.7 Experimental operations and caching

**Fact.** Experimental is an ownership/status namespace, not a license to omit validation, tests, bindings, or cache correctness (`tt-metal/ttnn/cpp/ttnn/operations/experimental/CMakeLists.txt:1-16`).

**Fact.** TTNN program keys combine a hash with collision-free canonical key material (`tt-metal/tt_metal/api/tt-metalium/program_cache.hpp:107-135`). Cache hits reapply descriptors or runtime arguments before enqueue (`tt-metal/ttnn/api/ttnn/device_operation.hpp:254-289`).

**Recommendation.** Land internal `ttnn::prim` operations first. Expose public `ttnn` APIs only after parity. If upstream policy requires an experimental incubation, preserve the final function signatures so promotion is a namespace/registration change rather than an API redesign.

### 4.8 Matmul and data-movement factory structure

**Fact.** Matmul separates reflected operation attributes and tensor arguments from a vector of output specs/tensors and a large factory variant (`tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.hpp:18-45`). Factory selection, validation, spec computation, and output allocation are distinct implementation phases (`tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2063-2104`, `tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2259-2263`, `tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2578-2610`).

**Inference.** This supports keeping planner-affecting wavelet values in reflected attributes and using factory variants for architecture/layout paths. Matmul's vector output is not appropriate for wavelet's fixed arity.

**Fact.** Data-movement primitives select different factories rather than force one generic kernel:

- reshape separates row-major and tiled factories and provides a custom program hash (`tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/reshape_device_operation.hpp:12-43`);
- transpose has eight layout/sharding factories and explicit dynamic cache-hit arguments (`tt-metal/ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_device_operation.hpp:23-74`);
- concat separates tiled, row-major, sharded, and multi-program factories (`tt-metal/ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.hpp:22-72`);
- reshard has same-width, same-height, generic, and N-D factories (`tt-metal/ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/reshard_device_operation.hpp:18-53`).

**Recommendation.** Wavelet should likewise keep strict first-PR factories and add new layout/sharding factories only when their access patterns are genuinely different. Do not branch through every layout in one large reader.

### 4.9 Source style and namespace findings

**Fact.** Public semantic declarations can live at top-level `ttnn`, as conv1d does (`tt-metal/ttnn/cpp/ttnn/operations/conv/conv1d/conv1d.hpp:15-52`). Implementation helpers and nanobind functions live in an operation-family namespace; conv1d binds the top-level function using the standard binding helper (`tt-metal/ttnn/cpp/ttnn/operations/conv/conv1d/conv1d_nanobind.cpp:24-89`). Device primitives live in `ttnn::prim`, as conv2d and TopK demonstrate (`tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_device_operation.hpp:25-45`, `tt-metal/ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.hpp:16-58`). Program factories and device kernels are owned below the operation's `device/` tree.

**Recommendation.** Exact namespace allocation:

| Artifact | Namespace |
|---|---|
| public functions/result aliases/boundary enum | `ttnn` |
| public registered operation constants, if the chosen registration API uses them | `ttnn` |
| public configuration accepted by more than one wavelet function | `ttnn` |
| private scheme ID and planners | `ttnn::operations::wavelet` |
| device-operation attributes and primitive launchers | `ttnn::prim` |
| program factories | `ttnn::prim` or their enclosing device-operation type, matching approved framework style |
| kernel-only host/device-common helpers | `ttnn::operations::wavelet::detail` |
| nanobind registration function | `ttnn::operations::wavelet` |

The alternatives `ttnn::operations::signal_processing::wavelet` and a permanent `ttnn::operations::experimental::wavelet` add nesting without a current source-tree family precedent that benefits users. A private `ttnn::operations::wavelet` implementation plus top-level public functions is the most consistent split. Experimental may be a temporary exposure status, not the final namespace.

## 5. FFT Case Study

### 5.1 Search result

**Fact.** Exact filename-token searches for `fft`, `ifft`, and `fourier`, plus whole-word source searches across the pinned tree, found no current TTNN FFT/IFFT/Fourier operation, device operation, program factory, or kernel. The only relevant matches are model-level Fourier positional-embedding calculations, for example `tt-metal/models/experimental/detr3d/ttnn/position_embedding.py:16-28`; they are not TTNN operation implementations.

**Conclusion.** There is no FFT implementation in this revision that can serve as a source-layout, binding, cache, multi-output, or device-program precedent. This absence is a fact, not evidence that an FFT-like API exists elsewhere.

### 5.2 Useful comparison despite the absence

**Inference.** FFT and wavelet transforms would share some API concerns—shape-changing complex or multi-output results, inverse shape restoration, plan caching, architecture-specialized kernels, and large-input decomposition—but their mathematical and dataflow contracts differ:

| Topic | Hypothetical FFT | Lifting wavelet |
|---|---|---|
| Core dependency | global/butterfly stages | bounded local stencil cones |
| Boundary modes | usually none/periodic | eight semantic modes |
| Outputs | complex or real-packed | fixed low/high bands |
| Schemes | radix/twiddle strategy | 106 lifting factorizations |
| Spatial partition | stage-dependent all-to-all | halo exchange |
| Odd inverse shape | transform-length metadata | explicit original shape |

**Recommendation.** Do not model the wavelet public result as a complex-like packed tensor and do not delay integration waiting for an FFT precedent. Use current TopK, conv, pad, and CCL patterns instead.

| Design concern | FFT approach at `bb9bb15` | Conv approach | Other relevant op | Recommendation for wavelet |
|---|---|---|---|---|
| Public API | absent | top-level `ttnn::conv1d/conv2d` | TopK has top-level semantic API | four top-level semantic functions |
| Primitive/composite | absent | conv1d composite over fused conv2d | normalization uses fused semantic primitives | thin wrapper over one fused wavelet primitive |
| Output model | absent | variant/tuple options for dimensions and prepared weights | TopK fixed tuple specs/tensors | fixed two/four tensor tuples |
| Shape/layout | absent | explicit convolution shapes and slicing config | pad separates logical/padded shape | exact logical bands plus physical layout padding |
| Large input | absent | DRAM slicing | pad/reshard and prefetch patterns | planner-derived dependency chunks from DRAM |
| Multi-device | absent | operation-specific distributed variants | CCL owns topology/fabric | reject initially; later native mesh with halos |
| Architecture | absent | host policy, distinct NoC rates/config | CCL has WH/BH packet/fabric values | shared math plus explicit WH/BH policy |
| Reusable signal abstraction | none found | sliding-window config is convolution-specific | no generic wavelet/FFT plan | retain wavelet planner inside its family |

## 6. Conv and Other Relevant Case Studies

### 6.1 Conv1d composite versus conv2d primitive

**Fact.** `ttnn::conv1d` reshapes its input and weights, maps padding, applies slicing policy, delegates to `ttnn::conv2d`, and normalizes variant outputs (`tt-metal/ttnn/cpp/ttnn/operations/conv/conv1d/conv1d.cpp:38-123`). Conv2d owns factory selection, validation, output specification, performance modeling, and explicit L1 estimation (`tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_device_operation.hpp:29-72`).

**Inference.** Conv1d shows that a public composite can normalize an interface around another fused primitive. It does not justify expressing lifting steps as generic TTNN operations: conv1d delegates to an existing single fused kernel family, whereas a generic wavelet decomposition would create many intermediate tensors and dispatches.

**Recommendation.** Keep the public wrapper thin like conv1d, but keep the complete transform in one wavelet primitive like conv2d.

### 6.2 Conv slicing and large inputs

**Fact.** Conv1d can route long sequences through DRAM-width slicing rather than requiring the complete logical input in L1 (`tt-metal/ttnn/cpp/ttnn/operations/conv/conv1d/conv1d.cpp:73-88`).

**Inference.** This is a useful policy precedent, not reusable convolution code. Wavelet already has exact dependency-cone chunking; its DRAM streaming unit must be a planned wavelet chunk with boundary halo, not a convolution slice.

### 6.3 Architecture-specific policy

**Fact.** Conv utilities select different transfer burst assumptions for Wormhole and Blackhole, and conv performance models use architecture-specific NoC/clock decisions (`tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:41-49`, `tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.cpp:430-570`).

**Recommendation.** Follow that separation: public wavelet semantics stay architecture-independent, while a small internal policy object selects reader compaction, workspace layout, interleave, transfer constraints, core budget, and fidelity.

### 6.4 TopK, pad, matmul, and CCL lessons

| Case | Reusable lesson | Do not copy |
|---|---|---|
| TopK | fixed tuple specs/outputs and optional preallocation | its dtype/shape restrictions |
| Pad | logical/padded shape validation and layout factory selection | implicit padding as wavelet extension |
| Matmul | large factory variant and cache-key discipline | a vector return for fixed wavelet arity |
| Conv | L1 cost model and architecture policy | convolution slicing semantics |
| CCL | native mesh workload and output topology | inter-chip traffic hidden as local NoC |
| Prefetcher | overlap/streaming ideas for future work | experimental dependency in first PR |

## 7. Proposed TTNN Wavelet Architecture

### 7.1 Layering

```text
Python: ttnn.lwt / ilwt / lwt_2d / ilwt_2d
                          |
                          v
C++ public ttnn wrapper
  - normalize string -> WaveletSchemeId
  - normalize boundary enum
  - validate semantic shape arguments
  - select defaults
                          |
                          v
ttnn::prim device operation (one per public transform)
  - cache-miss validation
  - exact output TensorSpec tuple
  - factory/policy selection
  - program-cache attributes and bindings
                          |
                          v
shared wavelet implementation
  - generated scheme registry
  - forward/inverse planners
  - 1D/2D dependency cones
  - Wormhole/Blackhole policy
                          |
                          v
fused reader -> SFPU lifting -> writer
```

**Recommendation.** Public namespaces contain only semantic APIs, enums, and result aliases. Internal planner and kernel concepts remain implementation details under `ttnn::operations::wavelet` or `ttnn::prim`; `ttwv` must not appear in the installed public API.

### 7.2 Primitive boundary

**Recommendation.** Each call launches one primitive and returns final device tensors. The primitive owns:

- parity split/interleave;
- boundary extension;
- route execution;
- terminal scale fusion;
- local plane/workspace allocation;
- final band writes;
- architecture policy;
- per-core chunk scheduling.

It must not expose EE/EO/OE/OO, route pages, local planes, padded workspaces, or synchronization tensors.

### 7.3 Preserved mathematical/dataflow invariants

**Recommendation.**

1. Preserve generated coefficient bit patterns and step order.
2. Preserve FP32 inputs, outputs, and destination accumulation until separately validated.
3. Preserve all eight logical boundary modes.
4. Preserve exact original-shape restoration for inverse.
5. Preserve fused terminal scale behavior.
6. Preserve the fused full-interior 2D split and tiled split dataflow.
7. Preserve dependency-cone chunking and exact L1 rejection.
8. Preserve Wormhole compact boundary/fallback code only; never noinline the fused interior split.
9. Preserve Blackhole tile-native and direct-interleave paths.
10. Make program cache hits bitwise-equivalent in runtime arguments and buffer bindings to misses.

### 7.4 Factory structure

**Recommendation.** The logical factory variants should be small and policy-based:

```text
1D forward:  row-major interleaved
1D inverse:  Wormhole row-major | Blackhole tile-native
2D forward:  compact-boundary | normal-boundary
2D inverse:  architecture-selected existing path
```

Route count and antireflect may select compact reader code on either architecture, while Wormhole always selects it. These are program-affecting variants and therefore cache-key material.

**Open question.** Whether those variants return `ProgramDescriptor`, `WorkloadDescriptor`, or `ProgramSpec` must be resolved with the TTNN framework owner. The selected path must provide native tensor/buffer bindings and dynamic scalar arguments without calling the forbidden temporary patching API directly.

## 8. Public API and Namespace Contract

### 8.1 C++ surface

**Recommendation.** Install these names in top-level `ttnn`:

```cpp
namespace ttnn {

enum class WaveletBoundaryMode : std::uint8_t {
    Zero,
    Constant,
    Symmetric,
    Reflect,
    Periodic,
    Smooth,
    Antisymmetric,
    Antireflect,
};

using LwtResult = std::tuple<Tensor, Tensor>;
using Lwt2DResult = std::tuple<Tensor, Tensor, Tensor, Tensor>;

LwtResult lwt(
    const Tensor& input,
    std::string_view wavelet,
    WaveletBoundaryMode mode = WaveletBoundaryMode::Symmetric,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<LwtResult>& output_tensors = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

Tensor ilwt(
    const Tensor& approximation,
    const Tensor& detail,
    std::string_view wavelet,
    std::uint32_t original_length,
    WaveletBoundaryMode mode = WaveletBoundaryMode::Symmetric,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

Lwt2DResult lwt_2d(
    const Tensor& input,
    std::string_view wavelet,
    WaveletBoundaryMode mode = WaveletBoundaryMode::Symmetric,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Lwt2DResult>& output_tensors = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

Tensor ilwt_2d(
    const Tensor& ll,
    const Tensor& lh,
    const Tensor& hl,
    const Tensor& hh,
    std::string_view wavelet,
    std::array<std::uint32_t, 2> output_shape,
    WaveletBoundaryMode mode = WaveletBoundaryMode::Symmetric,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn
```

Complete declarations, internal primitive signatures, and Python signatures are in Appendix C.

### 8.2 Scheme representation

**Recommendation.** Accept canonical PyWavelets-compatible names as strings at the public boundary and immediately map them to a validated internal `WaveletSchemeId`. Reasons:

- strings are natural in Python and match the existing JSON/generated registry;
- a closed internal enum gives exhaustive dispatch and stable cache keys;
- an installed public 106-value enum would make future generated additions an API/ABI burden;
- accepting arbitrary coefficient tensors in the first PR would change specialization, validation, caching, and numerical scope.

Unknown names must fail before output allocation and list or link to supported names. Aliases such as `haar` and `db1` may remain distinct accepted names even if they share mathematics, because cache and user-visible naming should be deterministic.

**Open question.** Should C++ additionally expose a public `Wavelet` value object for repeated calls? The recommended first answer is no: the program cache already amortizes compilation, and a string plus registry lookup is negligible compared with dispatch.

### 8.3 Output and inverse-input representation

**Recommendation.**

- `lwt` returns `(approximation, detail)`.
- `lwt_2d` returns `(ll, lh, hl, hh)` in that exact order.
- `ilwt` accepts two separate tensors.
- `ilwt_2d` accepts four separate tensors.
- Python returns normal fixed tuples, so unpacking is direct.
- C++ uses fixed `std::tuple`, following the TopK device-operation pattern.

A vector obscures arity; a struct complicates nanobind and differs from normal TTNN multi-output conventions; concatenation forces an artificial packing layout.

### 8.4 Shape restoration

**Recommendation.** `original_length` and 2D `output_shape` are mandatory inverse arguments. Coefficient lengths do not uniquely encode odd original extents under every boundary mode. Do not infer an ambiguous shape and do not smuggle host-only metadata inside device tensors.

**Recommendation.** A future multilevel API may return a lightweight host descriptor containing original shapes and level ordering, but single-level tensor results remain ordinary tensors.

### 8.5 Python surface

```python
approximation, detail = ttnn.lwt(
    input,
    wavelet="db7",
    mode="symmetric",
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

output = ttnn.ilwt(
    approximation,
    detail,
    wavelet="db7",
    original_length=65,
    mode="symmetric",
)

ll, lh, hl, hh = ttnn.lwt_2d(input, wavelet="db7", mode="smooth")
output = ttnn.ilwt_2d(
    ll, lh, hl, hh,
    wavelet="db7",
    output_shape=(64, 64),
    mode="smooth",
)
```

**Recommendation.** Python validates strings and integer shape ranges, then delegates tensor validation to the same C++ path. Documentation must name all eight accepted lowercase mode strings.

### 8.6 Promotion policy

**Recommendation.** Public naming is final, but promotion is gated:

1. internal `ttnn::prim` parity on both architectures;
2. cache-hit/miss parity and program-cache tests;
3. C++ public wrapper and nanobind;
4. performance non-regression against standalone;
5. stable top-level registration.

If upstream requires incubation, expose identical argument names and result ordering under `ttnn.experimental` temporarily. Do not maintain divergent stable and experimental semantics.

## 9. Tensor, Shape, Layout, and Memory Contract

### 9.1 TTNN tensor facts

**Fact.** TTNN tensors distinguish logical shape, padded shape, page layout, memory configuration, and optional sharding through `TensorSpec` (`tt-metal/tt_metal/api/tt-metalium/experimental/tensor/spec/tensor_spec.hpp:19-109`). Public tensors expose dtype, layout, logical/padded shape, memory configuration, topology, and shard information (`tt-metal/ttnn/api/ttnn/tensor/tensor.hpp:187-203`).

**Fact.** Layout choices are row-major and tile (`tt-metal/tt_metal/api/tt-metalium/experimental/tensor/spec/layout/layout.hpp:9-13`). Memory configuration separates interleaved/sharded layout from DRAM/L1 placement and defaults to interleaved DRAM (`tt-metal/tt_metal/api/tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp:25-47`).

### 9.2 First-PR tensor contract

| Operation | Input rank | Input layout | Input memory | Output logical shape | Output layout/memory |
|---|---:|---|---|---|---|
| `lwt` | 1 | row-major | interleaved DRAM | planner-derived low/high lengths | row-major, interleaved DRAM |
| `ilwt` | two rank-1 | row-major | same single device, interleaved DRAM | `[original_length]` | row-major, interleaved DRAM |
| `lwt_2d` | 2 | tile | interleaved DRAM | four equal planner-derived band shapes | tile, interleaved DRAM |
| `ilwt_2d` | four rank-2 | tile | same single device, interleaved DRAM | `output_shape` | tile, interleaved DRAM |

**Recommendation.** Require exact ranks initially. Do not silently flatten leading dimensions. Batch/channel semantics affect independent work ownership, output shapes, cache keys, and later mesh sharding, so they deserve an explicit follow-up contract.

**Recommendation.** Require every inverse input to have equal dtype, layout, memory placement, device, and expected logical band shape. Do not accept mixed-device or mixed-layout coefficients.

### 9.3 Logical versus padded shapes

**Recommendation.**

- 1D row-major output `TensorSpec` uses the exact logical coefficient length; physical page alignment remains a TTNN storage detail.
- 2D tile output `TensorSpec` uses the exact logical band height/width and a minimal tile-aligned padded shape.
- inverse output uses the explicit logical original/output shape and minimal layout padding.
- kernels receive both logical and physical strides where needed.
- extension always sees logical coordinates.

Physical zeros are neither `zero` boundary extension nor samples for another mode. This distinction is mandatory for odd and non-tile-aligned shapes.

### 9.4 Output allocation and preallocation

**Recommendation.** The primitive supports optional preallocated outputs in the first implementation because it is a standard TTNN allocation pattern and enables pipelines to control lifetime. Validation must require exact specs, device, layout, memory configuration, and non-aliasing unless an alias is explicitly proven safe.

**Recommendation.** Expose those optional outputs publicly in C++ and Python only if the upstream API owner accepts the standard TTNN spelling. They may be omitted from the initial documentation examples without changing the primitive.

**Recommendation.** Reject output/input aliasing in the first PR. In-place wavelet transforms are not part of current semantics, and the pinned descriptor adapter treats alias correctness as a sensitive cache concern (`tt-metal/tt_metal/api/tt-metalium/experimental/program_descriptor_patching.hpp:106-126`).

### 9.5 Dtype and special values

**Recommendation.** FP32 only. Reject BF16/BFP/int inputs rather than cast.

NaN and infinity handling follows hardware FP32 operations:

- no host scan;
- no promise to preserve NaN payload bits;
- deterministic finite inputs must meet the numerical contract;
- dedicated tests record propagation behavior for NaN, `+/-inf`, signed zero, maximum finite values, and subnormals where the hardware/runtime supports them.

**Open question.** Whether special-value propagation is normative or informational. Recommended: finite-input accuracy is normative; special-value tests are compatibility guards without a cross-architecture bitwise promise.

## 10. Large-Input Execution

“Does not fit” has three distinct meanings and must not be collapsed into one feature.

### 10.1 A. One core's L1 is insufficient

**Fact.** The current planners compute dependency cones, workspace rectangles, CB usage, metadata, synchronization, and headroom per chunk before selecting candidates (`tt-wavelet/tt_wavelet/include/lifting/plan_2d.hpp:512-680`, `tt-wavelet/tt_wavelet/include/lifting/plan_2d.hpp:851-1055`). The protocol separately accounts for boundary split scratch (`tt-wavelet/tt_wavelet/include/device_protocol/lwt_2d_config.hpp:18-35`).

**Recommendation.**

1. Search smaller output chunks.
2. Recompute exact halo/dependency storage for each candidate.
3. Select based on an architecture-calibrated latency cost, not maximum cores alone.
4. Fail before program creation if the smallest legal chunk cannot fit.
5. Report required bytes by workspace, CB, metadata, synchronization, architecture scratch, and headroom.

Never overcommit L1 and never infer fit from logical tensor size alone.

### 10.2 B. Tensor fits DRAM but not aggregate L1

**Recommendation.** This is the normal large-device-tensor case. Keep the full TTNN tensor in DRAM and stream dependency-cone chunks:

```text
DRAM input window + halo
        -> one core's staged tiles/L1 planes
        -> fused routes
        -> DRAM output band rectangle
        -> next chunk
```

Aggregate worker L1 is not a second tensor store and the full image need not reside there. Each chunk owns a disjoint final output region; halo inputs may overlap. Metadata is loaded once per chunk or in batches. The operation remains one TTNN launch even when each core processes multiple chunks.

**Recommendation.** Use TTNN's performance and program-cache machinery, but preserve the existing exact wavelet planner. Conv slicing is only a policy precedent (`tt-metal/ttnn/cpp/ttnn/operations/conv/conv1d/conv1d.cpp:73-88`).

### 10.3 C. Tensor does not fit device DRAM

**Recommendation.** This is not a valid first-PR TTNN tensor operation because an allocated device input/output tensor cannot exist at the requested size. Do not claim transparent support.

A later out-of-core composite may:

1. accept host/pinned slices or an explicit stream object;
2. compute exact input halos for each output interval/rectangle;
3. double-buffer host-to-device and device-to-host transfers;
4. invoke the normal device wavelet primitive on DRAM-resident windows;
5. crop duplicate halo outputs;
6. preserve logical boundary semantics only at global edges;
7. return host or externally managed storage.

**Open question.** Whether that belongs in TTNN, a data-loader/streaming library, or the application. Recommended: keep it outside the first wavelet op and design it with the runtime streaming owners.

### 10.4 Limits and errors

**Recommendation.** Checked arithmetic is required for logical/padded element counts, byte counts, page counts, route offsets, and 32-bit kernel arguments. Error messages must distinguish:

- invalid shape;
- unsupported layout/memory;
- one-chunk L1 infeasibility;
- device DRAM allocation failure;
- 32-bit protocol overflow;
- unsupported out-of-core request.

## 11. Multi-Chip and Mesh Execution

### 11.1 Hardware/runtime facts

**Fact.** Distributed tensor configuration distinguishes replication from sharding and maps tensors onto a mesh (`tt-metal/tt_metal/api/tt-metalium/experimental/distributed_tensor/topology/distributed_tensor_configs.hpp:15-85`). `MeshBuffer` likewise distinguishes replicated and sharded configuration (`tt-metal/tt_metal/api/tt-metalium/mesh_buffer.hpp:39-73`).

**Fact.** The checked-in performance documentation identifies N150 as one device and N300 as two devices (`tt-metal/tech_reports/FlashAttention/FlashDecode.md:163-171`). The Ethernet guide describes N300 as a two-chip board with one PCIe-local chip and one Ethernet-reachable remote chip (`tt-metal/tech_reports/EthernetMultichip/BasicEthernetGuide.md:153-159`).

**Fact.** N300 dispatch setup uses Ethernet dispatch paths (`tt-metal/ttnn/core/device.cpp:17-23`), and platform code treats an N300 card as an MMIO chip plus a remote chip (`tt-metal/tt_metal/llrt/tt_cluster.hpp:484-492`).

**Fact.** NoC is chip-local. TT-Fabric sends inter-chip packets through Ethernet/fabric and describes fabric as extending the communication model rather than turning remote traffic into ordinary local NoC (`tt-metal/tech_reports/TT-Fabric/TT-Fabric-Architecture.md:204-209`, `tt-metal/tech_reports/TT-Fabric/TT-Fabric-Architecture.md:822-845`).

**Conclusion.** Creating a mesh or running on N300 does not automatically give correct two-chip wavelet semantics. Tensor placement, program ranges, ownership, and inter-chip exchange must be explicit.

### 11.2 Mode A: replicated execution

**Recommendation.** Later support. Each chip owns a complete independent input and produces complete independent outputs. The operation may stamp the same program across a mesh only when tensor topology is replicated and every replica is semantically independent.

No wavelet halo crosses chips. This is useful for replicated inference or redundant execution, not for speeding up one image.

### 11.3 Mode B: batch sharding

**Recommendation.** First multi-chip feature after rank/batch support. Independent batch items are assigned to chips; each item runs the single-device program. No transform-axis halo crosses chips. Output topology mirrors input batch placement.

This requires a defined leading-dimension contract, which is intentionally absent from the first PR.

### 11.4 Mode C: 1D spatial sharding

**Recommendation.** Later native mesh operation. Partition final low/high coefficient intervals, run the planner backward to derive exact source halos, and exchange or preload those halos explicitly.

Required design:

- owner of each output coefficient interval;
- left/right halo source ranges per route;
- global boundary mode only at the signal ends;
- fabric/CCL transport for cross-chip halos;
- no duplicate final ownership;
- per-chip runtime descriptors;
- topology-aware cache key.

The dependency radius grows across lifting routes; a fixed filter-half-width halo is not generally sufficient.

### 11.5 Mode D: 2D spatial sharding

**Recommendation.** Last multi-chip mode. Partition final band rectangles and propagate each through both horizontal and vertical route dependencies. Edges require row halos, column halos, and corners. LL/LH/HL/HH ownership and placement must be explicit.

Potential strategies:

1. pre-exchange full dependency rectangles with fabric/CCL;
2. replicate limited boundary slabs;
3. pipeline route-level halo exchange.

The first is the safest initial spatial implementation. Route-level pipelining should follow measured need.

### 11.6 First-PR mesh validation

**Recommendation.** Accept only tensors whose storage resolves to one physical device. Reject:

- sharded multi-device tensors;
- spatially sharded MeshTensors;
- a replicated multi-device tensor unless replicated semantics have dedicated tests;
- coefficient tensors with different topology.

The error should name future supported directions rather than silently selecting one mesh coordinate.

### 11.7 Answers for the pinned mesh runtime

1. **Does TTNN invoke the operation independently on every shard?** **Fact:** the adapter derives covered tensor coordinates and builds or stamps programs over their ranges; uniform storage takes a whole-mesh fast path (`tt-metal/ttnn/api/ttnn/device_operation.hpp:315-328`). Whether work is independent is operation-defined.
2. **How is a local shard obtained?** **Fact:** tensor storage/topology maps buffers to mesh coordinates, and nonuniform inputs cause output shard filtering (`tt-metal/ttnn/api/ttnn/device_operation.hpp:479-495`). The operation receives tensor arguments plus a mesh dispatch coordinate in per-coordinate factories.
3. **Is orchestration automatic?** **Conclusion:** basic coordinate dispatch/cache plumbing is automatic; semantic partitioning and communication are operation-specific.
4. **How are output distributed specs created?** **Fact:** the op returns normal `TensorSpec`s and may provide custom output topologies before the framework updates outputs (`tt-metal/ttnn/api/ttnn/device_operation.hpp:484-495`).
5. **What is cached?** **Fact:** the MeshDevice program cache stores a cached program or cached mesh workload keyed by canonical operation material (`tt-metal/tt_metal/api/tt-metalium/program_cache.hpp:61-135`). Architecture must be device-scoped or explicit policy material.
6. **Can chips have different runtime arguments?** **Fact:** yes; native/per-coordinate workload factories retain shared variables by `MeshCoordinateRange`, and descriptors/programs are created for ranges (`tt-metal/ttnn/api/ttnn/device_operation.hpp:72-108`).
7. **Which operations partition across devices?** **Fact:** CCL operations such as all-gather own output topology and communication rather than merely replicate (`tt-metal/ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_device_operation.hpp:16-35`).
8. **Are fabric kernels needed for cross-chip communication?** **Conclusion:** yes, either directly or through a CCL/fabric primitive; ordinary on-chip NoC calls are insufficient.
9. **How should wavelet halos move?** **Recommendation:** compute exact host-side dependency rectangles, then use explicit fabric/CCL transfer into per-chip DRAM/L1 before the consuming route.
10. **Can communication be avoided at runtime?** **Recommendation:** sometimes. Host or a preceding op may duplicate complete dependency-expanded halos before launch. This trades memory/bandwidth for a communication-free transform and must be selected by cost, not assumed.

## 12. Wormhole and Blackhole Compatibility

### 12.1 Shared semantics, policy-selected implementation

**Recommendation.** Keep one logical planner, generated scheme registry, boundary library, and mathematical kernel source. Select an internal `WaveletArchitecturePolicy` using the device architecture.

| Policy | Wormhole B0 | Blackhole |
|---|---|---|
| 2D boundary reader | compact always | normal unless routes `>=52` or antireflect |
| 1D inverse workspace | row-major | tile-native default |
| final inverse interleave | existing non-direct path | direct when tile-native |
| interior 2D split | fused tiled split | fused tiled split |
| reader size gate | executable segment `<=0x4000` | gate not applicable |
| NoC/alignment | Wormhole-specific validated values | Blackhole-specific validated values |
| performance target | preserve correctness/fit | preserve tile-native speed |

**Fact.** The standalone policy already encodes the first three differences (`tt-wavelet/tt_wavelet/include/lifting/policy.hpp:21-45`).

| Concern | Current tt-wavelet behavior | Wormhole requirement | Blackhole requirement | Proposed TTNN handling |
|---|---|---|---|---|
| Mathematical source | shared lifting/scheme code | same coefficients/order | same coefficients/order | one shared private implementation |
| Reader code size | policy plus route/mode threshold | compact boundary/fallback | normal fast code unless complex case | factory compile define in cache key |
| Interior split | fused tiled traversal | remain inline/fused | remain inline/fused | never apply boundary noinline policy to it |
| Inverse workspace | architecture policy | row-major | tile-native default | internal factory policy |
| Final interleave | architecture policy | existing non-direct | direct tile-native | internal writer variant |
| NoC selection | accessors and architecture constraints | WH burst/alignment | BH burst/alignment | public HAL/preferred-NoC APIs |
| Compute config | HiFi4 plus FP32 in standalone | resolve HiFi3 warning | validate current HiFi4 behavior | architecture default plus owner-approved override |
| CB/L1 | planner/resource model | account WH scratch/code | account BH tile path | factory calculates exact per-core bytes |
| Kernel defines | scheme/mode/policy | `WORMHOLE_B0` JIT target and compact flag | `BLACKHOLE` JIT target | rely on normal JIT architecture defines; no public arch argument |
| Runtime-arg limits | manually planned protocol | retain within WH firmware limits | validate BH limits | checked descriptor construction |
| Fabric | no spatial multi-chip algorithm | explicit for remote halos | explicit for remote halos | future native mesh/CCL path |
| ELF gate | external script | mandatory `0x4000` | skip | architecture CI gate |

### 12.2 Wormhole NCRISC compaction

**Recommendation.** Preserve exactly this forward selection:

```cpp
const bool compact_boundary_code =
    arch == tt::ARCH::WORMHOLE_B0 ||
    route_count >= kCompactBoundaryRouteThreshold ||
    boundary_mode == WaveletBoundaryMode::Antireflect;
```

Compaction applies to boundary/fallback helpers. The full interior macro-tile loop remains the fused tiled implementation and must not be marked noinline (`tt-wavelet/kernels/dataflow/lwt_2d_reader.cpp:508-588`).

Every successful Wormhole `lwt_2d_reader` specialization generated by tests must pass the existing executable-segment gate, and CI must report maximum `.text`, maximum executable segment, and remaining headroom (`scripts/check_ncrisc_elf_size.py:193-201`).

### 12.3 FP32 fidelity blocker

**Fact.** The pinned TTNN source warns about the exact current combination of HiFi4 and FP32 destination accumulation on Wormhole (`tt-metal/ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.cpp:43-67`).

**Recommendation.** Before integration sign-off, run a controlled matrix:

```text
architecture: Wormhole B0, Blackhole
fidelity: HiFi3, HiFi4
fp32_dest_acc_en: true
schemes: db1, db7, bior3.9, synthetic-k17, coif17 JIT-only
modes: all eight
shapes: odd, even, tile-edge, large
```

Choose HiFi3 on Wormhole only if it preserves or improves the established finite-input tolerance and does not violate performance acceptance. Do not change coefficient order or fuse arithmetic differently to compensate.

**Open question.** Is the standalone HiFi4 result part of the required bitwise baseline, or may Wormhole move to HiFi3 for correctness? Recommended: finite-reference accuracy takes precedence over bitwise parity, but the change must be explicit and separately reviewed.

### 12.4 Architecture headers and public API

**Fact.** The Metal JIT target supplies architecture-specific include trees and exactly one of `ARCH_WORMHOLE` or `ARCH_BLACKHOLE` (`tt-metal/tt_metal/jit_build/fake_kernels_target/CMakeLists.txt:52-92`, `tt-metal/tt_metal/jit_build/fake_kernels_target/CMakeLists.txt:144-159`). Production TTNN kernels can use public compute APIs and an architecture macro for a narrow workaround without directly selecting LLK trees; embedding backward is an example (`tt-metal/ttnn/cpp/ttnn/operations/embedding_backward/device/kernels/compute/embedding_backward.cpp:5-45`).

**Fact.** Standalone wavelet currently forwards `ckernel.h` directly to one architecture-specific LLK tree and has a small Wormhole-only address-modifier operation (`tt-wavelet/kernels/ckernel.h:3-14`, `tt-wavelet/kernels/sfpi/lwt_sfpi_common.h:5-33`).

**Recommendation.** Public APIs never expose architecture. Internal policy consumes `input.device()->arch()` and current TTNN compute configuration. Prefer checked-in public compute/SFPI APIs and the normal JIT-supplied macros. Remove the repository-relative LLK forwarding header during the port if public APIs express every required operation; retain one private, narrowly reviewed adapter only if an operation is demonstrably unavailable. Do not modify LLK, SFPI, or firmware to make wavelet fit.

## 13. Scheme Specialization and Program Caching

### 13.1 Initial specialization strategy

**Recommendation.** Preserve compile-time generated `Scheme` types. On a cache miss:

1. validate public name and get `WaveletSchemeId`;
2. dispatch the ID to a generated scheme type;
3. plan routes and resources;
4. create the selected specialized program descriptor/spec;
5. let JIT/program cache retain it.

This preserves coefficient bits, step count, compile-time `K`, fused scale selection, and present SFPI specialization.

**Fact.** The source registry already provides the closed dispatch domain and metadata (`tt-wavelet/tt_wavelet/include/schemes/generated/registry.hpp:120-345`).

### 13.2 Variant count

The minimum compute-specialization domain is:

```text
106 schemes * 2 directions * 2 dimensionalities * 2 architectures = 848
```

Reader/writer variants add boundary behavior, compact/non-compact choice, and layout policy. Eagerly materializing the full product would be wasteful.

**Recommendation.**

- compile only called variants;
- keep boundary mode out of compute-kernel identity when the compute source is identical;
- include it in reader/program identity when extension code is compile-time;
- canonicalize equivalent public aliases only if generated arithmetic and metadata are identical;
- expose program-cache telemetry in performance tests;
- set an explicit supported registry version/generator hash in build artifacts.

### 13.3 Cache-key contract

| Value | Cache key? | Reason |
|---|---|---|
| operation identity/direction/dimension | yes | different program |
| internal scheme ID | yes | coefficients and route count |
| architecture | yes/device-scoped | policy/kernel binary |
| boundary mode | yes when compile-time reader code differs | extension specialization |
| compact reader flag | yes | code generation differs |
| input logical and padded shape | yes initially | planner/core args/program shape |
| output logical/padded shapes | yes initially | band mapping/program shape |
| input/output layout and memory layout | yes | factory and accessors |
| core limit/grid and chosen chunk geometry | yes | program/core topology |
| compute kernel config | yes | kernel compilation |
| tensor buffer addresses | no | bindings/runtime args |
| metadata/output buffer addresses | no | bindings/runtime args |
| queue ID | no | dispatch property |
| host Python string object identity | no | canonical ID replaces it |

**Recommendation.** Start conservatively with shape-specific cache keys. Later omit shape fields only after every changed per-core scalar, CB address, and loop bound is proven dynamic and cache-hit parity-tested.

### 13.4 Cache-miss and cache-hit behavior

**Recommendation.**

```text
cache miss:
  validate -> compute specs -> allocate outputs -> plan -> build descriptor/spec
  -> bind input/output/metadata resources -> enqueue

cache hit:
  validate compatible dynamic state
  -> patch current input/output bindings
  -> patch all explicitly dynamic scalars
  -> enqueue without rebuilding the program
```

Use TTNN's canonical key mechanism rather than a custom 64-bit-only hash (`tt-metal/tt_metal/api/tt-metalium/program_cache.hpp:107-135`).

**Open question.** Metadata buffers are currently separately allocated Metalium objects. Recommended: encode invariant descriptor words in operation-owned workload resources retained by the cached program and rewrite only genuinely dynamic chunk addresses. The exact lifetime mechanism depends on the approved descriptor/spec API.

### 13.5 Runtime/hybrid coefficients

**Recommendation.** Defer runtime or hybrid coefficient programs. They could reduce variants, but may increase runtime loads, inhibit compile-time unrolling, enlarge generic control paths, worsen Wormhole NCRISC fit, and alter SFPI numerical order.

A follow-up is accepted only if it measures:

- cold JIT reduction;
- warm latency;
- ELF sizes on Wormhole;
- all 106 schemes and eight modes;
- numerical parity;
- program cache population;
- Blackhole performance.

## 14. Validation and Safety Checks

### 14.1 Public validation

Every call must reject, before enqueue:

1. unknown wavelet name;
2. unknown boundary mode;
3. zero extent;
4. rank other than the supported exact rank;
5. non-FP32 dtype;
6. unsupported row-major/tile layout;
7. non-interleaved or non-DRAM memory in the first PR;
8. host or unallocated tensors;
9. mixed physical devices/topologies;
10. unsupported multi-device storage;
11. inverse band length/shape mismatch;
12. inverse original/output shape inconsistent with coefficient geometry;
13. wrong preallocated output spec;
14. unsupported input/output alias;
15. checked-size or protocol integer overflow;
16. L1 infeasibility for the minimum chunk;
17. unsupported architecture.

The general TTNN launcher already enforces allocated device tensors (`tt-metal/ttnn/api/ttnn/device_operation.hpp:453-465`); wavelet-specific validation remains in the primitive.

### 14.2 Planner and resource invariants

**Recommendation.** Retain fatal assertions for:

- disjoint final output ownership;
- every required source index inside the dependency cone after extension;
- exact route rectangle containment;
- route order and storage-slot lifetime;
- no CB index collision;
- metadata page size/alignment;
- L1 total not exceeding allocatable capacity;
- scratch count covering the chosen boundary mode;
- all buffer offsets and transfer sizes aligned per architecture;
- output TensorSpec agreeing with planner shapes.

### 14.3 Build and generated-artifact safeguards

**Recommendation.**

- generator output must be reproducible and checked clean in CI;
- registry count must equal generated-header count;
- every name maps to one valid ID and every ID has forward/inverse types;
- generated coefficient bit patterns receive a manifest checksum;
- kernel includes must resolve only within installed/generated operation sources;
- no runtime file-system dependency on the standalone repository;
- no 106-way hand-written switch without generated exhaustiveness checks.

### 14.4 Pre-integration lock

**Recommendation.** Do not rely on “integrate first, add tests later.” Before moving code, save:

- exact standalone revision and old tt-metal revision;
- full registry manifest;
- representative strict correctness results;
- all-mode db7 results;
- coif17 clearly marked JIT-only;
- Wormhole ELF gate results;
- device-only latency telemetry;
- a list of known failures and unsupported cases.

This avoids confusing integration regressions with pre-existing numerical limitations.

### 14.5 Decision timing for remaining integration topics

| Topic | Timing | Required direction |
|---|---|---|
| API names and namespace | before implementation | four top-level names; private primitive |
| operation granularity | before implementation | one primitive per direction/dimension |
| scheme, boundary, output representation | before implementation | string/enum normalization and fixed tuples |
| inverse odd-shape metadata | before implementation | explicit host length/shape |
| rank, batch, channel semantics | before implementation | exact ranks first; no implicit batch/channel |
| dtype/layout/memory/buffer support | before implementation | FP32 native layout, interleaved DRAM |
| output memory and preallocation | before implementation | exact spec; primitive preallocation |
| architecture fidelity and compaction | before implementation | owner gate and WH size policy |
| cache-key boundaries | before implementation | conservative complete key |
| descriptor/ProgramSpec framework | before implementation | framework-owner approval |
| async behavior | during implementation | standard TTNN asynchronous return; no internal final `Finish` |
| queue ID | during implementation | follow the then-current registered-op convention; do not invent a private queue |
| tensor ownership/lifetime | during implementation | framework tensors and workload-owned metadata |
| thread safety | during implementation | constexpr registry; no mutable global scheme state |
| deterministic execution | during implementation | deterministic planner/core ownership for a fixed key |
| error mechanism | during implementation | normal TTNN validation/`TT_FATAL` conventions with actionable text |
| profiler/Tracy integration | during implementation | normal operation name and performance model hooks |
| persistent JIT cache packaging | during implementation | use upstream JIT/cache; report footprint |
| docs/examples | during public API PR | all four APIs, odd inverse, modes, limitations |
| CI duration split | during implementation | smoke per PR; scheduled exhaustive 106-by-mode jobs |
| single-device sharding | defer | measured factory follow-up |
| batch/channel axes | defer | explicit axis contract |
| replicated/batch multi-chip | defer | after batch support |
| spatial multi-chip | defer | native mesh/fabric design |
| multilevel | defer | composite and result metadata |
| out-of-core host streaming | defer | explicit non-Tensor orchestration contract |
| runtime/hybrid coefficients | defer | benchmarked specialization study |
| BF16/BFP formats | defer | new numerical contract |

**Recommendation.** Standard TTNN asynchronous semantics replace the standalone benchmark's internal `Finish`; callers synchronize through ordinary tensor/queue behavior. Timing tests insert synchronization only at the measurement boundary. Queue selection must follow the current upstream registered-operation convention at implementation time because the pinned operation surface is in active transition.

## 15. Testing and Performance Contract

### 15.1 Test layers

| Layer | Required coverage |
|---|---|
| generator/unit | registry completeness, step inversion, bit patterns |
| planner/unit | output shapes, dependency cones, ownership, L1 rejection |
| host API/unit | names, enums, rank, dtype, layout, memory, inverse shape errors |
| split/unit-device | EE/EO/OE/OO against scalar reference at tile boundaries |
| primitive/device | four ops, cache off/on, preallocation, address changes |
| Python | signatures, tuple ordering, errors, docs examples |
| architecture | Wormhole B0 and Blackhole policy branches |
| performance | cold compile, first dispatch, warm device latency, cache entries |
| ELF | all successful Wormhole reader ELFs within `0x4000` |

### 15.2 Numerical matrix

**Recommendation.** Before stable promotion:

- all 106 generated schemes compile for forward and inverse 1D/2D;
- all eight modes compile and execute;
- a defined strict subset has PyWavelets forward and roundtrip correctness on both architectures;
- every scheme has at least finite, bounded, deterministic cases appropriate to its known stability;
- odd/even, singleton, tile-edge, and large shapes are included;
- input patterns include zero, constant, ramps, alternating, impulses, bounded random, and large finite;
- NaN/infinity behavior is recorded separately.

The existing 2D validator already enumerates eight modes, representative shapes, and multiple input types (`scripts/validate_lwt_2d_extension_modes.py:31-64`, `scripts/validate_lwt_2d_extension_modes.py:108-157`).

**Fact.** Coif17 currently has an observed large-error comparison outside the JIT-only intent. Therefore:

```text
coif17 compile/execute pass != coif17 numerical pass
```

It must remain excluded from correctness claims until diagnosed and assigned a justified tolerance or fixed without weakening arithmetic.

### 15.3 Cache tests

For every primitive:

1. run with program cache disabled;
2. run miss then hit with same tensors;
3. hit with different buffers but same specs;
4. hit with preallocated outputs;
5. vary every excluded dynamic runtime argument;
6. confirm output equality and no cache growth;
7. vary each compile-affecting attribute and confirm a distinct key;
8. run descriptor/spec parity checks offered by the selected framework path.

### 15.4 Performance methodology

**Recommendation.** Report three distinct times:

- cold call including JIT;
- first prepared/enqueued call;
- median and percentile warm device execution.

Input upload and output readback remain outside device-only timing. Enable the program cache, reuse tensors and executable state, synchronize only at the defined timing boundary, and report active cores, chunks, route count, L1 bytes/headroom, layout, architecture, clock, and scheme.

### 15.5 Acceptance

**Recommendation.**

- no finite-input correctness regression versus the locked standalone baseline;
- no Blackhole db7 warm-latency regression larger than 5% unless approved;
- no Wormhole db7 warm-latency regression larger than 5%;
- Wormhole reader executable segments all `<=0x4000`;
- cache-hit latency does not rebuild/JIT;
- TTNN wrapper overhead is measured and reported;
- program-cache cardinality matches the documented key;
- no claim that TTNN integration itself fixes the broader latency floor.

The original diagnosis indicates fixed orchestration and data movement dominate. Integration should preserve the current fused optimizations; further performance work remains a separate measured program.

## 16. Proposed Source-Tree Layout

**Recommendation.** Use one operation family with shared implementation rather than four duplicated trees:

```text
ttnn/cpp/ttnn/operations/wavelet/
├── CMakeLists.txt
├── wavelet.hpp
├── wavelet.cpp
├── wavelet_types.hpp
├── wavelet_nanobind.hpp
├── wavelet_nanobind.cpp
├── common/
│   ├── boundary.hpp
│   ├── checked_shapes.hpp
│   ├── architecture_policy.hpp
│   ├── scheme_dispatch.hpp
│   └── tiling_2d.hpp
├── generated/
│   ├── registry.hpp
│   ├── manifest.json
│   └── schemes/*.hpp
├── planner/
│   ├── forward_1d.hpp
│   ├── inverse_1d.hpp
│   ├── forward_2d.hpp
│   ├── inverse_2d.hpp
│   └── resource_model.hpp
└── device/
    ├── wavelet_device_operation_types.hpp
    ├── lwt_device_operation.hpp
    ├── lwt_device_operation.cpp
    ├── ilwt_device_operation.hpp
    ├── ilwt_device_operation.cpp
    ├── lwt_2d_device_operation.hpp
    ├── lwt_2d_device_operation.cpp
    ├── ilwt_2d_device_operation.hpp
    ├── ilwt_2d_device_operation.cpp
    ├── program_factory_common.hpp
    ├── program_factory_common.cpp
    └── kernels/
        ├── dataflow/
        │   ├── lwt_reader.cpp
        │   ├── lwt_writer.cpp
        │   ├── lwt_2d_reader.cpp
        │   └── lwt_2d_writer.cpp
        ├── compute/
        │   ├── lwt_compute.cpp
        │   └── lwt_2d_compute.cpp
        ├── sfpi/
        └── primitives/
```

Tests and docs:

```text
tests/ttnn/unit_tests/operations/wavelet/
├── test_lwt.py
├── test_ilwt.py
├── test_lwt_2d.py
├── test_ilwt_2d.py
├── test_wavelet_cache.py
└── test_wavelet_validation.py

tests/ttnn/unit_tests/gtests/wavelet/
├── test_planners.cpp
├── test_generated_schemes.cpp
└── test_split_2d.cpp

tests/ttnn/perf_tests/operations/wavelet/
├── perf_lwt.py
└── perf_lwt_2d.py

ttnn/ttnn/operations/wavelet.py            # only if Python export convention requires it
ttnn/cpp/ttnn-nanobind/operations/wavelet.cpp
docs/source/ttnn/ttnn/api/operations/wavelet.rst
tools/wavelet/generate_static_schemes.py
```

**Fact.** Current stable operation families own their build targets under their family directory and link through the umbrella operation target (`tt-metal/ttnn/cpp/ttnn/operations/CMakeLists.txt:15-53`). Wavelet should follow that ownership pattern and add `TTNN::Ops::Wavelet`.

**Recommendation.** Keep generated code adjacent to its owner, but keep the generator in a tooling location so CI can reproduce it. The generated manifest records input JSON hashes, generator revision, count, maximum taps, maximum steps, and registry ordering.

**Recommendation.** Internal 32-by-16 FP32 CB tiles used by the current 1D compute path may remain an internal kernel page/CB geometry; current program creation uses that geometry (`tt-wavelet/tt_wavelet/src/lifting/device.cpp:418-502`). It must not appear as a nonstandard public TTNN tensor layout. Public tensors remain standard row-major or 32-by-32 tile tensors.

## 17. Migration and PR Plan

### 17.1 Coexistence principle

**Recommendation.** Do not delete or rewrite the standalone backend during initial integration. Maintain a compatibility period where the same generated schemes feed both backends, tests execute both, numerical results are compared automatically, and device latency is reported side by side.

The TTNN path becomes authoritative only after parity. Standalone removal, if ever desired, is a later owner decision.

### 17.2 Proposed PR sequence

1. **Baseline and generated artifacts.** Add a machine-readable scheme manifest, generator reproducibility test, locked standalone numerical/performance reports, known-failure list, and Wormhole ELF report. No TTNN op.
2. **Shared pure host code.** Move or copy boundary enums, checked shapes, static scheme definitions, and planners into the proposed family with C++ unit tests. Standalone remains buildable.
3. **1D internal primitives.** Add `ttnn::prim::lwt` and `ttnn::prim::ilwt`, FP32 row-major interleaved DRAM, single device, cache tests, no public Python.
4. **1D public API and bindings.** Add `ttnn::lwt`, `ttnn::ilwt`, documentation, Python tuples, and standalone-versus-TTNN parity.
5. **2D internal primitives.** Port the existing five-plane scheduler and fused interior split; retain boundary fallback, compact reader, architecture policy, and ELF gate.
6. **2D public API and bindings.** Add `ttnn::lwt_2d`, `ttnn::ilwt_2d`, tuple ordering, explicit output shape, tests, and performance gates.
7. **Stable promotion.** Complete 106-scheme compilation coverage, agreed correctness matrix, WH/BH performance sign-off, and API review; promote if prior PRs incubated under experimental.
8. **Batch and sharding.** Add leading batch semantics, single-device L1 sharding/resharding where measured, then batch-sharded multi-chip.
9. **Spatial mesh wavelets.** Add a native mesh workload with explicit fabric halo exchange and output topology.
10. **Out-of-core/multilevel.** Design explicit host streaming and multilevel result metadata; do not overload the single-level primitive.

### 17.3 Build-transition safeguards

**Recommendation.**

- Develop against the pinned new tt-metal revision, not the six-month-old submodule.
- Resolve the framework factory API before PR 3.
- Keep compatibility adapters isolated and temporary.
- Never modify TT-Metal firmware/SFPI semantics solely to make the port compile.
- Run standalone parity before and after every kernel move.
- Keep Wormhole and Blackhole CI as separate required jobs.
- Require a source/diff audit ensuring no unrelated operation or firmware change.

### 17.4 Stage-by-stage implementation handoff

| Stage | Files added/modified | Move versus wrap | Tests | Principal risk | Completion criterion |
|---|---|---|---|---|---|
| Baseline | standalone reports, generated manifest, CI script | no source move | current scripts plus ELF/perf capture | recording pre-existing failures as passes | reproducible artifact with exact SHAs and qualification labels |
| Common host code | `operations/wavelet/common`, `generated`, `planner`, family CMake | copy first; keep standalone source | generator/planner/boundary C++ unit tests | source drift between backends | identical registry and planner results for locked vectors |
| 1D primitives | two device-operation files, common program factory, 1D kernels | port kernel sources; wrap with TTNN tensors/cache | cache miss/hit, strict parity, WH/BH | 32-by-16 internal tile and newest descriptor API | TTNN primitive matches standalone outputs/latency gate |
| 1D public | `wavelet.hpp/.cpp`, nanobind, docs | wrapper only | C++/Python signatures/errors/examples | accidental API overbreadth | public forward/inverse pass locked matrix |
| 2D primitives | two device-operation files, 2D kernels/protocol | port existing five-plane scheduler unchanged | split, route, all modes, ELF, cache | Wormhole NCRISC overflow or split regression | fused interior verified and WH/BH parity passes |
| 2D public | public/nanobind/docs additions | wrapper only | tuple order, odd/non-square inverse, PyWavelets | logical/padded shape confusion | all public 2D contract tests pass |
| Promotion | registration/docs/CI policy | no algorithm move | 106-scheme JIT, qualification matrix, perf | high CI cost and high-order errors | owner accepts qualification and 5% gates |
| Batch/shard | new planner dimensions/factories | extend, do not rewrite | axes, zero-copy/reshard cost, cache | implicit channel semantics | explicit support matrix and measured win |
| Mesh spatial | native mesh workload/CCL/fabric files | new algorithmic distribution layer | 2-chip halos/topology/cache | incorrect dependency exchange | single-device parity for shard joins |
| Multilevel/out-of-core | public composite/streaming owner | new high-level layer | level shapes, lifetime, global edges | API/result-container lock-in | separately reviewed contract and performance |

## 18. Decisions Required From the Project Owner

Each question below materially changes API or implementation. The recommendation is a default, not a hidden assumption.

### Decision 1: Public namespace

**Open question.** Should the installed API be top-level `ttnn`, `ttnn::wavelet`, or `ttnn::experimental`?

- **Why it matters:** discoverability, promotion cost, and consistency with TTNN semantic operations.
- **Options:** top-level stable; nested stable family; experimental incubation.
- **Recommended default:** final names at top-level `ttnn`; internal implementation in `ttnn::prim`.
- **Consequences:** top-level commits to API stability; nesting adds verbosity; experimental lowers initial stability commitment but requires a promotion plan.

### Decision 2: Experimental versus stable first exposure

**Open question.** Should users see the API before full parity?

- **Why it matters:** upstream policy and compatibility expectations.
- **Options:** stable immediately; experimental until parity; internal primitive only.
- **Recommended default:** internal primitives first, then experimental only if upstream mandates it, then stable after gates.
- **Consequences:** immediate stable maximizes risk; experimental permits iteration; primitive-only delays user testing.

### Decision 3: Scheme argument representation

**Open question.** Should users pass a string, enum, object, numeric ID, coefficient tensors, or precompiled object?

- **Why it matters:** Python usability, ABI, cache keys, and specialization.
- **Options:** canonical string; public 106-value enum; value object; arbitrary coefficients; precompiled handle.
- **Recommended default:** canonical string mapped to private ID.
- **Consequences:** strings are extensible and friendly; enums freeze generated names into ABI; arbitrary coefficients and handles create a separate compilation/runtime contract.

### Decision 4: 1D forward output representation

**Open question.** Tuple or packed tensor?

- **Why it matters:** inverse ergonomics and tensor layout.
- **Options:** `(approximation, detail)`; named struct; packed tensor plus offsets.
- **Recommended default:** fixed tuple.
- **Consequences:** tuple matches TTNN multi-output and avoids packing; struct complicates binding; packed form adds an artificial materialization and metadata.

### Decision 5: 2D forward output representation

**Open question.** Four tensors or one packed tensor?

- **Why it matters:** LL/LH/HL/HH ownership, independent consumption, and mesh topology.
- **Options:** `(ll, lh, hl, hh)`; named struct; one packed tensor.
- **Recommended default:** fixed four-tensor tuple in that order.
- **Consequences:** tuple is explicit and unpackable; packed output can save allocations only with a new layout contract and may obstruct independent band placement.

### Decision 6: Inverse shape restoration

**Open question.** How should odd original shapes be recovered?

- **Why it matters:** coefficients do not always uniquely determine original extent.
- **Options:** explicit length/shape; result metadata object; inference; auxiliary shape tensor.
- **Recommended default:** mandatory host scalar `original_length` or `output_shape`.
- **Consequences:** explicit is unambiguous and cacheable; metadata object couples forward/inverse; inference can be wrong; a shape tensor adds device traffic for host-known data.

### Decision 7: Single-level versus multilevel first API

**Open question.** Should level recursion be public initially?

- **Why it matters:** result arity, shapes, storage lifetime, and cache behavior.
- **Options:** single level only; `levels=N` composite; packed multilevel coefficients.
- **Recommended default:** single-level only.
- **Consequences:** single-level preserves current proven semantics; immediate multilevel requires a result container and allocation strategy before primitive parity.

### Decision 8: Supported tensor ranks

**Open question.** Exact rank 1/2 or arbitrary leading dimensions?

- **Why it matters:** planner indexing and independent work units.
- **Options:** exact rank; rank `>=1/2` with flattened leading dims; rank-4 TTNN convention.
- **Recommended default:** rank 1 for 1D and rank 2 for 2D.
- **Consequences:** exact rank is safe and clear; arbitrary ranks need defined batch/channel semantics; forced rank 4 is unnatural for a general signal API.

### Decision 9: Batch semantics

**Open question.** When supported, which dimensions are independent batch items?

- **Why it matters:** output shapes and mesh data parallelism.
- **Options:** all leading dims; one explicit batch dim; separate batched API.
- **Recommended default:** later accept all leading dimensions as independent transforms, after exact-rank parity.
- **Consequences:** general leading dims are flexible but complicate scheduling; one batch dim is simpler; separate API fragments the surface.

### Decision 10: Channel semantics

**Open question.** Are channels transformed independently or included in spatial axes?

- **Why it matters:** image users may expect H/W-only transforms while tensor layout may be NCHW/NHWC.
- **Options:** no channel concept; explicit `axes`; fixed NCHW/NHWC convention.
- **Recommended default:** no channel semantics in first PR; later add explicit transform axes.
- **Consequences:** explicit axes generalize safely; implicit layout conventions are convenient but ambiguous and enlarge cache/planner scope.

### Decision 11: Input and output layouts

**Open question.** Should the first public wrapper convert layouts automatically?

- **Why it matters:** hidden data movement and performance predictability.
- **Options:** strict native layouts; automatic conversion; both with a flag.
- **Recommended default:** strict 1D row-major and 2D tile, output matching the native layout.
- **Consequences:** strict is transparent and fastest; automatic conversion is convenient but adds dispatch/DRAM traffic; a flag complicates the contract.

### Decision 12: Sharded tensors in the first PR

**Open question.** Accept interleaved DRAM only or also L1/sharded tensors?

- **Why it matters:** CB backing, zero-copy opportunities, halo ownership, and validation.
- **Options:** DRAM interleaved only; single-device sharded; all TTNN memory configs.
- **Recommended default:** DRAM interleaved only.
- **Consequences:** narrow scope maps directly to current behavior; sharding may improve pipelines but needs separate program factories; “all” is not credible without layout-specific tests.

### Decision 13: Multi-chip scope in the first PR

**Open question.** Should replicated or sharded MeshTensors be accepted?

- **Why it matters:** mesh launch does not define wavelet ownership.
- **Options:** reject multi-device; replicated only; batch-sharded; spatially sharded.
- **Recommended default:** reject multi-device.
- **Consequences:** rejection is correct and explicit; replication is a small later addition; batch requires rank semantics; spatial requires fabric halos.

### Decision 14: Preallocated output support

**Open question.** Expose optional outputs publicly now?

- **Why it matters:** pipeline memory control, aliasing, and binding tests.
- **Options:** allocate always; primitive-only preallocation; public optional outputs.
- **Recommended default:** implement in primitive and expose if standard upstream naming is approved.
- **Consequences:** allocation-only is simplest; primitive support protects architecture; public support improves reuse but expands validation.

### Decision 15: Low-level tuning configuration

**Open question.** Expose core count, chunk shape, reader compaction, or workspace layout?

- **Why it matters:** autotuning versus API stability and unsafe configurations.
- **Options:** expose all; expose an expert config; expose only compute-kernel config; expose none.
- **Recommended default:** expose only standard `DeviceComputeKernelConfig`; keep planner and architecture knobs internal with debug telemetry.
- **Consequences:** internal tuning permits evolution; expert knobs aid research but become compatibility obligations; exposing compaction can break Wormhole fit.

### Decision 16: Internal `ttwv` namespace

**Open question.** May it remain in private implementation code?

- **Why it matters:** migration size and public symbol hygiene.
- **Options:** rename everything immediately; retain private namespace; retain public namespace.
- **Recommended default:** allow `ttwv` temporarily only in non-installed/private code, then migrate to `ttnn::operations::wavelet`.
- **Consequences:** immediate rename creates noisy risk; private retention is harmless if no installed/API leakage; public retention violates the proposed TTNN contract.

### Decision 17: Numerical tolerance

**Open question.** What finite-input metric and threshold gates promotion?

- **Why it matters:** FP32 route length and architecture affect accumulated error.
- **Options:** one absolute threshold; per-scheme thresholds; combined absolute/relative/PCC policy.
- **Recommended default:** locked per-scheme absolute/relative tolerances derived from valid standalone and PyWavelets baselines, plus roundtrip thresholds.
- **Consequences:** one threshold may be too weak for easy schemes or impossible for unstable high-order cases; per-scheme policy is more work but honest.

### Decision 18: Known high-order FP32 differences

**Open question.** Are schemes such as coif17 allowed as compile/JIT-only support?

- **Why it matters:** “supports 106 schemes” can mean executable or numerically validated.
- **Options:** block release until all pass; mark defined schemes experimental/JIT-only; remove failing schemes.
- **Recommended default:** keep all 106 executable, publish a correctness qualification table, and block numerical claims for known failures until resolved.
- **Consequences:** blocking all delays integration; qualification is transparent; removal breaks required scheme coverage.

### Decision 19: First-PR performance threshold

**Open question.** What regression is acceptable versus standalone?

- **Why it matters:** TTNN wrapper/cache integration can add overhead to a latency-bound path.
- **Options:** zero regression; 5%; 10%; correctness-only first.
- **Recommended default:** at most 5% warm device-latency regression on representative db7 cases on both architectures, with cold JIT reported separately.
- **Consequences:** strict zero may be noisy; 5% catches meaningful loss; correctness-only risks institutionalizing a slow path.

### Decision 20: Ownership of generated 106-scheme artifacts

**Open question.** Which team owns generator inputs, generated headers, reviews, and future additions?

- **Why it matters:** generated numerical code needs stable review and CI responsibility.
- **Options:** wavelet op owner; TTNN core; an external repository/submodule; shared ownership.
- **Recommended default:** wavelet operation CODEOWNER owns generator and artifacts, with numerical-review approval for coefficient changes.
- **Consequences:** single ownership is actionable; TTNN core is too broad; external ownership complicates reproducible builds; shared ownership can stall reviews without explicit roles.

## 19. Assumptions Verified, Corrected, or Rejected

### Assumption 1: “N300 execution requires sending a program to another chip through NoC.”

**Verdict: incorrect.** N300 is a two-chip Wormhole card, but NoC is intra-chip. Host/mesh dispatch places programs on device ranges; inter-chip payload communication uses Ethernet/fabric. The corrected model separates host dispatch, local NoC, and explicit fabric/CCL.

### Assumption 2: “The same program automatically runs twice on two chips.”

**Verdict: partially correct.** An adapted mesh workload can stamp one single-device program over a mesh range (`tt-metal/tt_metal/api/tt-metalium/program_cache.hpp:70-82`), but operation semantics and tensor topology determine whether that is valid. Replicated tensors may justify identical work; spatially sharded tensors require different arguments and possibly different programs.

### Assumption 3: “A tensor can be divided into independent regions across two chips.”

**Verdict: partially correct.** Distributed tensors can be sharded, but wavelet spatial regions are not mathematically independent without dependency halos. Batch items are independent; 1D intervals and 2D rectangles need planner-derived halo exchange or duplication.

### Assumption 4: “The custom `ttwv` namespace cannot be used anywhere in a TTNN operation.”

**Verdict: incorrect.** It must not leak into installed public TTNN API, but private implementation namespaces do not affect user API. A staged internal rename is safe; final private code should follow upstream ownership conventions for maintainability.

### Assumption 5: “Updating to the latest tt-metal main is sufficient to understand current FFT behavior.”

**Verdict: partially correct.** Pinning/fetching current main is necessary for a current source conclusion. It is not sufficient without exact filename and whole-word searches plus device/kernel tracing. Those searches found no TTNN FFT operation at `bb9bb15`.

### Assumption 6: “Integration can safely happen before capturing a baseline.”

**Verdict: incorrect.** The submodule has diverged by thousands of commits, current high-order numerical behavior includes a known failure, and TTNN adds cache/layout/mesh variables. Without a locked standalone baseline, regressions cannot be attributed.

### Assumption 7: “All 106 schemes should necessarily be exposed as 106 public operations.”

**Verdict: incorrect.** They should be 106 validated data choices behind four semantic operations. Public-operation proliferation harms discoverability, bindings, ABI, and caching without changing semantics.

### Assumption 8: “A large input problem is equivalent to an L1-capacity problem.”

**Verdict: incorrect.** Per-core L1 working-set fit, device DRAM capacity, and host/out-of-core capacity are three different levels. The current dependency-cone planner substantially addresses the first and normal DRAM streaming addresses the second; the third requires an explicit external contract.

### Assumption 9: “Wormhole and Blackhole should use identical device source code.”

**Verdict: partially correct.** They should share mathematical and logical kernel sources, but compile-time architecture policies are necessary. Wormhole needs compact reader code and a 16-KiB NCRISC gate; Blackhole needs its tile-native performance path and different transfer assumptions.

### Assumption 10: “Conv is the best single template for the wavelet operation.”

**Verdict: incorrect.** No single operation covers the contract. Conv contributes composite/primitive layering and L1 modeling; TopK contributes fixed multi-output specs; pad contributes logical/padded validation; CCL contributes mesh topology; the device-operation framework contributes caching.

## 20. Final Recommendation

Proceed with integration against pinned tt-metal `bb9bb15`, but do it as a compatibility-preserving series rather than a source dump.

1. **Public API:** `ttnn::lwt`, `ttnn::ilwt`, `ttnn::lwt_2d`, and `ttnn::ilwt_2d`; strings map to private scheme IDs; forward returns fixed tuples; inverse accepts separate tensors and mandatory original/output shape.
2. **Namespace:** top-level `ttnn` semantics, `ttnn::prim` device operations, private implementation under `ttnn::operations::wavelet`; no public `ttwv`.
3. **Directory:** one owned `operations/wavelet` family with common planners, generated registry, four device operations, shared program factories/kernels, nanobind, tests, perf, docs, and generator tooling.
4. **Layering:** thin public normalization wrapper calling one fused primitive per direction/dimension. Do not expose or materialize split, lift, scale, route, or interleave stages.
5. **First-PR support:** single-level FP32, exact rank-1 row-major 1D and rank-2 tiled 2D, interleaved DRAM, one physical device, all eight modes, all 106 schemes executable, preallocated outputs in the primitive.
6. **Large inputs:** exact per-core L1 chunk planning; DRAM-resident chunk streaming when aggregate L1 is insufficient; explicit unsupported/error for beyond-DRAM tensors until a separate host-streaming design exists.
7. **Multi-chip:** reject first; add replicated and batch-sharded execution after batch semantics; implement spatial sharding only as a native mesh workload with explicit fabric halo traffic and topology.
8. **Specialization:** retain on-demand compile-time generated schemes and bounded canonical cache keys; evaluate hybrid runtime coefficients only with numerical, ELF, JIT, cache, and performance evidence.
9. **Architecture:** shared arithmetic plus explicit policy. Wormhole always compacts boundary/fallback reader code and gates NCRISC at `0x4000`; Blackhole preserves non-compact tile-native performance. Never noinline the fused interior split.
10. **Baseline:** capture standalone registry, correctness qualification, known failures, coif17 JIT-only status, WH/BH latency, planner/L1 telemetry, and Wormhole ELF sizes before moving kernels.
11. **PR sequence:** baseline; shared host code; 1D primitives; 1D API/bindings; 2D primitives; 2D API/bindings; promotion; batch/sharding; spatial mesh; multilevel/out-of-core.

### Top five technical risks

1. **Framework transition:** the pinned descriptor/spec/cache API is changing toward Metal 2.0; choosing a temporary binding path would create immediate technical debt.
2. **Wormhole numerical/size constraints:** HiFi4+FP32 accuracy warning and NCRISC `0x4000` fit must both be satisfied.
3. **Specialization explosion:** 106 schemes across four transforms and architecture/layout variants can create cold-JIT and binary-cache pressure.
4. **Shape/layout correctness:** logical versus padded shapes and odd inverse reconstruction can silently corrupt boundary semantics.
5. **False mesh generality:** accepting MeshTensors before ownership/halos/fabric are designed can return plausible but incorrect results.

### Top five unanswered owner decisions

1. final stable versus experimental exposure timing;
2. Wormhole HiFi3 versus HiFi4 numerical policy;
3. exact per-scheme numerical qualification, especially high-order schemes;
4. whether public preallocated outputs are in the first API;
5. long-term owner and review policy for generated scheme artifacts.

**Decisive conclusion.** Approve the four-API, four-primitive, single-device FP32 design and begin with the baseline PR. Do not begin kernel migration until the TTNN framework owner confirms the current descriptor/spec path and the project owner accepts the numerical qualification policy. Once those two gates are resolved, the existing fused algorithms can move without changing wavelet mathematics or sacrificing the Wormhole/Blackhole policies.

## Appendix A. TTNN Operation Inventory

The inventory is based on every immediate directory under `tt-metal/ttnn/cpp/ttnn/operations` at pinned revision `bb9bb15`. The stable CMake ownership list is at `tt-metal/ttnn/cpp/ttnn/operations/CMakeLists.txt:15-53`; several legacy umbrella families are registered elsewhere as that file notes at lines 8-13.

| Top-level family | Primary role/pattern observed | Relevance to wavelet |
|---|---|---|
| `bernoulli` | random element generation | cache-varying seed arguments are a dynamic-argument precedent |
| `ccl` | collective communication and topology | future multi-chip halos and native mesh workloads |
| `conv` | composite wrappers, fused primitive, slicing, architecture policy | public/primitive layering and L1 cost model |
| `copy` | copy compatibility/entry points | no reason to express wavelet as copies |
| `core` | layout/device/core utilities | `to_layout`, compute config, device semantics |
| `creation` | tensor creation | output allocation conventions |
| `data_movement` | pad, slice, transpose, concat, tilize, reshard, roll | logical/padded validation and factory selection |
| `debug` | debug/inspection operations | possible planner/cache telemetry only |
| `eltwise` | unary/binary/ternary elementwise families | scale precedent, but separate dispatch is rejected |
| `embedding` | indexed reads | irregular read/accessor examples |
| `embedding_backward` | indexed accumulation | no direct semantic reuse |
| `examples` | reference operation structure | useful only after checking production families |
| `experimental` | incubating owned operations | possible temporary exposure, not lower quality |
| `full` | filled-tensor creation | physical padding utility precedent |
| `full_like` | spec-derived creation | output-like allocation precedent |
| `generic` | generic device operation | rejected for wavelet because specialization matters |
| `index_fill` | index-based mutation | no public wavelet role |
| `kernel_helper_functions` | shared kernel helpers | possible low-level reuse, no public API |
| `kv_cache` | stateful/update operations | buffer lifetime and optional output precedents |
| `loss` | semantic composites/reductions | composite naming precedent |
| `matmul` | many factories, sharding, performance model | cache/factory complexity and architecture modeling |
| `moreh` | broad training/math operation set | optional outputs and multi-stage operations |
| `normalization` | fused reductions and optional statistics | best semantic analogy for one fused primitive |
| `point_to_point` | explicit device-to-device transfer | future direct halo movement, not first PR |
| `pool` | sliding-window reductions | bounded neighborhood and output shape precedents |
| `prefetcher` | DRAM/compute overlap | future chunk streaming, experimental dependency avoided initially |
| `rand` | random creation | dynamic cache/runtime seed precedent |
| `randn` | random normal creation | same |
| `reduction` | shape-changing and multi-output operations | TopK fixed-tuple model |
| `sliding_window` | geometry/configuration shared by conv/pool | conceptually similar halos, but wavelet planner is retained |
| `transformer` | large fused and distributed kernels | evidence that semantic fusion belongs in one family |
| `uniform` | random uniform fill | dynamic arguments, otherwise unrelated |

### A.1 Experimental subfamilies

**Fact.** The pinned experimental CMake registers adaptive pool, conv3d, CNN, CCL/MoE, matmul variants, reductions, indexer/top-k variants, paged cache, `isin`, minimal matmul, padded slice, slice write, copy, dropout, deformable attention, reshape, SSM, tensor prefetcher, and test operations (`tt-metal/ttnn/cpp/ttnn/operations/experimental/CMakeLists.txt:18-69`).

**Inference.** Experimental placement is determined by maturity/ownership, not by one implementation style. Wavelet should not copy an experimental prefetch or reshape contract into its first primitive merely because those features may be useful later.

### A.2 Deeply traced production patterns

| Pattern | Source | Design consequence |
|---|---|---|
| Composite 1D over fused 2D | `conv1d` reshapes/slices then calls `conv2d` | thin semantic normalization is acceptable |
| Primitive contract | `Conv2dDeviceOperation` | four wavelet primitives use attrs/args/specs/factory/validate |
| Fixed multi-output | `TopKDeviceOperation` | forward results are tuples of exact specs/tensors |
| Logical/padded output | `PadDeviceOperation` | extension and physical padding remain separate |
| Layout conversion | `to_layout` | conversion is explicit outside strict primitive |
| Reshard | `data_movement/sharded/reshard` | sharding support requires a measured separate path |
| Reduction/normalization | fused semantic device ops | internal wavelet stages are not public operations |
| Matmul factories | many factory variants and custom cache concerns | keep wavelet variants bounded and explicit |
| CCL/mesh | custom topologies and native cached mesh workloads | spatial wavelet needs operation-specific orchestration |
| Prefetcher | explicit prefetch program | out-of-core/overlap is a later feature |
| Nanobind | per-family binding files and named arguments | one wavelet binding file exposes four functions |
| Tests/perf | operation unit/perf trees | parity, cache, architecture, and latency suites are required |

## Appendix B. Source References

All references below are relative to parent revision `94ee3c0` and pinned submodule revision `bb9bb15`.

### B.1 Standalone wavelet

| Subject | Exact source |
|---|---|
| Four executable targets | `tt-wavelet/CMakeLists.txt:34-50` |
| 1D host executable/buffer API | `tt-wavelet/tt_wavelet/include/lifting/device.hpp:23-70`, `tt-wavelet/tt_wavelet/include/lifting/device.hpp:92-164` |
| 2D host executable/buffer API | `tt-wavelet/tt_wavelet/include/lifting/device_2d.hpp:27-79`, `tt-wavelet/tt_wavelet/include/lifting/device_2d.hpp:81-172` |
| Step representation | `tt-wavelet/tt_wavelet/include/lifting/step.hpp:7-13`, `tt-wavelet/tt_wavelet/include/lifting/static_scheme.hpp:12-44` |
| Registry and 106 schemes | `tt-wavelet/tt_wavelet/include/schemes/generated/registry.hpp:120-345` |
| db7 forward/inverse | `tt-wavelet/tt_wavelet/include/schemes/generated/db7.hpp:23-164` |
| Eight boundary modes | `tt-wavelet/tt_wavelet/include/common/boundary.hpp:15-42` |
| 2D logical/storage contract | `tt-wavelet/tt_wavelet/include/common/tiling_2d.hpp:17-109` |
| 1D forward plan | `tt-wavelet/tt_wavelet/include/lifting/plan.hpp:20-95`, `tt-wavelet/tt_wavelet/include/lifting/plan.hpp:224-290` |
| 1D inverse dependencies | `tt-wavelet/tt_wavelet/include/lifting/inverse_plan.hpp:108-253` |
| 2D route order/bands | `tt-wavelet/tt_wavelet/include/lifting/plan_2d.hpp:447-489` |
| 2D cones/resources/search | `tt-wavelet/tt_wavelet/include/lifting/plan_2d.hpp:512-680`, `tt-wavelet/tt_wavelet/include/lifting/plan_2d.hpp:851-1055` |
| Inverse 2D plan | `tt-wavelet/tt_wavelet/include/lifting/inverse_plan_2d.hpp:256-456` |
| 2D protocol/scratch/pages | `tt-wavelet/tt_wavelet/include/device_protocol/lwt_2d_config.hpp:11-101` |
| Architecture policy | `tt-wavelet/tt_wavelet/include/lifting/policy.hpp:12-45` |
| Compact selection | `tt-wavelet/tt_wavelet/src/lifting/device_2d.cpp:550-567` |
| Compact parity helper | `tt-wavelet/kernels/dataflow/lwt_2d_reader.cpp:360-487` |
| Fused interior split | `tt-wavelet/kernels/dataflow/lwt_2d_reader.cpp:508-588` |
| Interior/boundary dispatch | `tt-wavelet/kernels/dataflow/lwt_2d_reader.cpp:590-753` |
| Fused 2D compute/scales | `tt-wavelet/kernels/compute/lwt_2d_compute.cpp:19-211` |
| Full-tile route/band writes | `tt-wavelet/kernels/dataflow/lwt_2d_writer.cpp:46-75`, `tt-wavelet/kernels/dataflow/lwt_2d_writer.cpp:135-207` |
| Internal 1D narrow tiles | `tt-wavelet/tt_wavelet/src/lifting/device.cpp:418-502` |
| 2D validator matrix | `scripts/validate_lwt_2d_extension_modes.py:31-64`, `scripts/validate_lwt_2d_extension_modes.py:108-157` |
| Wormhole ELF gate | `scripts/check_ncrisc_elf_size.py:11-20`, `scripts/check_ncrisc_elf_size.py:75-125`, `scripts/check_ncrisc_elf_size.py:180-201` |

### B.2 Pinned TTNN/Metalium

| Subject | Exact source |
|---|---|
| Stable operation ownership/registration | `tt-metal/ttnn/cpp/ttnn/operations/CMakeLists.txt:1-53` |
| Experimental registration | `tt-metal/ttnn/cpp/ttnn/operations/experimental/CMakeLists.txt:1-75` |
| Device operation cached types | `tt-metal/ttnn/api/ttnn/device_operation.hpp:32-49` |
| Cache hit/miss dispatch | `tt-metal/ttnn/api/ttnn/device_operation.hpp:254-329` |
| Device launch contract | `tt-metal/ttnn/api/ttnn/device_operation.hpp:413-502` |
| Canonical program key | `tt-metal/tt_metal/api/tt-metalium/program_cache.hpp:107-135` |
| Cached program/workload types | `tt-metal/tt_metal/api/tt-metalium/program_cache.hpp:18-105` |
| Workload-owned resources | `tt-metal/tt_metal/api/tt-metalium/workload_descriptor.hpp:16-68` |
| Temporary descriptor patch warning | `tt-metal/tt_metal/api/tt-metalium/experimental/program_descriptor_patching.hpp:7-19` |
| Emerging ProgramSpec adapter | `tt-metal/ttnn/api/ttnn/mesh_device_operation_adapter.hpp:760-780` |
| Conv1d composition/slicing | `tt-metal/ttnn/cpp/ttnn/operations/conv/conv1d/conv1d.cpp:38-123` |
| Conv2d primitive/L1 model | `tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_device_operation.hpp:25-72` |
| Conv architecture transfer policy | `tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:41-49` |
| TopK multi-output contract | `tt-metal/ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.hpp:16-58` |
| TopK validate/spec/allocate | `tt-metal/ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp:115-317` |
| Pad factory/validation | `tt-metal/ttnn/cpp/ttnn/operations/data_movement/pad/device/pad_device_operation.cpp:70-211` |
| Pad spec/allocation | `tt-metal/ttnn/cpp/ttnn/operations/data_movement/pad/device/pad_device_operation.cpp:213-271` |
| To-layout composite | `tt-metal/ttnn/cpp/ttnn/operations/core/to_layout/to_layout_op.cpp:25-220` |
| All-gather device topology | `tt-metal/ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_device_operation.hpp:16-35` |
| CCL architecture values | `tt-metal/ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:39-70`, `tt-metal/ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:2035-2058` |
| TensorSpec | `tt-metal/tt_metal/api/tt-metalium/experimental/tensor/spec/tensor_spec.hpp:19-109` |
| Tensor properties | `tt-metal/ttnn/api/ttnn/tensor/tensor.hpp:187-203` |
| Memory configuration | `tt-metal/tt_metal/api/tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp:25-47` |
| Mesh tensor placement | `tt-metal/tt_metal/api/tt-metalium/experimental/distributed_tensor/topology/distributed_tensor_configs.hpp:15-85` |
| MeshBuffer placement | `tt-metal/tt_metal/api/tt-metalium/mesh_buffer.hpp:39-73` |
| N150/N300 device counts | `tt-metal/tech_reports/FlashAttention/FlashDecode.md:163-171` |
| N300 physical/dispatch topology | `tt-metal/tech_reports/EthernetMultichip/BasicEthernetGuide.md:153-159` |
| N300 dispatch | `tt-metal/ttnn/core/device.cpp:17-23` |
| N300 chip topology example | `tt-metal/tt_metal/llrt/tt_cluster.hpp:484-492` |
| Fabric versus NoC | `tt-metal/tech_reports/TT-Fabric/TT-Fabric-Architecture.md:204-209`, `tt-metal/tech_reports/TT-Fabric/TT-Fabric-Architecture.md:822-845` |
| Wormhole FP32 fidelity warning | `tt-metal/ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.cpp:43-67` |
| FFT negative search evidence | no operation source; model-only example at `tt-metal/models/experimental/detr3d/ttnn/position_embedding.py:16-28` |

## Appendix C. Proposed API Signatures

### C.1 Public C++ declarations

```cpp
namespace ttnn {

enum class WaveletBoundaryMode : std::uint8_t {
    Zero,
    Constant,
    Symmetric,
    Reflect,
    Periodic,
    Smooth,
    Antisymmetric,
    Antireflect,
};

using LwtResult = std::tuple<Tensor, Tensor>;
using Lwt2DResult = std::tuple<Tensor, Tensor, Tensor, Tensor>;

LwtResult lwt(
    const Tensor& input,
    std::string_view wavelet,
    WaveletBoundaryMode mode = WaveletBoundaryMode::Symmetric,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<LwtResult>& output_tensors = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

Tensor ilwt(
    const Tensor& approximation,
    const Tensor& detail,
    std::string_view wavelet,
    std::uint32_t original_length,
    WaveletBoundaryMode mode = WaveletBoundaryMode::Symmetric,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

Lwt2DResult lwt_2d(
    const Tensor& input,
    std::string_view wavelet,
    WaveletBoundaryMode mode = WaveletBoundaryMode::Symmetric,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Lwt2DResult>& output_tensors = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

Tensor ilwt_2d(
    const Tensor& ll,
    const Tensor& lh,
    const Tensor& hl,
    const Tensor& hh,
    std::string_view wavelet,
    std::array<std::uint32_t, 2> output_shape,
    WaveletBoundaryMode mode = WaveletBoundaryMode::Symmetric,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn
```

### C.2 Primitive operation types

The exact descriptor return type is intentionally written as `ApprovedProgramArtifacts`, pending the framework-owner decision identified above.

```cpp
namespace ttnn::prim {

struct WaveletOperationAttributes {
    WaveletSchemeId scheme;
    WaveletBoundaryMode boundary;
    tt::tt_metal::MemoryConfig output_memory;
    DeviceComputeKernelConfig compute_config;
    std::optional<CoreRangeSet> sub_core_grid;  // internal/defaulted first PR
};

struct InverseShape1D {
    std::uint32_t original_length;
};

struct InverseShape2D {
    std::uint32_t output_height;
    std::uint32_t output_width;
};

struct LwtTensorArgs {
    Tensor input;
    std::optional<std::tuple<Tensor, Tensor>> preallocated_outputs;
};

struct IlwtTensorArgs {
    Tensor approximation;
    Tensor detail;
    std::optional<Tensor> preallocated_output;
};

struct Lwt2DTensorArgs {
    Tensor input;
    std::optional<std::tuple<Tensor, Tensor, Tensor, Tensor>> preallocated_outputs;
};

struct Ilwt2DTensorArgs {
    Tensor ll;
    Tensor lh;
    Tensor hl;
    Tensor hh;
    std::optional<Tensor> preallocated_output;
};

struct LwtDeviceOperation {
    using operation_attributes_t = WaveletOperationAttributes;
    using tensor_args_t = LwtTensorArgs;
    using spec_return_value_t = std::tuple<TensorSpec, TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;
    using program_factory_t = /* approved factory variant */;

    static program_factory_t select_program_factory(
        const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(
        const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&, const tensor_args_t&);
};

struct IlwtDeviceOperation {
    using operation_attributes_t =
        std::pair<WaveletOperationAttributes, InverseShape1D>;
    using tensor_args_t = IlwtTensorArgs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    // same required device-operation methods
};

struct Lwt2DDeviceOperation {
    using operation_attributes_t = WaveletOperationAttributes;
    using tensor_args_t = Lwt2DTensorArgs;
    using spec_return_value_t =
        std::tuple<TensorSpec, TensorSpec, TensorSpec, TensorSpec>;
    using tensor_return_value_t =
        std::tuple<Tensor, Tensor, Tensor, Tensor>;
    // same required device-operation methods
};

struct Ilwt2DDeviceOperation {
    using operation_attributes_t =
        std::pair<WaveletOperationAttributes, InverseShape2D>;
    using tensor_args_t = Ilwt2DTensorArgs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    // same required device-operation methods
};

}  // namespace ttnn::prim
```

**Recommendation.** In implementation, use named attribute structs rather than the illustrative `std::pair` aliases above so reflection produces stable readable cache keys.

### C.3 Compile-time and runtime placement

| Placement | Exact content |
|---|---|
| Operation attributes | scheme ID, boundary mode, inverse output shape, output memory config, compute config, supported core-grid policy |
| Tensor arguments | input tensors and optional preallocated output tensors |
| Factory selection | architecture policy, compact flag, 1D inverse workspace layout, supported input layout/memory |
| Compile defines/template types | generated scheme header/type, direction, dimension, boundary specialization, compact flag, layout policy, architecture-required constants |
| Compile-time kernel args | CB IDs/page sizes, static route count where compiled, protocol page size, accessor compile args, feature flags |
| Common runtime args | input/output/metadata buffer bindings, logical/padded dimensions, page strides |
| Per-core runtime args | core chunk begin/count, chunk/route/band descriptor offsets, local plane offsets, final output rectangle, synchronization IDs |
| Excluded dynamic values | current tensor addresses and operation-owned buffer addresses |
| Conservative initial cache key | all operation attrs plus logical/padded specs, selected factory/policy, core/chunk geometry, architecture |

### C.4 Registered primitive functions

```cpp
namespace ttnn::prim {

std::tuple<Tensor, Tensor> lwt(
    const Tensor& input,
    WaveletSchemeId scheme,
    WaveletBoundaryMode boundary,
    const MemoryConfig& output_memory,
    const DeviceComputeKernelConfig& compute_config,
    const std::optional<std::tuple<Tensor, Tensor>>& outputs = std::nullopt);

Tensor ilwt(
    const Tensor& approximation,
    const Tensor& detail,
    WaveletSchemeId scheme,
    std::uint32_t original_length,
    WaveletBoundaryMode boundary,
    const MemoryConfig& output_memory,
    const DeviceComputeKernelConfig& compute_config,
    const std::optional<Tensor>& output = std::nullopt);

std::tuple<Tensor, Tensor, Tensor, Tensor> lwt_2d(
    const Tensor& input,
    WaveletSchemeId scheme,
    WaveletBoundaryMode boundary,
    const MemoryConfig& output_memory,
    const DeviceComputeKernelConfig& compute_config,
    const std::optional<std::tuple<Tensor, Tensor, Tensor, Tensor>>& outputs =
        std::nullopt);

Tensor ilwt_2d(
    const Tensor& ll,
    const Tensor& lh,
    const Tensor& hl,
    const Tensor& hh,
    WaveletSchemeId scheme,
    std::array<std::uint32_t, 2> output_shape,
    WaveletBoundaryMode boundary,
    const MemoryConfig& output_memory,
    const DeviceComputeKernelConfig& compute_config,
    const std::optional<Tensor>& output = std::nullopt);

}  // namespace ttnn::prim
```

### C.5 Nanobind registration shape

```cpp
void bind_wavelet_operations(nb::module_& module) {
    bind_registered_operation(
        module,
        ttnn::lwt,
        "input"_a,
        "wavelet"_a,
        "mode"_a = "symmetric",
        "memory_config"_a = nb::none(),
        "output_tensors"_a = nb::none(),
        "compute_kernel_config"_a = nb::none());

    bind_registered_operation(
        module,
        ttnn::ilwt,
        "approximation"_a,
        "detail"_a,
        "wavelet"_a,
        "original_length"_a,
        "mode"_a = "symmetric",
        "memory_config"_a = nb::none(),
        "output_tensor"_a = nb::none(),
        "compute_kernel_config"_a = nb::none());

    // lwt_2d and ilwt_2d follow the same named-argument order.
}
```

Binding code should use the exact helper current at implementation time; this snippet specifies names and defaults, not a commitment to a helper that may change during the Metal 2.0 transition.

### C.6 Python type signatures

```python
def lwt(
    input: ttnn.Tensor,
    wavelet: str,
    mode: str = "symmetric",
    *,
    memory_config: ttnn.MemoryConfig | None = None,
    output_tensors: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]: ...

def ilwt(
    approximation: ttnn.Tensor,
    detail: ttnn.Tensor,
    wavelet: str,
    original_length: int,
    mode: str = "symmetric",
    *,
    memory_config: ttnn.MemoryConfig | None = None,
    output_tensor: ttnn.Tensor | None = None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
) -> ttnn.Tensor: ...

def lwt_2d(
    input: ttnn.Tensor,
    wavelet: str,
    mode: str = "symmetric",
    *,
    memory_config: ttnn.MemoryConfig | None = None,
    output_tensors: tuple[
        ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor
    ] | None = None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]: ...

def ilwt_2d(
    ll: ttnn.Tensor,
    lh: ttnn.Tensor,
    hl: ttnn.Tensor,
    hh: ttnn.Tensor,
    wavelet: str,
    output_shape: tuple[int, int],
    mode: str = "symmetric",
    *,
    memory_config: ttnn.MemoryConfig | None = None,
    output_tensor: ttnn.Tensor | None = None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
) -> ttnn.Tensor: ...
```

## Appendix D. Support Matrices

### D.1 First-PR operation matrix

| Operation | Input layout | Memory layout | Buffer | Output layout | Output memory | First PR |
|---|---|---|---|---|---|---|
| `lwt` | row-major | interleaved | DRAM | row-major | interleaved DRAM | yes |
| `ilwt` | row-major | interleaved | DRAM | row-major | interleaved DRAM | yes |
| `lwt_2d` | tile | interleaved | DRAM | tile | interleaved DRAM | yes |
| `ilwt_2d` | tile | interleaved | DRAM | tile | interleaved DRAM | yes |
| any op | tile for 1D | any | any | any | any | no |
| any op | row-major for 2D | any | any | any | any | no |
| any op | any | height-sharded | L1/DRAM | any | any | no |
| any op | any | width-sharded | L1/DRAM | any | any | no |
| any op | any | block-sharded | L1/DRAM | any | any | no |
| any op | any | interleaved | L1 | any | any | no |

### D.2 Shape and semantic matrix

| Capability | First PR | Later |
|---|---|---|
| Single-level | yes | remains primitive |
| Multilevel | no | public composite/result descriptor |
| 1D rank 1 | yes | yes |
| 1D leading batch dims | no | yes after axis contract |
| 2D rank 2 | yes | yes |
| 2D leading batch/channel dims | no | yes after axis contract |
| Odd dimensions | yes, explicit inverse shape | yes |
| Empty dimensions | reject | reject |
| FP32 | yes | yes |
| BF16/BFP/int | reject | measured future work only |
| All 106 names | executable | qualification grows |
| All eight modes | yes | yes |
| Custom coefficient tensors | no | possible separate API |
| Preallocated outputs | primitive yes | public based on owner decision |
| Input/output alias | reject | only after proof |

### D.3 Architecture matrix

| Capability | Wormhole B0 | Blackhole |
|---|---|---|
| Public semantics | identical | identical |
| 1D forward | supported | supported |
| 1D inverse | row-major policy | tile-native policy |
| 2D forward/inverse | supported | supported |
| Compact 2D boundary code | always | routes `>=52` or antireflect |
| Fused interior split | required | required |
| Direct final inverse interleave | no/default existing path | yes when tile-native |
| NCRISC gate | `<=0x4000` executable segment | not applicable |
| FP32 fidelity | HiFi3/HiFi4 owner gate | current validated default |
| Performance regression gate | `<=5%` recommended | `<=5%` recommended |

### D.4 Large-input matrix

| Situation | 1D LWT/ILWT | 2D LWT/ILWT | First PR |
|---|---|---|---|
| Working set exceeds one core L1 | smaller dependency-cone chunks | smaller rectangle chunks | yes |
| Tensor exceeds aggregate L1 but fits DRAM | DRAM interval streaming | DRAM rectangle streaming | yes |
| Partial final chunk | exact logical crop/padding | exact logical rectangle/padding | yes |
| Tensor exceeds device DRAM | explicit unsupported/allocation error | same | yes, error only |
| Host out-of-core stream | future interval windows | future rectangle windows | no |
| Multilevel DRAM pressure | future level scheduling | future band/level scheduling | no |

### D.5 Mesh matrix

| Placement/mode | Communication | First PR | Recommended order |
|---|---|---|---|
| One physical device | chip-local NoC | yes | 1 |
| Replicated mesh tensor | none between transforms | reject initially | 2 |
| Batch-sharded mesh tensor | none wavelet-specific | reject initially | 3 |
| 1D spatial shard | left/right fabric halos | reject | 4 |
| 2D spatial shard | row/column/corner fabric halos | reject | 5 |
| Out-of-core across devices | host/fabric orchestration | reject | after spatial design |

### D.6 Specialization matrix

| Axis | Compile-time first | Runtime | Cache-key treatment |
|---|---|---|---|
| Scheme | generated type | public string normalized to ID | ID included |
| Direction | separate primitive/kernel | none | operation identity |
| Dimension | separate primitive/kernel | none | operation identity |
| Step coefficients | `coeff_bits` | none | implied by scheme |
| Boundary | specialized reader where current | mode argument | included if code differs |
| Compact reader | compile define | policy-derived | included |
| Architecture | JIT target/policy | device-derived | device-scoped/included |
| Logical shape | planner-specific initially | runtime args also carry it | included initially |
| Addresses | no | bindings | excluded |
| Chunk offsets | no | per-core args/metadata | excluded only when geometry key remains compatible |

### D.7 Validation matrix

| Check | Forward 1D | Inverse 1D | Forward 2D | Inverse 2D |
|---|---:|---:|---:|---:|
| allocated device tensor | yes | yes, both | yes | yes, all four |
| same device/topology | n/a | yes | n/a | yes |
| FP32 | yes | yes | yes | yes |
| exact rank | 1 | 1 | 2 | 2 |
| native layout | row-major | row-major | tile | tile |
| interleaved DRAM | yes | yes | yes | yes |
| positive logical shape | yes | yes | yes | yes |
| scheme/mode valid | yes | yes | yes | yes |
| coefficient shape compatible | output | yes | output | yes |
| explicit inverse shape | n/a | length | n/a | height/width |
| output spec/preallocation | two | one | four | one |
| L1/CB fit | yes | yes | yes | yes |
| architecture supported | yes | yes | yes | yes |
| no alias | yes | yes | yes | yes |
| checked byte/index arithmetic | yes | yes | yes | yes |
