You are working as a senior Tenstorrent/TTNN systems engineer and technical design reviewer.

Your task is to research the latest `tt-metal`/TTNN source tree and produce a detailed integration design for moving the existing `tt-wavelet` implementation into upstream-quality TTNN operations.

## Primary objective

Design how the following existing operations should become TTNN operations:

* 1D forward Lifting Wavelet Transform: `lwt`
* 1D inverse Lifting Wavelet Transform: `ilwt`
* 2D forward Lifting Wavelet Transform: `lwt_2d`
* 2D inverse Lifting Wavelet Transform: `ilwt_2d`

The output of this task is a technical design document, not an implementation.

Create the following file:

```text
docs/TTNN_WAVELET_INTEGRATION_DESIGN.md
```

The document must be written in English.

## Non-negotiable constraints

1. Do not replace, simplify, or redesign the mathematical LWT/ILWT algorithm.
2. Preserve the existing device-side execution strategy unless a change is strictly required by a TTNN contract.
3. Preserve all currently supported 106/106 lifting schemes.
4. Preserve both forward and inverse transforms.
5. Preserve both 1D and 2D transforms.
6. Preserve all currently supported boundary-extension modes:

   * `zero`
   * `constant`
   * `symmetric`
   * `reflect`
   * `periodic`
   * `smooth`
   * `antisymmetric`
   * `antireflect`
7. The integration must support both:

   * Wormhole
   * Blackhole
8. FP32 correctness must remain the primary correctness contract.
9. This task is primarily about:

   * TTNN API design;
   * operation contracts;
   * memory behavior;
   * large-input execution;
   * device and mesh behavior;
   * namespace and source-tree organization;
   * architecture compatibility;
   * validation;
   * testing;
   * upstream PR readiness.
10. Do not implement the TTNN operations in this task.
11. Do not refactor the current kernels or planner in this task.
12. Apart from updating the `tt-metal` submodule pointer and creating the requested Markdown document, do not modify source code.

## Repository safety rules

Before making any change:

```bash
git status --short
git submodule status
git -C tt-metal status --short
git -C tt-metal rev-parse HEAD
```

Record:

* the current parent repository commit;
* the current `tt-metal` submodule commit;
* whether either working tree is dirty.

Do not overwrite, discard, reset, or modify existing uncommitted work.

If the `tt-metal` submodule working tree is clean, update it to the newest commit from the upstream `main` branch:

```bash
git -C tt-metal fetch origin main
git -C tt-metal checkout --detach origin/main
```

After updating, record:

```bash
git -C tt-metal rev-parse HEAD
git -C tt-metal log -1 --oneline
git diff --submodule=log
```

The design document must state both the old and new `tt-metal` commit SHAs.

If the submodule contains uncommitted changes, do not reset or overwrite them. Continue the analysis using the existing checkout and document:

* why the update was not performed;
* the current SHA;
* the latest fetched `origin/main` SHA;
* the exact safe command sequence that should be run after the local changes are preserved.

## Required research approach

Do not inspect only one operation and generalize from it.

You must inspect the complete organization of:

```text
ttnn/cpp/ttnn/operations/
```

First create an inventory of all top-level operation families in that directory.

Then select and deeply inspect representative operations covering different TTNN design patterns.

At minimum, deeply inspect the following categories:

1. Convolution:

   ```text
   ttnn/cpp/ttnn/operations/conv/conv1d/
   ttnn/cpp/ttnn/operations/conv/conv2d/
   ttnn/cpp/ttnn/operations/conv/conv2d/device/
   ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/
   ```

2. FFT:

   * Locate every FFT-related implementation in the current source tree.
   * Do not assume its path.
   * Search for:

     ```bash
     rg -n --hidden --glob '!build/**' \
       'fft|FFT|ifft|IFFT|Fourier' \
       ttnn tt_metal tests
     ```
   * Determine whether FFT is:

     * public or experimental;
     * a composite operation or device operation;
     * single-device or mesh-aware;
     * implemented using primitive TTNN operations or custom kernels;
     * restricted by dtype, layout, shape, rank, or architecture.

3. Matrix multiplication:

   * Inspect how a major compute operation separates:

     * public API;
     * operation attributes;
     * tensor arguments;
     * validation;
     * output specification;
     * program factory;
     * program cache;
     * architecture-specific compute configuration.

4. Data movement operations:

   * `pad`
   * `slice`
   * `reshape`
   * `transpose`
   * `concat`
   * `to_layout`
   * sharding/res harding operations
   * operations that stream or slice data because the full working set does not fit in L1.

5. Reduction or normalization operations:

   * Inspect at least one operation that returns multiple outputs or has nontrivial output-shape calculation.

6. Collective communication or multi-device operations:

   * Locate CCL, distributed, mesh, all-gather, reduce-scatter, or related operations.
   * Determine how public TTNN operations execute on `MeshDevice`.
   * Determine how distributed tensors describe replication and sharding.

7. Experimental operations:

   * Determine when new operations are placed under an experimental namespace or directory.
   * Determine the criteria for promotion to the main public TTNN API.

8. Operations with architecture-specific behavior:

   * Find examples that support both Wormhole and Blackhole.
   * Find examples that:

     * select different kernels;
     * change compile-time arguments;
     * change compute configuration;
     * use `hal::get_arch()`;
     * use `DeviceComputeKernelConfig`;
     * use architecture macros inside kernels;
     * reject unsupported architectures.

For every significant conclusion, cite the exact source path and relevant line range in the Markdown document.

Distinguish conclusions using these labels:

* **Fact** — directly supported by source code or documentation.
* **Inference** — your technical interpretation.
* **Recommendation** — the proposed design for `tt-wavelet`.
* **Open question** — a decision that must be made by the project owner.

Do not state uncertain behavior as fact.

## Existing `tt-wavelet` analysis

Before proposing a TTNN structure, inspect the current `tt-wavelet` repository completely enough to understand:

* public host API;
* planners;
* execution plans;
* scheme representation;
* generated scheme headers;
* use of compile-time scheme types;
* kernel compilation model;
* 1D reader, writer, and compute kernels;
* 2D reader, writer, and compute kernels;
* L1 working-set model;
* DRAM buffers;
* chunk planning;
* halo/dependency-cone planning;
* logical and padded shapes;
* row-major and tile layouts;
* output layout;
* boundary extension;
* MeshBuffer or distributed APIs currently used;
* Wormhole/Blackhole-specific code;
* program caching or JIT behavior;
* current tests;
* current benchmarks;
* current Python exposure, if any.

Pay particular attention to:

```text
tt_wavelet/include/
kernels/
kernels/compute/
kernels/dataflow/
kernels/sfpi/
tests/
generated/
```

Also search for:

```bash
rg -n \
  'namespace ttwv|ARCH_WORMHOLE|ARCH_BLACKHOLE|hal::get_arch|MeshDevice|MeshBuffer|CreateProgram|CreateKernel|CreateCircularBuffer|SetRuntimeArgs|override_runtime_arguments|program_cache|scheme|boundary|padding|shard|chunk|halo' \
  .
```

The design document must include a concise architecture diagram of the current implementation before describing the TTNN target structure.

## Questions that must be answered

### 1. Large inputs and inputs that do not fit

The phrase “does not fit” is ambiguous. Analyze it at three separate levels:

#### A. Input does not fit in one core’s L1

Explain:

* how TTNN operations normally stream blocks from DRAM;
* how circular buffers limit the active working set;
* how program factories calculate per-core CB memory;
* how operations reject configurations that exceed available L1;
* how the existing `tt-wavelet` chunk/dependency-cone approach maps to this model;
* whether the current algorithm already solves this case.

#### B. Full tensor fits in device DRAM but not in aggregate L1

Explain:

* whether the tensor should remain interleaved in DRAM;
* when an input should be sharded to L1;
* whether the op should support both interleaved and sharded inputs;
* whether the first TTNN version should accept DRAM tensors only;
* whether outputs should be DRAM-interleaved by default;
* how chunk scheduling should work;
* how partial final chunks and padding should be handled.

#### C. Full tensor does not fit in device DRAM

Determine whether TTNN operations provide a standard out-of-core tensor contract.

If not, state that clearly.

Propose whether this should be handled by:

* application-level streaming;
* a future sliced API;
* batched windows;
* host orchestration;
* multi-device sharding;
* an explicit unsupported-input error in the first PR.

Do not silently treat an out-of-core problem as an L1 problem.

Provide separate recommendations for:

* 1D LWT/ILWT;
* 2D LWT/ILWT;
* single-level transforms;
* future multilevel transforms.

### 2. Multi-chip behavior

Investigate actual current TTNN and Metalium behavior rather than relying on assumptions.

Specifically verify or correct the following assumptions:

* Wormhole N150 contains one chip.
* Wormhole N300 contains two chips.
* A program must be manually sent from one chip to another.
* Inter-chip communication uses the same mechanism as ordinary intra-chip NoC traffic.
* Launching one operation on a mesh automatically duplicates the same work on all chips.
* A tensor can instead be partitioned into different spatial regions across chips.

Clearly distinguish:

* on-chip NoC communication;
* inter-chip Ethernet/fabric communication;
* host dispatch to multiple chips;
* replicated tensors;
* sharded/distributed tensors;
* data-parallel execution;
* tensor-parallel or spatially partitioned execution.

Determine how a registered TTNN operation receives a tensor placed on a `MeshDevice`.

Answer:

1. Does TTNN invoke the same operation independently on every device shard?
2. How does the operation obtain the local tensor shard?
3. Is mesh orchestration automatic or operation-specific?
4. How are output distributed tensor specifications created?
5. What program caching occurs per device or per architecture?
6. Can a single logical operation launch different runtime arguments on different chips?
7. Which existing operations partition work across devices rather than merely replicate it?
8. Does an operation need explicit fabric kernels for cross-chip communication?
9. What is the correct mechanism for exchanging wavelet halos across chip boundaries?
10. Can halos be duplicated before execution to avoid runtime chip-to-chip communication?

For `tt-wavelet`, analyze at least these execution modes:

#### Mode A: Replicated execution

Each chip receives a complete independent input and performs the same operation.

Use cases:

* independent batch elements;
* throughput benchmarking;
* data-parallel inference.

#### Mode B: Batch sharding

Different batch elements are assigned to different chips.

Determine whether this can be supported without any wavelet-specific cross-chip communication.

#### Mode C: 1D spatial sharding

Different intervals of one signal are assigned to different chips.

Analyze:

* boundary halos;
* lifting-step dependency growth;
* whether a one-time expanded input interval is sufficient;
* whether halos can be precomputed by the host planner;
* whether inter-chip communication is needed during execution;
* inverse-transform implications.

#### Mode D: 2D spatial sharding

Different rectangles of one image or matrix are assigned to different chips.

Analyze:

* horizontal and vertical halos;
* corner halos;
* vertical-first versus horizontal-first execution;
* LL/LH/HL/HH output ownership;
* whether each chip can own an independent dependency-expanded rectangle;
* whether intermediate cross-chip exchange is required.

Provide a staged recommendation:

* what the first upstream PR should support;
* what should explicitly be rejected;
* what should be deferred to a later multi-chip PR.

### 3. TTNN operation source style and namespaces

Inspect current source style across multiple operations.

Answer:

* where public operation declarations live;
* where public operation definitions live;
* where nanobind bindings live;
* where primitive/device operations live;
* where program factories live;
* where device kernels live;
* where common utility types live;
* how public operations are registered;
* how Python names are exposed;
* how docs are attached to registered operations;
* how operation names are organized.

Determine the preferred namespace structure for wavelet operations.

Evaluate at least these options:

```cpp
namespace ttnn::operations::wavelet
namespace ttnn::operations::signal_processing::wavelet
namespace ttnn::operations::experimental::wavelet
namespace ttnn::prim
```

Determine where the current custom namespace:

```cpp
namespace ttwv
```

may remain appropriate.

Possible distinction to evaluate:

* public TTNN wrapper in `ttnn::operations::wavelet`;
* device primitive in `ttnn::prim`;
* private scheme/kernel implementation helpers in an internal namespace;
* removal of `ttwv` from the public API surface.

Do not assume that the Conv namespace layout is universally correct. Compare it with several operations.

Propose exact namespaces for:

* public operation structs;
* registered operation constants;
* configuration structs;
* scheme identifiers;
* device-operation attributes;
* program factories;
* kernel-only helper types;
* nanobind functions.

### 4. Wormhole and Blackhole differences

The current implementation includes architecture-sensitive kernel code.

Investigate how upstream TTNN operations support multiple architectures.

Answer:

* when one common kernel source is used;
* when separate architecture-specific kernel files are used;
* how `ARCH_WORMHOLE` and `ARCH_BLACKHOLE` are supplied;
* whether direct LLK architecture headers should be included by an upstream TTNN kernel;
* whether public compute-kernel APIs can remove the need for custom architecture-header forwarding;
* how `hal::get_arch()` is used on the host;
* how `DeviceComputeKernelConfig` is initialized;
* whether Wormhole and Blackhole need different:

  * SFPI code;
  * tile register handling;
  * CB sizes;
  * kernel compile definitions;
  * math fidelity settings;
  * `fp32_dest_acc_en`;
  * runtime-argument limits;
  * NoC/fabric behavior.

Inspect the existing `tt-wavelet` architecture-selection code and determine whether it is upstream-compatible.

Create a table with columns:

| Concern | Current tt-wavelet behavior | Wormhole requirement | Blackhole requirement | Proposed TTNN handling |

The recommendation should minimize architecture-specific code while preserving correctness and performance.

### 5. Additional design topics that may have been missed

Identify all other major design decisions required before an upstream-quality TTNN PR.

At minimum, investigate:

* public API naming;
* operation granularity;
* single-level versus multilevel API;
* forward output representation;
* inverse input representation;
* separate coefficient tensors versus one packed tensor;
* LL/LH/HL/HH tuple versus packed 2D output;
* logical shape versus padded shape;
* odd signal lengths;
* odd image height and width;
* batch dimensions;
* channel dimensions;
* supported tensor ranks;
* supported dtypes;
* supported tensor layouts;
* supported memory layouts;
* supported buffer types;
* output memory configuration;
* output layout configuration;
* preallocated output tensors;
* asynchronous execution;
* queue ID support;
* tensor ownership and lifetime;
* device consistency;
* mesh consistency;
* aliasing restrictions;
* in-place execution;
* output shape inference;
* program-cache keys;
* runtime-argument override;
* compile-time argument selection;
* generated scheme-header packaging;
* 106-scheme kernel compilation cost;
* binary size;
* JIT compilation latency;
* persistent kernel cache;
* thread safety;
* deterministic execution;
* error messages;
* `TT_FATAL` versus validation errors;
* special-value behavior;
* NaN/Inf policy;
* zero-sized tensors;
* integer overflow in shape and byte calculations;
* huge tensors requiring 64-bit indexing;
* profiler integration;
* Tracy zones or TTNN profiling hooks;
* documentation;
* examples;
* code ownership;
* CI duration;
* benchmark methodology;
* backward compatibility.

For each topic, classify it as:

* must be decided before implementation;
* can be decided during implementation;
* can be deferred after the first PR.

### 6. Pre-integration safeguards for `tt-wavelet`

Evaluate what should be validated before changing the operation into TTNN form.

The project owner currently prefers to integrate first and then add more tests and fixes. Analyze the risks of this approach.

Propose a minimal pre-integration baseline that does not delay integration unnecessarily.

At minimum, determine whether the following should be captured before the submodule/API migration:

* current correctness test count;
* 106/106 scheme coverage;
* boundary-mode coverage;
* 1D forward coverage;
* 1D inverse coverage;
* 2D forward coverage;
* 2D inverse coverage;
* odd/even dimensions;
* square and non-square matrices;
* Wormhole results;
* Blackhole results;
* maximum absolute error;
* maximum relative error;
* LWT→ILWT round-trip error;
* known high-order FP32 failures;
* device-only timing;
* host-to-device-inclusive timing;
* program compilation timing;
* memory usage;
* L1 usage;
* DRAM usage.

Recommend a small “migration lock” test suite that must pass before and after TTNN integration.

Do not require a complete testing rewrite before integration.

Separate the test strategy into:

1. pre-migration baseline;
2. TTNN bring-up tests;
3. full correctness tests;
4. architecture matrix;
5. multi-device tests;
6. performance regressions;
7. upstream CI tests.

### 7. FFT investigation

Locate and inspect the current FFT implementation after updating the `tt-metal` submodule.

The document must answer:

* where FFT lives;
* whether it is public or experimental;
* its public Python and C++ APIs;
* namespace structure;
* nanobind structure;
* validation model;
* output-shape model;
* dtype/layout restrictions;
* memory-config handling;
* program factories;
* device kernels;
* architecture handling;
* large-input behavior;
* multi-device behavior;
* testing strategy;
* documentation;
* any reusable signal-processing abstractions.

Compare FFT with the proposed wavelet integration.

Create a table:

| Design concern | FFT approach | Conv approach | Other relevant op | Recommendation for wavelet |

Do not copy FFT blindly. Explain which patterns are appropriate and which are not.

### 8. Valid TTNN operation contract

Define what contracts `lwt`, `ilwt`, `lwt_2d`, and `ilwt_2d` must provide to be valid TTNN operations.

Cover these layers separately:

#### Public semantic contract

* mathematical meaning;
* boundary behavior;
* scheme selection;
* output ordering;
* shape behavior;
* inverse reconstruction behavior;
* precision expectations.

#### Tensor contract

* accepted ranks;
* accepted logical shapes;
* accepted padded shapes;
* dtype;
* layout;
* memory layout;
* storage location;
* device or mesh placement;
* batch semantics.

#### Operation validation contract

* exact validation checks;
* failure conditions;
* error messages;
* unsupported combinations.

#### Output specification contract

* shape inference;
* dtype inference;
* layout;
* memory configuration;
* distributed tensor specification;
* output allocation.

#### Device-operation contract

* operation attributes;
* tensor arguments;
* optional tensor arguments;
* hash/program-cache inputs;
* program factory selection;
* runtime-argument override;
* architecture-dependent configuration.

#### Python binding contract

* Python function names;
* positional and keyword-only arguments;
* enum/string handling;
* return types;
* docstrings;
* defaults;
* exceptions.

#### Testing contract

* golden/reference implementation;
* PyWavelets comparison;
* tolerance policy;
* round-trip policy;
* architecture coverage;
* mesh coverage.

### 9. Proposed public API

Propose concrete C++ and Python API signatures for all four operations.

At minimum, evaluate these questions:

#### Scheme representation

Should the API accept:

* a wavelet name such as `"db7"`;
* a `Wavelet` enum;
* a `WaveletScheme` struct;
* a numeric scheme ID;
* coefficient tensors;
* a precompiled scheme object?

All 106 schemes must remain supported.

Discuss how the selected representation affects:

* nanobind;
* program-cache keys;
* compile-time kernel specialization;
* generated scheme headers;
* Python usability;
* ABI stability;
* adding future schemes.

#### 1D forward output

Evaluate:

```python
approx, detail = ttnn.lwt(...)
```

versus:

```python
output = ttnn.lwt(...)  # packed representation
```

#### 1D inverse input

Evaluate:

```python
output = ttnn.ilwt(approx, detail, ...)
```

versus accepting one packed tensor.

#### 2D forward output

Evaluate:

```python
ll, lh, hl, hh = ttnn.lwt_2d(...)
```

versus one packed tensor.

#### 2D inverse input

Evaluate:

```python
output = ttnn.ilwt_2d(ll, lh, hl, hh, ...)
```

versus one packed tensor.

#### Shape restoration metadata

Determine how ILWT knows the original odd/even logical dimensions.

Evaluate:

* explicit `output_shape`;
* metadata encoded in a result object;
* deterministic inference;
* returned auxiliary shape values;
* padded tensor metadata.

Recommend one upstream-compatible design.

Provide complete proposed signatures for:

* C++ public API;
* registered operation declarations;
* nanobind bindings;
* Python-facing usage examples.

### 10. Primitive versus composite operation structure

Determine whether each public wavelet operation should be:

* one registered primitive device operation;
* a composite TTNN wrapper around multiple primitive operations;
* a public wrapper calling one internal device primitive;
* separate primitives for split, lifting, scale, and interleave;
* one primitive per transform direction and dimensionality.

The existing algorithm should remain fused where it is currently fused.

Consider this possible structure:

```text
ttnn::lwt
  -> public validation and API normalization
  -> ttnn::prim::lwt
  -> host planner
  -> program factory
  -> reader/compute/writer kernels
```

But verify whether this matches current upstream conventions.

Propose an exact directory tree, for example:

```text
ttnn/cpp/ttnn/operations/wavelet/
    lwt.hpp
    lwt.cpp
    lwt_nanobind.hpp
    lwt_nanobind.cpp
    common/
    device/
        lwt_device_operation.hpp
        lwt_device_operation.cpp
        lwt_program_factory.hpp
        lwt_program_factory.cpp
        kernels/
            dataflow/
            compute/
```

Do not use this example mechanically. Derive the final tree from current TTNN best practices.

Include:

* CMake/build registration files;
* nanobind module registration;
* Python exports;
* documentation location;
* unit-test location;
* performance-test location.

### 11. Scheme specialization and compilation strategy

This is a critical part of the design.

The current implementation uses static scheme specialization and generated scheme headers.

All 106 schemes are required.

Analyze:

* whether each scheme creates a unique compute-kernel binary;
* whether LWT and ILWT create separate binaries;
* whether row-major and tile layouts create separate binaries;
* whether 1D and 2D create separate binaries;
* whether boundary modes affect compute binaries or only readers;
* potential total kernel-variant count;
* compile-time cost;
* JIT cache cost;
* binary-cache size;
* program-cache key size;
* CI cost;
* Python first-call latency.

Compare three strategies:

1. preserve fully compile-time-specialized schemes;
2. use runtime coefficient/step metadata;
3. hybrid design:

   * compile-time bounded kernel structure;
   * runtime coefficients and scheme metadata;
   * specialized fast paths where justified.

The algorithm must not be changed merely to reduce compile time.

If compile-time specialization is required for current performance or register allocation, state that.

Recommend how all 106 schemes should be registered, generated, packaged, and tested in an upstream TTNN tree.

### 12. Memory and sharding contract

Use TTNN tensor terminology precisely:

* tensor layout:

  * row-major;
  * tile.
* memory layout:

  * interleaved;
  * height-sharded;
  * width-sharded;
  * block-sharded.
* buffer type:

  * DRAM;
  * L1.

For every operation, define the recommended first-PR support matrix.

Example format:

| Operation | Input layout | Input memory layout | Input buffer | Output layout | Output memory layout | Supported in first PR |
| --------- | ------------ | ------------------- | ------------ | ------------- | -------------------- | --------------------- |

Explicitly address:

* current use of narrow 32×16 FP32 tiles in 1D kernels;
* standard TTNN tile contracts;
* whether custom tile shapes are acceptable for an upstream public op;
* logical versus physical padding;
* output memory configuration;
* sharded outputs;
* resharding costs;
* DRAM round trips;
* zero-copy opportunities;
* preallocated outputs;
* L1 capacity checks;
* per-core CB allocation.

### 13. Program factory and cache design

Based on current TTNN operations, propose:

* operation attribute structs;
* tensor argument structs;
* tensor-return structs;
* `validate`;
* `compute_output_specs`;
* `create_output_tensors`;
* program factory classes;
* cached-program type;
* `create`;
* `override_runtime_arguments`.

List exactly what belongs in:

* operation attributes;
* compile-time arguments;
* runtime arguments;
* program-cache key;
* per-core runtime arguments.

Consider:

* scheme ID;
* direction;
* boundary mode;
* logical dimensions;
* padded dimensions;
* input/output layout;
* memory configuration;
* core grid;
* chunk count;
* dependency intervals;
* coefficient metadata;
* architecture;
* compute-kernel config.

Avoid putting dynamic tensor addresses into the compile cache key when runtime-argument override is appropriate.

### 14. Validation and safeguards

Propose exact validation checks.

At minimum:

* tensor is allocated;
* tensor is on device when required;
* all tensors are on the same device or compatible mesh;
* dtype is supported;
* layout is supported;
* memory layout is supported;
* rank is supported;
* dimensions are nonzero where required;
* dimensions fit index types;
* shape is compatible with boundary mode;
* scheme ID is valid;
* generated scheme metadata is internally consistent;
* tap count does not exceed kernel capacity;
* coefficient counts fit compile/runtime argument limits;
* inverse inputs have compatible shapes;
* inverse output shape is valid;
* output memory config is valid;
* core grid is valid;
* selected CB sizes fit L1;
* sharding is compatible with work partition;
* mesh distribution strategy is supported;
* unsupported architecture is rejected;
* no accidental input/output aliasing;
* no integer overflow in byte-size calculations.

Also determine whether host-side input scanning for NaN/Inf is appropriate or too expensive.

Document expected Tensix special-value behavior and whether wavelet operations should:

* propagate values;
* reject them;
* provide optional debug detection;
* leave handling to the caller.

### 15. Recommended first PR scope

Propose a realistic first upstream PR.

The scope should be small enough to review but must preserve the core value of the project.

Evaluate a first PR containing:

* all 106 schemes;
* 1D and 2D;
* forward and inverse;
* all boundary modes;
* Wormhole and Blackhole;
* single-device only;
* FP32 only;
* DRAM-interleaved inputs and outputs;
* no out-of-core support;
* no cross-chip spatial partitioning.

If this is too large for one PR, propose a dependency-ordered PR series without removing required final functionality.

For example:

1. common scheme and boundary infrastructure;
2. 1D LWT/ILWT primitive;
3. 1D public API and bindings;
4. 2D LWT/ILWT primitive;
5. 2D public API and bindings;
6. sharded tensor support;
7. multi-chip support;
8. optimization and performance follow-up.

Provide recommended PR boundaries and explain why each is independently reviewable.

### 16. Migration plan

Produce a step-by-step implementation plan.

For each stage include:

* files to add;
* files to modify;
* source code moved versus wrapped;
* expected tests;
* risks;
* completion criteria.

The migration plan must preserve the current standalone `tt-wavelet` implementation until the TTNN version reaches correctness parity.

Do not suggest deleting the existing implementation during initial integration.

Include a compatibility period where:

* standalone backend and TTNN backend can both run;
* results are compared automatically;
* performance regressions are measured;
* the TTNN implementation becomes authoritative only after parity.

### 17. Questions for the project owner

Create a dedicated section:

```text
## Decisions Required From the Project Owner
```

Ask only questions that materially affect the API or implementation.

For every question provide:

* why the decision matters;
* available options;
* your recommended default;
* consequences of each option.

At minimum include decisions about:

1. public namespace;
2. experimental versus stable API;
3. scheme argument representation;
4. 1D output representation;
5. 2D output representation;
6. inverse shape restoration;
7. single-level versus multilevel first API;
8. supported tensor ranks;
9. batch semantics;
10. channel semantics;
11. input and output layouts;
12. sharded tensor support in first PR;
13. multi-chip scope in first PR;
14. preallocated output support;
15. whether to expose low-level tuning configuration;
16. whether the existing `ttwv` namespace remains internally;
17. expected numerical tolerance;
18. whether known high-order FP32 differences are acceptable;
19. first-PR performance regression threshold;
20. long-term ownership of generated 106-scheme artifacts.

Do not ask questions whose answers can be obtained from the source tree.

### 18. Correct or challenge project assumptions

Create a section:

```text
## Assumptions Verified, Corrected, or Rejected
```

Evaluate the assumptions in this request.

At minimum include:

* “N300 execution requires sending a program to another chip through NoC.”
* “The same program automatically runs twice on two chips.”
* “A tensor can be divided into independent regions across two chips.”
* “The custom `ttwv` namespace cannot be used anywhere in a TTNN operation.”
* “Updating to the latest `tt-metal` main is sufficient to understand current FFT behavior.”
* “Integration can safely happen before capturing a baseline.”
* “All 106 schemes should necessarily be exposed as 106 public operations.”
* “A large input problem is equivalent to an L1-capacity problem.”
* “Wormhole and Blackhole should use identical device source code.”
* “Conv is the best single template for the wavelet operation.”

For each assumption label it:

* verified;
* partially correct;
* incorrect;
* architecture/version dependent;
* unresolved.

Explain the corrected model.

### 19. Final recommendation

End the document with a decisive recommendation.

It must contain:

1. proposed public API;
2. proposed namespace;
3. proposed directory structure;
4. proposed primitive/composite layering;
5. first-PR device/layout support;
6. large-input policy;
7. multi-chip policy;
8. scheme-specialization policy;
9. Wormhole/Blackhole policy;
10. pre-integration test baseline;
11. recommended PR sequence;
12. top five technical risks;
13. top five unanswered owner decisions.

The conclusion must not be vague.

## Required document structure

Use this approximate structure:

```text
# TTNN Integration Design for tt-wavelet

## 1. Executive Summary
## 2. Repository and Version Baseline
## 3. Current tt-wavelet Architecture
## 4. Survey of TTNN Operation Patterns
## 5. FFT Case Study
## 6. Conv and Other Relevant Case Studies
## 7. Proposed TTNN Wavelet Architecture
## 8. Public API and Namespace Contract
## 9. Tensor, Shape, Layout, and Memory Contract
## 10. Large-Input Execution
## 11. Multi-Chip and Mesh Execution
## 12. Wormhole and Blackhole Compatibility
## 13. Scheme Specialization and Program Caching
## 14. Validation and Safety Checks
## 15. Testing and Performance Contract
## 16. Proposed Source-Tree Layout
## 17. Migration and PR Plan
## 18. Decisions Required From the Project Owner
## 19. Assumptions Verified, Corrected, or Rejected
## 20. Final Recommendation
## Appendix A. TTNN Operation Inventory
## Appendix B. Source References
## Appendix C. Proposed API Signatures
## Appendix D. Support Matrices
```

You may improve the structure, but every required topic must remain covered.

## Quality requirements

The document must:

* be based on the current checked-out source;
* use exact source paths;
* include line references;
* inspect several operation implementations;
* avoid treating Conv as the only reference;
* include FFT;
* distinguish source facts from recommendations;
* explicitly discuss uncertainty;
* provide concrete API signatures;
* provide concrete directory layouts;
* provide concrete support matrices;
* provide a concrete first-PR scope;
* provide a concrete migration plan;
* identify blockers;
* identify owner decisions;
* correct inaccurate assumptions;
* preserve the existing algorithm and all 106 schemes.

Do not produce a superficial summary.

Do not stop after locating a single registered operation or FFT wrapper.

Do not implement the operation.

The final response after completing the work should contain:

1. the path of the generated Markdown file;
2. the old and new `tt-metal` SHAs;
3. a concise list of the most important recommendations;
4. any blockers or unresolved source ambiguities;
5. `git status --short`;
6. `git diff --stat`;
7. confirmation that no algorithm source code was modified.
