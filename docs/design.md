# Single-Program 1D LWT Design (Phase 1)

## Goal

Implement Phase 1 of full 1D lifting wavelet transform as one TT-Metal program launch that:

1. Performs wavelet padding and even/odd split exactly once for the whole signal.
2. Executes all predict/update lifting steps in sequence inside the same program.
3. Uses the existing SFPI horizontal stencil kernel for predict/update steps.
4. Keeps the current naive row-major row0 tilization path for the first integration.

This document describes the proposed device-side structure only. It is a design for the next implementation step.

The optimized mapping from [ROW_MAJOR.md](./ROW_MAJOR.md) (AdvancedRowMajor path) is explicitly deferred to a follow-up
phase after the single-program predict/update path is stable.

## Non-Goals

- This design does not change the SFPI stencil math.
- This design does not require a different host program per predict/update step.
- This design does not materialize a fully tileized padded signal in DRAM.
- This design does not include AdvancedRowMajor packing yet.
- This design does not require scale/swap to be merged into the same kernel loop in Phase 1.

## Core Idea

The program has two logical phases, but they are part of one LWT program launch:

1. Preprocess phase, runs once
   - read raw input
   - apply physical wavelet padding
   - split into even and odd row-major streams
   - write those streams into DRAM working buffers

2. Step phase, runs for every predict/update lifting step in Phase 1
  - select source stream and base stream for current step
  - build only the SFPI halo `17 - k` needed for this step
  - pack row-major source/base into the current row0 tile layout
  - run SFPI stencil
  - add base stream
  - write updated stream back as row-major sticks

There is no per-step `already_padded` check. The wavelet padding and split are not conditionally skipped; they are
structurally executed before the step loop starts.

## Why This Is Still One Program

`Program` here means one TT-Metal program object and one launch from host.

Inside that one program, the reader, compute, and writer kernels each contain:

1. a one-time preprocess section
2. a loop over predict/update lifting steps (Phase 1)

So the host does not launch:

- one program for pad/split
- then another program for stencil

Instead it launches one LWT program, and the kernels internally move from preprocess into the step loop.

## Existing Pieces We Reuse

- Whole-signal symmetric padding and split logic from [pad_split_1d_reader.cpp](../tt-wavelet/kernels/dataflow/pad_split_1d_reader.cpp)
- Existing pad/split buffer layout from [pad_split_device.cpp](../tt-wavelet/tt-wavelet/tt_wavelet/src/pad_split_device.cpp)
- Existing SFPI stencil kernel from [stencil_compute.cpp](../tt-wavelet/kernels/stencil_compute.cpp) and
  [stencil_sfpi.h](../tt-wavelet/kernels/stencil_sfpi.h)
- Existing naive row0 helpers from [row_major_tile.hpp](../tt-wavelet/kernels/utils/row_major_tile.hpp) and
  [lwt_row_major_writer.cpp](../tt-wavelet/kernels/dataflow/lwt_row_major_writer.cpp)

## DRAM Buffers

The program needs the following DRAM buffers.

### Input and metadata

- `input_raw`
  - original 1D signal in row-major sticks
- `step_desc_table`
  - array of fixed-size per-step descriptors used by the device step loop

### Working streams

- `even_ping`
- `even_pong`
- `odd_ping`
- `odd_pong`

These are row-major stick buffers. At any step:

- one buffer of a pair is the current readable stream
- the other buffer is the writable output stream

### Final output

- optional `output_low`
- optional `output_high`

These can alias the final active even/odd buffers if desired.

## Why Ping/Pong Is Needed

Predict and update steps consume the previous step output. We must not overwrite the currently readable stream while
still reading from it.

Example:

- predict reads `even_cur` and `odd_cur`, writes `odd_next`
- update reads `odd_cur` and `even_cur`, writes `even_next`

After each step, ownership flips for the updated stream only.

This keeps the untouched stream stable during the current step.

## Step Metadata Layout

The simplest DRAM representation is one fixed-size descriptor per step:

```cpp
struct DeviceStepDesc {
    uint32_t type;
    uint32_t k;
    int32_t shift;
    uint32_t reserved;
    uint32_t coeffs_packed[17];
};
```

This kind of struct is valid for TT-Metal kernels as long as it follows the usual kernel-compiler restrictions:

- fixed-size scalar fields only
- optional nested structs
- optional 1D arrays
- no pointers
- no 2D arrays
- no virtual or complex C++ features

Notes:

- `k` is runtime metadata.
- `shift` stays in the descriptor even if the first integration keeps the current SFPI contract and does not yet use it
  in the compute kernel.
- `coeffs_packed` should contain exactly 17 packed fp32 coefficient slots, with unused entries zero-filled.

For `bior3.9`, this naturally supports `k = 1`, then `k = 2`, then `k = 9`, all in one LWT execution.

Recommended first implementation:

- host builds `DeviceStepDesc[]` directly
- host writes that table into DRAM once before launch
- reader walks the descriptor table
- reader pushes one current-step descriptor into `cb_step_desc`
- compute consumes that descriptor

This is simpler than splitting metadata and coefficients into separate DRAM tables. A more compact metadata table plus a
flat coefficient array is still possible later if DRAM footprint becomes important.

## CB Layout

The following circular buffers are proposed for one core.

### Preprocess phase CBs

- `cb_pre_even`
  - preprocess output sticks for even stream
- `cb_pre_odd`
  - preprocess output sticks for odd stream
- `cb_pre_cache`
  - single-stick read cache reused from current pad/split reader

### Step phase CBs

- `cb_src_tile0`
  - first packed source tile for current stencil window
- `cb_src_tile1`
  - second packed source tile for current stencil window
- `cb_base_tile`
  - packed base tile for the stream being updated
- `cb_step_desc`
  - current step descriptor for compute, containing step type, `k`, reserved shift field, and coefficients padded to 17
    fp32 values
- `cb_out_tile`
  - packed tile output from compute before writer inverse-packs it
- `cb_step_cache`
  - single-stick cache for source/base row-major reads during step reader

### Optional control CB

- `cb_ctrl`
  - if we want a small control message stream instead of direct semaphores

## CB Count Summary

Minimum practical count:

- preprocess: 3 CBs
- step path: 6 CBs

Total: 9 CBs

This can be reduced by reusing cache/control CB ids across phases because preprocess and step execution do not overlap
logically.

## Kernel Roles

## Reader Kernel

The reader owns all address-generation and stream preparation.

### Reader phase 0: preprocess once

The reader:

1. reads `input_raw`
2. applies symmetric wavelet padding
3. splits padded signal into even and odd streams
4. pushes row-major sticks into `cb_pre_even` and `cb_pre_odd`

The writer consumes those CBs and writes `even_ping` and `odd_ping`.

This phase is structurally separate from the step loop. It is not implemented as:

```cpp
if (already_padded) { ... }
```

inside the hot per-step loop.

### Reader phase 1: step loop

For every predict/update step (Phase 1):

1. load `DeviceStepDesc`
2. select source stream and base stream
3. push the current descriptor into `cb_step_desc`
4. build source halo of size `17 - k`
5. pack source stream into current naive row0 tile layout
6. pack base stream into matching naive row0 tile layout
7. push packed tiles into `cb_src_tile0`, `cb_src_tile1`, and `cb_base_tile`

The reader is the only place that needs to know:

- which stream is source vs base
- how far to extend left halo for current `k`
- how to map row-major sticks into the current row0 tile layout

## Compute Kernel

The compute kernel stays arithmetic-focused.

### Predict/update

For each packed chunk it:

1. reads current step descriptor from `cb_step_desc`
2. loads source tiles into dst regs
3. loads base tile into dst regs
4. runtime-dispatches on `k`
5. runs SFPI stencil
6. adds base stream
7. packs output tile to `cb_out_tile`

### Runtime `k` with one compute binary

`k` changes per lifting step, so it should not be fixed for the whole transform.

However, we still want the templated SFPI path. The proposed structure is:

```cpp
switch (k) {
    case 1:  run_step<1>();  break;
    case 2:  run_step<2>();  break;
    ...
    case 17: run_step<17>(); break;
}
```

This gives:

- one compute kernel binary
- one TT-Metal program
- runtime `k` per step
- templated unrolled SFPI body for each supported `k`

So `k` is runtime at the LWT execution level, but still resolves into compile-time template specializations inside the
kernel binary.

### Scale and swap

Phase 1 recommendation:

1. Keep scale/swap outside the first single-program predict/update integration.
2. Continue using the existing path for those steps until predict/update is stable.

Phase 2 options:

1. Keep them in the same compute kernel with a runtime branch on `step.type`
2. Handle them in lightweight dedicated code paths inside the same kernel binary

If merged in Phase 2:

- predict/update use the stencil path
- scale-even and scale-odd use a simple row-wise multiply path
- swap does not move data; it swaps buffer ownership metadata

## Writer Kernel

The writer owns all DRAM writes.

### Writer phase 0: preprocess output

Consumes:

- `cb_pre_even`
- `cb_pre_odd`

Writes:

- `even_ping`
- `odd_ping`

### Writer phase 1: step loop

Consumes:

- `cb_out_tile`

Performs:

1. row0 extraction back to row-major sticks (current fixed writer mapping)
2. writes updated stream into the selected target DRAM buffer

The writer does not need to understand stencil semantics. It only needs:

- destination buffer address
- logical output length
- row0 tile to row-major mapping

## Source and Base Selection

The step loop needs a small ownership table on device.

Recommended reader-side state:

```cpp
struct StreamState {
    uint32_t even_read_addr;
    uint32_t even_write_addr;
    uint32_t odd_read_addr;
    uint32_t odd_write_addr;
};
```

Step behavior:

- `predict`
  - source = `even_read`
  - base = `odd_read`
  - output = `odd_write`
  - after step: swap `odd_read` and `odd_write`

- `update`
  - source = `odd_read`
  - base = `even_read`
  - output = `even_write`
  - after step: swap `even_read` and `even_write`

- `scale-even`
  - source = `even_read`
  - base = none
  - output = `even_write`
  - swap even ownership

- `scale-odd`
  - source = `odd_read`
  - base = none
  - output = `odd_write`
  - swap odd ownership

- `swap`
  - no DRAM movement required
  - swap logical interpretation of final low/high stream ownership

## Compile-Time Args

These should stay fixed for the whole program.

### Reader compile-time args

- CB ids
- stick size in bytes
- stick width
- maximum supported coefficient count, `17`
- TensorAccessor payload sizes if needed

Example:

```cpp
reader_ct_args = {
    cb_pre_even,
    cb_pre_odd,
    cb_pre_cache,
    cb_src_tile0,
    cb_src_tile1,
    cb_base_tile,
    cb_step_desc,
    cb_step_cache,
    stick_nbytes,
    stick_width,
    max_k
};
```

### Compute compile-time args

- CB ids for source/base/output/meta/coeff
- optional max supported `k = 17`

Example:

```cpp
compute_ct_args = {
    cb_src_tile0,
    cb_src_tile1,
    cb_base_tile,
    cb_step_desc,
    cb_out_tile
};
```

### Writer compile-time args

- CB ids for preprocess outputs and step outputs
- stick size in bytes
- stick width

## Runtime Args

These describe one concrete LWT execution.

### Program-level runtime args

- `input_raw_addr`
- `input_length`
- `left_wavelet_pad`
- `right_wavelet_pad`
- `step_desc_table_addr`
- `num_steps`
- `even_ping_addr`
- `even_pong_addr`
- `odd_ping_addr`
- `odd_pong_addr`
- optional final output addresses

### Reader runtime args

Recommended reader runtime payload:

```cpp
reader_rt_args = {
    input_raw_addr,
    input_length,
    left_wavelet_pad,
    right_wavelet_pad,
    step_desc_table_addr,
    num_steps,
    even_ping_addr,
    even_pong_addr,
    odd_ping_addr,
    odd_pong_addr
};
```

### Compute runtime args

The compute kernel should need little runtime data beyond lengths and maybe a few constants:

```cpp
compute_rt_args = {
    num_steps,
    even_length,
    odd_length
};
```

All per-step details come through `cb_step_desc`, not through host-updated runtime args.

### Writer runtime args

```cpp
writer_rt_args = {
    num_steps,
    even_ping_addr,
    even_pong_addr,
    odd_ping_addr,
    odd_pong_addr,
    even_length,
    odd_length
};
```

## Why Per-Step Metadata Should Not Come From Host Runtime Args

If the host had to re-issue runtime args for every predict/update step, we would effectively move back toward a
multi-launch design.

The point of the single-program design is:

- host launches once
- device kernels loop over all steps internally

So the full step table must already be available to the device kernels at launch time.

Another important constraint is that runtime args are written by the host before execution and are not a device-side
communication mechanism. A compute kernel cannot modify runtime args that a reader kernel later observes. For dynamic
per-step coordination inside one running program, use one of:

- circular buffers
- semaphores
- shared L1 scratch / mailbox-style data

For this design, CB-based control is the cleanest option because it matches the existing reader -> compute -> writer
dataflow model.

## Coefficient Delivery

The current stencil test path passes coefficients as compile args because it only runs one fixed stencil case.

That is not sufficient for full LWT.

Recommended full-LWT design:

- host builds one `DeviceStepDesc[]` table for the selected wavelet
- host writes that table to DRAM before launch
- reader fetches the current descriptor from the table
- reader writes that descriptor into `cb_step_desc`
- compute pops one descriptor and uses only the first `k` coefficient entries

Padding coefficients to 17 elements inside `DeviceStepDesc` and `cb_step_desc` keeps compute-side loading simple.

## Why `cb_step_desc` Instead of Separate Metadata and Coefficient CBs

Separate metadata and coefficient CBs are not strictly required. The earlier split was mainly for clarity while laying
out responsibilities.

The cleaner version is:

- program-global values stay in runtime args
  - base addresses
  - lengths
  - descriptor table address
  - number of steps
- per-step values are streamed in one descriptor CB
  - `type`
  - `k`
  - reserved `shift`
  - padded coefficient block

This avoids an extra CB without changing the control model.

## Why Not Pure Runtime Args For Per-Step Data

They are known to the host before launch, but they are not compile-time constants for the kernel binary, and they are
not good candidates for pure runtime args if we want one internal device-side step loop.

Runtime args are best used for execution-global values set once at launch. Per-step values must still be advanced inside
the device program. That is why the reader should consume a descriptor table from DRAM and stream one current-step
descriptor to compute.

## Suggested Control Strategy

The simplest robust control scheme is:

1. Host sets one runtime arg block per kernel at launch.
2. Reader walks the step table in DRAM.
3. Reader pushes one `DeviceStepDesc` per step into `cb_step_desc`.
4. Compute pops that descriptor and runs the correct branch for the current step.
5. Writer does not need step coefficients; it only needs destination ownership and row-major output geometry.

This avoids trying to mutate runtime args on device and keeps all dynamic step information inside normal kernel-to-kernel
channels.

## One-Time Wavelet Padding and Split

This is the key structural requirement.

The program should be organized as:

```cpp
preprocess_once();

for (step_idx = 0; step_idx < num_steps; ++step_idx) {
    run_step(step_idx);
}
```

Not as:

```cpp
for (step_idx = 0; step_idx < num_steps; ++step_idx) {
    if (!already_padded) {
        preprocess_once();
    }
    run_step(step_idx);
}
```

The difference matters because:

- the second version puts phase selection into the hot loop
- the first version makes preprocess a separate structural phase

So the answer to "how do we make sure wavelet padding + split is done only one time?" is:

- by kernel structure, not by a repeated per-step flag check

## Synchronization

The program needs lightweight phase and chunk synchronization between reader, compute, and writer.

Two workable approaches:

1. normal CB backpressure plus one phase semaphore
2. explicit semaphores for preprocess-done and step-done events

Recommended:

- use CB backpressure for chunk flow
- use one small phase semaphore to tell compute and writer when preprocess is complete and step execution may begin

## Step Execution Example

For `bior3.9`:

1. preprocess
   - symmetric pad full raw signal
   - split to `even_ping`, `odd_ping`

2. step 0, predict, `k = 1`
   - source = even
   - base = odd
   - writer stores result to `odd_pong`
   - swap odd ping/pong ownership

3. step 1, update, `k = 2`
   - source = odd
   - base = even
   - writer stores result to `even_pong`
   - swap even ping/pong ownership

4. step 2, predict, `k = 9`
   - source = even
   - base = odd
   - writer stores result to `odd_ping` or `odd_pong`, depending on ownership after step 0

This is exactly why `k` must be runtime metadata for the step loop.

## Recommended First Implementation Scope

To keep the next patch manageable:

1. Implement one-program preprocess plus step loop for predict/update only
2. Keep the current naive row0 reader/writer packing path as-is
3. Keep `shift` in metadata but do not expand the addressing policy beyond the current SFPI contract yet
4. Add runtime `switch(k)` dispatch in compute
5. Add `cb_step_desc`
6. Keep scale/swap as follow-up work or simple branches after predict/update path is stable

## Follow-Up Scope (Phase 2)

After Phase 1 is stable and validated:

1. Introduce AdvancedRowMajor packing from [ROW_MAJOR.md](./ROW_MAJOR.md)
2. Add inverse mapping in writer for the advanced layout
3. Re-tune chunk geometry (horizontal expansion first) for better stencil utilization
4. Extend performance/regression tests to compare naive row0 vs advanced packing

## Summary

The recommended design is:

- one TT-Metal program
- one preprocess phase inside that program
- one internal step loop over all predict/update lifting steps
- runtime `k` from device step metadata
- compile-time SFPI specialization selected by runtime `switch(k)`
- row-major working streams in DRAM
- current naive row0 tile packing performed only at the point of stencil execution
- no repeated wavelet padding or split after preprocess

This satisfies the main constraints:

- predict/update execution is one program in Phase 1
- wavelet padding + split happen once
- SFPI path remains reusable
- `k` can vary across steps such as in `bior3.9`
