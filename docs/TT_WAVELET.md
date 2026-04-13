# TT Wavelet Example Walkthrough

## Goal

This section shows the full execution path of the current TT-wavelet pipeline on a deliberately small example, from JSON scheme loading to final device buffers.

The example is chosen to be:

- mathematically small enough to compute by hand
- structurally rich enough to exercise:
  - symmetric pad
  - even/odd split
  - fused `predict/update`
  - scalar scale steps
  - ping-pong stream buffering

The concrete scheme is [bior1.3](../ttnn-wavelet/lifting_schemes/bior1.3.json).

## Example Setup

Input signal:

$$
x = [1, 2, 3, 4, 5]
$$

Scheme:

```json
{
  "tap_size": 6,
  "delay": { "even": 1, "odd": 2 },
  "steps": [
    { "type": "predict", "shift": 0, "coefficients": [-1.0] },
    { "type": "update", "shift": -1, "coefficients": [0.0625, 0.5, -0.0625] },
    { "type": "scale-even", "shift": 0, "coefficients": [1.4142135623730951] },
    { "type": "scale-odd", "shift": 0, "coefficients": [0.7071067811865475] }
  ]
}
```

Interpretation:

- `tap_size = 6` means preprocess uses:

$$
P = \text{tap\_size} - 1 = 5
$$

- the logical split-stream origins are:

$$
\delta_e = 1, \qquad \delta_o = 2
$$

- the steps are:
  - one `predict`
  - one `update`
  - one `scale-even`
  - one `scale-odd`

## Full Code Path

The end-to-end host path is:

1. [tt-wavelet/main.cpp](../tt-wavelet/main.cpp)
2. [tt-wavelet/tt_wavelet/src/lifting/plan.cpp](../tt-wavelet/tt_wavelet/src/lifting/plan.cpp)
3. [tt-wavelet/tt_wavelet/src/pad_split/device.cpp](../tt-wavelet/tt_wavelet/src/pad_split/device.cpp)
4. [tt-wavelet/tt_wavelet/src/lifting/device.cpp](../tt-wavelet/tt_wavelet/src/lifting/device.cpp)
5. device kernels:
   - [pad_split_1d_reader.cpp](../tt-wavelet/kernels/dataflow/pad_split_1d_reader.cpp)
   - [pad_split_1d_writer.cpp](../tt-wavelet/kernels/dataflow/pad_split_1d_writer.cpp)
   - [lwt_reader.cpp](../tt-wavelet/kernels/dataflow/lwt_reader.cpp)
   - [lwt_compute.cpp](../tt-wavelet/kernels/lwt_compute.cpp)
   - [lwt_writer.cpp](../tt-wavelet/kernels/dataflow/lwt_writer.cpp)
   - [lwt_scale_reader.cpp](../tt-wavelet/kernels/dataflow/lwt_scale_reader.cpp)
   - [lwt_mul_compute.cpp](../tt-wavelet/kernels/lwt_mul_compute.cpp)
   - [lwt_row_major_writer.cpp](../tt-wavelet/kernels/dataflow/lwt_row_major_writer.cpp)

The high-level control flow is:

1. `main.cpp`
   - parses scheme path and signal
   - allocates the input `MeshBuffer`
   - loads the runtime lifting scheme
   - builds preprocess and lifting execution objects
2. `create_lifting_preprocess_program(...)`
   - allocates working buffers:
     - `even_ping`
     - `even_pong`
     - `odd_ping`
     - `odd_pong`
   - calls `make_forward_lifting_plan(...)`
   - builds the pad+split program
3. `run_preprocess(...)`
   - physically materializes padded-even and padded-odd streams into:
     - `even_ping`
     - `odd_ping`
4. `execute_forward_lifting(...)`
   - runs all leading `predict/update` steps in one LWT program
   - then runs scale steps one-by-one
5. `main.cpp`
   - reads back the final active even/odd buffers

## Physical Buffers

### Host-visible DRAM descriptors

The base descriptor type is [SignalBuffer](../tt-wavelet/tt_wavelet/include/common/signal.hpp).

For this example:

- input logical length is `5`
- `stick_width = 32`
- `element_size_bytes = 4`

So the input buffer occupies:

- logical length: `5`
- physical stick count: `1`
- physical length: `32`
- physical bytes: `128`

### Working buffers

After pad+split, the padded signal length is:

$$
L_{\text{padded}} = 5 + 5 + 5 = 15
$$

Hence:

- even logical length:

$$
\left\lceil \frac{15}{2} \right\rceil = 8
$$

- odd logical length:

$$
\left\lfloor \frac{15}{2} \right\rfloor = 7
$$

The working set is therefore:

- `even_ping`: capacity for `8` logical samples
- `even_pong`: capacity for `8` logical samples
- `odd_ping`: capacity for `7` logical samples
- `odd_pong`: capacity for `7` logical samples

All four are allocated as `MeshBuffer`s in [lifting/device.cpp](../tt-wavelet/tt_wavelet/src/lifting/device.cpp).

### Important distinction: capacity vs logical active span

The physical buffer capacity is fixed at preprocess size. It does **not** shrink after each lifting step.

What changes per step is:

- `StreamState.shift`
- `StreamState.length`
- `LiftingStepRoute.output_length`
- `source_offset`
- `base_offset`

So:

- physical buffers are reusable scratch storage
- logical validity shrinks through planner state
- kernels process only the logical prefix described by runtime args

This is the central buffer-management rule of the current TT path.

## Preprocess: Symmetric Pad and Split

### Mathematical result

With symmetric extension and pad `5`:

$$
x_{\text{padded}} = [5,4,3,2,1,\,1,2,3,4,5,\,5,4,3,2,1]
$$

The split streams are:

$$
e_0 = x_{\text{padded}}[0::2] = [5,3,1,2,4,5,3,1]
$$

$$
o_0 = x_{\text{padded}}[1::2] = [4,2,1,3,5,4,2]
$$

### Logical stream state

The initial planner states in [lifting/plan.cpp](../tt-wavelet/tt_wavelet/src/lifting/plan.cpp) are:

$$
E^{(0)}: (\text{shift}=1, \text{length}=8)
$$

$$
O^{(0)}: (\text{shift}=2, \text{length}=7)
$$

This means:

- dense buffer `even_ping` stores 8 values, but its logical support starts at index `1`
- dense buffer `odd_ping` stores 7 values, but its logical support starts at index `2`

### Device execution

The preprocess device program is built in [pad_split/device.cpp](../tt-wavelet/tt_wavelet/src/pad_split/device.cpp).

It launches:

- reader: [pad_split_1d_reader.cpp](../tt-wavelet/kernels/dataflow/pad_split_1d_reader.cpp)
- writer: [pad_split_1d_writer.cpp](../tt-wavelet/kernels/dataflow/pad_split_1d_writer.cpp)

Reader runtime args are:

```text
[src_addr, input_length, padded_length, left_pad]
```

For this example:

```text
[input_addr, 5, 15, 5]
```

The preprocess reader:

- reads padded symmetric values from the original input
- emits even positions into `cb_even`
- emits odd positions into `cb_odd`

The preprocess writer:

- flushes `cb_even` into `even_ping`
- flushes `cb_odd` into `odd_ping`

At this point:

- `even_ping` contains `e_0`
- `odd_ping` contains `o_0`
- `even_pong` and `odd_pong` are allocated but not yet written

## Step Routing and Ping-Pong Slots

The planner in [lifting/plan.cpp](../tt-wavelet/tt_wavelet/src/lifting/plan.cpp) starts with:

```text
active.even = even:ping
active.odd  = odd:ping
```

For this scheme, the routes are:

| step | type       | source     | base       | output     | source_length | base_length | source_offset | base_offset | output_length |
|------|------------|------------|------------|------------|---------------|-------------|---------------|-------------|---------------|
| 0    | predict    | even:ping  | odd:ping   | odd:pong   | 8             | 7           | 1             | 0           | 7             |
| 1    | update     | odd:pong   | even:ping  | even:pong  | 7             | 8           | 0             | 2           | 5             |
| 2    | scale-even | even:pong  | even:pong  | even:ping  | 5             | 5           | 0             | 0           | 5             |
| 3    | scale-odd  | odd:pong   | odd:pong   | odd:ping   | 7             | 7           | 0             | 0           | 7             |

So the slot trajectory is:

```text
preprocess:
  even_ping <- e0
  odd_ping  <- o0

predict:
  read  even_ping + odd_ping
  write odd_pong

update:
  read  odd_pong + even_ping
  write even_pong

scale-even:
  read  even_pong
  write even_ping

scale-odd:
  read  odd_pong
  write odd_ping
```

Final active outputs are therefore:

```text
even -> ping
odd  -> ping
```

## Step 0: Predict

### JSON definition

```json
{ "type": "predict", "shift": 0, "coefficients": [-1.0] }
```

So:

- source stream: even
- base stream: odd
- $k = 1$
- $\Delta = 0$
- $h[0] = -1$

### Planner geometry

Source state:

$$
\sigma_s = 1, \quad L_s = 8
$$

Base state:

$$
\sigma_b = 2, \quad L_b = 7
$$

Compute geometry:

$$
\text{conv\_shift} = 1 + 0 + \min(8,1) - 1 = 1
$$

$$
\text{conv\_length} = 8 - 1 + 1 = 8
$$

$$
\text{out\_shift} = \max(2,1) = 2
$$

$$
\text{out\_end} = \min(2+7,\;1+8) = 9
$$

$$
\text{out\_length} = 9 - 2 = 7
$$

$$
\text{source\_offset} = 2 - 1 = 1
$$

$$
\text{base\_offset} = 2 - 2 = 0
$$

### Dense compute equation

Because $k=1$:

$$
y[p] = o_0[p] - e_0[p+1], \qquad p \in [0,7)
$$

Using:

$$
e_0 = [5,3,1,2,4,5,3,1]
$$

$$
o_0 = [4,2,1,3,5,4,2]
$$

we get:

$$
o_1 = [1,1,-1,-1,0,1,1]
$$

### Device buffers used

Physical route:

- source DRAM buffer: `even_ping`
- base DRAM buffer: `odd_ping`
- output DRAM buffer: `odd_pong`

Reader runtime block for step 0 in [lifting/device.cpp](../tt-wavelet/tt_wavelet/src/lifting/device.cpp):

```text
[src_addr=even_ping,
 src_len=8,
 base_addr=odd_ping,
 base_len=7,
 out_len=7,
 out_sticks=1,
 src_off=1,
 base_off=0,
 src_left_pad=16]
```

The `src_left_pad=16` value comes from:

$$
17 - k = 17 - 1 = 16
$$

and exists only to satisfy the SFPI stencil warmup convention. It is not a logical wavelet shift.

The reader in [lwt_reader.cpp](../tt-wavelet/kernels/dataflow/lwt_reader.cpp):

- reads source values from `even_ping` starting at dense index `1`
- reads base values from `odd_ping` starting at dense index `0`
- packs:
  - source tile 0
  - source tile 1
  - base tile

The compute kernel in [lwt_compute.cpp](../tt-wavelet/kernels/lwt_compute.cpp):

- unpacks `k=1`
- runs `hstencil_row<1>(...)`
- adds the result to the base tile

The writer in [lwt_writer.cpp](../tt-wavelet/kernels/dataflow/lwt_writer.cpp):

- writes the packed output tile to `odd_pong`

After step 0:

- `odd_pong` contains the new odd stream
- logical odd state becomes:

$$
(\text{shift}=2,\; \text{length}=7)
$$

## Step 1: Update

### JSON definition

```json
{ "type": "update", "shift": -1, "coefficients": [0.0625, 0.5, -0.0625] }
```

So:

- source stream: updated odd
- base stream: original even
- $k = 3$
- $\Delta = -1$
- coefficients:

$$
h = \left[\frac{1}{16}, \frac{1}{2}, -\frac{1}{16}\right]
$$

### Planner geometry

Source state is now the predicted odd stream:

$$
\sigma_s = 2, \quad L_s = 7
$$

Base state remains:

$$
\sigma_b = 1, \quad L_b = 8
$$

Compute geometry:

$$
\text{conv\_shift} = 2 + (-1) + \min(7,3) - 1 = 3
$$

$$
\text{conv\_length} = 7 - 3 + 1 = 5
$$

$$
\text{out\_shift} = \max(1,3) = 3
$$

$$
\text{out\_end} = \min(1+8,\;3+5) = 8
$$

$$
\text{out\_length} = 8 - 3 = 5
$$

$$
\text{source\_offset} = 3 - 3 = 0
$$

$$
\text{base\_offset} = 3 - 1 = 2
$$

### Dense compute equation

The step equation is:

$$
y[p] = e_0[2+p] + \frac{1}{16}o_1[p+2] + \frac{1}{2}o_1[p+1] - \frac{1}{16}o_1[p]
$$

for:

$$
p \in [0,5)
$$

Using:

$$
o_1 = [1,1,-1,-1,0,1,1]
$$

$$
e_0 = [5,3,1,2,4,5,3,1]
$$

we get the valid update result:

$$
e_1 = [1.375,\;1.375,\;3.5625,\;5.125,\;3.5625]
$$

### Device buffers used

Physical route:

- source DRAM buffer: `odd_pong`
- base DRAM buffer: `even_ping`
- output DRAM buffer: `even_pong`

Reader runtime block:

```text
[src_addr=odd_pong,
 src_len=7,
 base_addr=even_ping,
 base_len=8,
 out_len=5,
 out_sticks=1,
 src_off=0,
 base_off=2,
 src_left_pad=14]
```

Now:

$$
17 - k = 17 - 3 = 14
$$

Again, the reader performs dense slicing. It does not understand wavelet semantics. It only knows:

- where source begins
- where base begins
- how many logical output elements are valid
- how much left stencil warmup padding is needed

After step 1:

- `even_pong` contains the updated even stream
- logical even state becomes:

$$
(\text{shift}=3,\; \text{length}=5)
$$

## Step 2: Scale-even

### JSON definition

```json
{ "type": "scale-even", "shift": 0, "coefficients": [1.4142135623730951] }
```

This step is not part of the fused predict/update program. It is executed separately in [lifting/device.cpp](../tt-wavelet/tt_wavelet/src/lifting/device.cpp) using:

- reader: [lwt_scale_reader.cpp](../tt-wavelet/kernels/dataflow/lwt_scale_reader.cpp)
- compute: [lwt_mul_compute.cpp](../tt-wavelet/kernels/lwt_mul_compute.cpp)
- writer: [lwt_row_major_writer.cpp](../tt-wavelet/kernels/dataflow/lwt_row_major_writer.cpp)

The scalar is:

$$
\alpha = \sqrt{2}
$$

So:

$$
e_2 = \alpha e_1
$$

which gives:

$$
e_2 = [1.9445436483,\;1.9445436483,\;5.0381358160,\;7.2478445072,\;5.0381358160]
$$

Buffers:

- source DRAM buffer: `even_pong`
- output DRAM buffer: `even_ping`

Important:

- the scale reader emits two tiles:
  - input tile
  - coefficient tile filled with the scalar repeated across row 0
- the multiply kernel is pure FP32 tile multiply
- the writer stores the scaled result back into row-major DRAM

## Step 3: Scale-odd

### JSON definition

```json
{ "type": "scale-odd", "shift": 0, "coefficients": [0.7071067811865475] }
```

The scalar is:

$$
\beta = \frac{1}{\sqrt{2}}
$$

So:

$$
o_2 = \beta o_1
$$

which gives:

$$
o_2 = [0.7071067812,\;0.7071067812,\;-0.7071067812,\;-0.7071067812,\;0,\;0.7071067812,\;0.7071067812]
$$

Buffers:

- source DRAM buffer: `odd_pong`
- output DRAM buffer: `odd_ping`

At this point, the final active streams are:

- `even_ping`
- `odd_ping`

## On-chip Circular Buffers

### Preprocess program

In [pad_split/device.cpp](../tt-wavelet/tt_wavelet/src/pad_split/device.cpp):

- `c_0`: even output stream
- `c_1`: odd output stream
- `c_2`: input stick cache

### Fused predict/update program

In [lifting/device.cpp](../tt-wavelet/tt_wavelet/src/lifting/device.cpp):

- `c_0`: source tile 0
- `c_1`: source tile 1
- `c_2`: base tile
- `c_16`: output tile
- `c_3`: source stick cache
- `c_4`: base stick cache
- `c_5`: sync token between reader and writer across fused steps

The two source tiles exist because the stencil may need values that cross the first 32-lane row-major stick boundary.

### Scale program

In [lifting/device.cpp](../tt-wavelet/tt_wavelet/src/lifting/device.cpp):

- `c_0`: input tile
- `c_1`: coefficient tile
- `c_16`: output tile
- `c_2`: source stick cache

## Why the Buffers Are Organized This Way

The current design is optimized around three invariants:

### 1. Dense DRAM, not sparse logical arrays

All device-visible buffers are dense row-major sticks. Logical shifts are planner metadata only.

### 2. Reusable working storage

The four working buffers:

- `even_ping`
- `even_pong`
- `odd_ping`
- `odd_pong`

are scratch buffers reused across the full scheme. They are not reallocated per step.

### 3. No in-place predict/update hazards

Each update writes into the toggled slot, so the source snapshot remains stable for the duration of the current step.

This is why the route table always looks like:

- read active slot
- write inactive slot
- toggle active reference

## End-to-end Summary

For this tiny example, the complete logical pipeline is:

```text
input x
  -> symmetric pad
  -> split
  -> e0 in even_ping
  -> o0 in odd_ping
  -> predict(e0 -> o1) writes odd_pong
  -> update(o1 -> e1) writes even_pong
  -> scale-even(e1 -> e2) writes even_ping
  -> scale-odd(o1 -> o2) writes odd_ping
  -> final outputs are even_ping and odd_ping
```

Final values:

$$
\text{approximation} = e_2 = [1.9445436483,\;1.9445436483,\;5.0381358160,\;7.2478445072,\;5.0381358160]
$$

$$
\text{detail} = o_2 = [0.7071067812,\;0.7071067812,\;-0.7071067812,\;-0.7071067812,\;0,\;0.7071067812,\;0.7071067812]
$$

This example captures the full current TT-wavelet execution model:

- logical support is tracked in the planner
- physical storage stays dense and fixed-capacity
- reader kernels convert logical geometry into packed tiles
- compute kernels execute stencil or scalar math only
- writer kernels flush dense row-major results back to DRAM
