# Overview

In this documentation is provided an overview how lifting wavelet transform (LWT) using `tt-metal` APIs is implemented.

The overview goes through the exact example of 1D LWT with concrete wavelet scheme [bior1.3](../ttnn-wavelet/lifting_schemes/bior1.3.json).


```json
{
    "tap_size": 6,
    "delay": {
        "even": 1,
        "odd": 2
    },
    "steps": [
        {
            "type": "predict",
            "shift": 0,
            "coefficients": [
                -1.0
            ]
        },
        {
            "type": "update",
            "shift": -1,
            "coefficients": [
                0.0625,
                0.49999999999999994,
                -0.0625
            ]
        },
        {
            "type": "scale-even",
            "shift": 0,
            "coefficients": [
                1.4142135623730951
            ]
        },
        {
            "type": "scale-odd",
            "shift": 0,
            "coefficients": [
                0.7071067811865475
            ]
        }
    ],
    "meta": {
        "l2": {
            "qq": 5.798244541183285e-17,
            "fp64": 7.046532608201141e-17
        }
    }
}
```

Here the full code path:

1. [tt-wavelet/main.cpp](../tt-wavelet/main.cpp)
2. [tt-wavelet/tt_wavelet/src/lifting/plan.cpp](../tt-wavelet/tt_wavelet/src/lifting/plan.cpp)
3. [tt-wavelet/tt_wavelet/src/pad_split/device.cpp](../tt-wavelet/tt_wavelet/src/pad_split/device.cpp)
4. [tt-wavelet/tt_wavelet/src/lifting/device.cpp](../tt-wavelet/tt_wavelet/src/lifting/device.cpp)
5. device kernels:

   5.1 [pad_split_1d_reader.cpp](../tt-wavelet/kernels/dataflow/pad_split_1d_reader.cpp)

   5.2 [pad_split_1d_writer.cpp](../tt-wavelet/kernels/dataflow/pad_split_1d_writer.cpp)

   5.3 [lwt_reader.cpp](../tt-wavelet/kernels/dataflow/lwt_reader.cpp)

   5.4 [lwt_compute.cpp](../tt-wavelet/kernels/lwt_compute.cpp)

   5.5 [lwt_writer.cpp](../tt-wavelet/kernels/dataflow/lwt_writer.cpp)

   5.6 [lwt_scale_reader.cpp](../tt-wavelet/kernels/dataflow/lwt_scale_reader.cpp)

   5.7 [lwt_mul_compute.cpp](../tt-wavelet/kernels/lwt_mul_compute.cpp)

   5.8 [lwt_row_major_writer.cpp](../tt-wavelet/kernels/dataflow/lwt_row_major_writer.cpp)


## Initial interpretation of lifting scheme.

Input signal:

$$
x = [1, 2, 3, 4, 5]
$$


First, we introduce a logical symmetric extension of the signal. The size of the extension is determined by the
**tap_size** parametr.

Interpretation:

- `tap_size = 6` means preprocess uses:

$$
P = \text{tap\\ size} - 1 = 5
$$


P stands for padding size.

### What `delay` means

The `delay` field defines the initial logical origin of the split streams after preprocess.

For this scheme:

$$
\delta_e = \text{delay.even} = 1, \qquad \delta_o = \text{delay.odd} = 2
$$

This means:

- the dense even buffer produced by pad+split is interpreted as starting at logical index `1`
- the dense odd buffer produced by pad+split is interpreted as starting at logical index `2`

So `delay` is not an extra device-side padding step.
It is initial planner metadata that tells the lifting pipeline where the logical support of each split stream begins.

This is also why the initial stream states later become:

$$
E^{(0)}: (\text{shift}=1, \text{length}=8), \qquad
O^{(0)}: (\text{shift}=2, \text{length}=7)
$$

### How `delay` is handled in code

The JSON values are first loaded into the runtime scheme in [lifting/plan.cpp](../tt-wavelet/tt_wavelet/src/lifting/plan.cpp):

```cpp
.delay_even = obj.value("delay", nlohmann::json::object()).value("even", 0),
.delay_odd = obj.value("delay", nlohmann::json::object()).value("odd", 0),
```

Then, when `make_forward_lifting_plan(...)` builds the planner state, these values become the initial stream shifts:

```cpp
StreamState even_state{.shift = scheme.delay_even, .length = even_ping.length};
StreamState odd_state{.shift = scheme.delay_odd, .length = odd_ping.length};
```

From that point on, `delay` is handled simply as the initial value of `StreamState.shift`.
Each lifting step then combines the current stream shift with its own per-step `shift` inside `compute_step_geometry()`.

### Physical Buffers

The base descriptor type is [SignalBuffer](../tt-wavelet/tt_wavelet/include/common/signal.hpp).


```cpp
struct SignalBuffer {
    uint64_t dram_address{0};                    ///< Base DRAM address of the first stick.
    size_t length{0};                            ///< Logical number of scalar samples in the signal.
    uint32_t stick_width{32};                    ///< Number of scalar samples per physical stick.
    uint32_t element_size_bytes{sizeof(float)};  ///< Size of one scalar element in bytes.
```

>Important note the code operates with sticks which can be interpreted as signal chunk of size 32.
>A **stick** is a fixed-width row of scalar samples that forms the minimum addressable transfer unit for the NOC. For the current fp32 device path `stick_width` is 32, meaning each stick is 128 bytes (32 x 4 B).


For this example:

- input logical length is `5`
- `stick_width = 32`
- `element_size_bytes = 4`

So the input buffer occupies:

- logical length: `5`
- physical stick count: `1`
- physical length: `32`
- physical bytes: `128`


## Pad Split Program

The pad split program is the preprocessing stage executed before the lifting steps.

Its job is:

1. logically extend the input signal with symmetric padding
2. split the padded sequence into two streams
3. write these two streams into separate DRAM buffers

In other words, it transforms one input signal

$$
x[n]
$$

into two output signals

$$
x_{\text{even}}[k] = x_{\text{pad}}[2k], \qquad x_{\text{odd}}[k] = x_{\text{pad}}[2k + 1]
$$

where $x_{\text{pad}}$ is the symmetrically padded logical signal.


### Host-side layout

The host creates the split layout in [pad_split/layout.hpp](../tt-wavelet/tt_wavelet/include/pad_split/layout.hpp).

For an input logical length `N`, left padding `L`, and right padding `R`:

$$
N_{\text{pad}} = N + L + R
$$

The two output lengths are:

$$
N_{\text{even}} = \left\lceil \frac{N_{\text{pad}}}{2} \right\rceil, \qquad
N_{\text{odd}} = \left\lfloor \frac{N_{\text{pad}}}{2} \right\rfloor
$$

This is exactly what [make_split_signal()](../tt-wavelet/tt_wavelet/include/common/signal.hpp) computes.

For the running example:

- input length is `5`
- wavelet padding on both sides is `5`
- padded length is `15`
- even length is `8`
- odd length is `7`


### Device program structure

The device program is created in [tt_wavelet/src/pad_split/device.cpp](../tt-wavelet/tt_wavelet/src/pad_split/device.cpp).

It uses:

- one **reader kernel**: [pad_split_1d_reader.cpp](../tt-wavelet/kernels/dataflow/pad_split_1d_reader.cpp)
- one **writer kernel**: [pad_split_1d_writer.cpp](../tt-wavelet/kernels/dataflow/pad_split_1d_writer.cpp)
- three circular buffers in L1:
  - `c_0` for even output sticks
  - `c_1` for odd output sticks
  - `c_2` for a one-stick source cache

So the flow is:

$$
\text{input DRAM} \rightarrow \text{reader kernel} \rightarrow \text{L1 CBs} \rightarrow \text{writer kernel} \rightarrow \text{even/odd DRAM}
$$

The reader produces the split streams into circular buffers, and the writer drains those buffers into the final DRAM outputs.


### Reader kernel arguments

The reader kernel receives runtime arguments:

- `src_addr`: DRAM address of the input signal
- `input_length`: logical input length before padding
- `padded_length`: logical length after padding
- `left_pad`: amount of logical extension on the left

It also receives compile-time arguments:

- `cb_id_even`: circular buffer used for even sticks
- `cb_id_odd`: circular buffer used for odd sticks
- `stick_nbytes`: physical size of one stick in bytes
- `cb_id_cache`: circular buffer used as the source-stick cache
- `stick_width`: number of `float` values inside one stick

The split geometry is computed first:

```cpp
const uint32_t even_stick_count = even_stick_count(padded_length, stick_width);
const uint32_t odd_stick_count = odd_stick_count(padded_length, stick_width);
const uint32_t pair_count = padded_length / 2;
```

Here:

- `pair_count` is the number of complete `(even, odd)` pairs in the padded stream
- `even_stick_count` is the number of physical sticks needed to store all even samples
- `odd_stick_count` is the number of physical sticks needed to store all odd samples


### Source caching

The kernel does not read every scalar from DRAM independently.

Instead it uses [StickReadCache](../tt-wavelet/kernels/primitives/stick_cache.hpp), which stores exactly one source stick in the cache circular buffer:

```cpp
ttwv::kernels::primitives::StickReadCache read_cache{
    cb_id_cache, stick_nbytes, stick_width,
    ttwv::kernels::primitives::kInvalidStick, false};
```

The idea is simple:

- if the next requested scalar belongs to the same source stick as the previous one, reuse the cached stick from L1
- otherwise read the new stick from DRAM into the cache CB

This is handled by `cache_source_stick()` and used by `read_padded_symmetric_value()`.


### How one padded value is read

The central helper is `read_padded_symmetric_value(src, read_cache, input_length, left_pad, out_idx)`.

Its logic is:

1. interpret `out_idx` as an index in the padded logical signal
2. shift it back into the original signal coordinate system:

$$
\text{logical} = \text{out\_idx} - \text{left\_pad}
$$

3. reflect this logical index into the valid source range `[0, input_length - 1]` using `symmetric_index()`
4. locate the corresponding physical stick and lane
5. fetch the stick into cache if needed
6. return the scalar value from the cached stick

So padding is **logical only**.
No padded signal is materialized in DRAM as a separate temporary buffer.
Each padded value is synthesized on demand from the original input signal.

But in Cirrcular Buffer space, the reader kernel produces two full physical streams of sticks corresponding to the even and odd indexed padded values, which are then consumed by the writer kernel for the final DRAM output.

### Symmetric extension for the example

For the example input

$$
x = [1, 2, 3, 4, 5]
$$

with `left_pad = 5` and `right_pad = 5`, the padded logical signal of length `15` becomes:

$$
x_{\text{pad}} = [5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1]
$$

This follows directly from the reflection rule implemented by `symmetric_index()`.

Then the split outputs are:

$$
x_{\text{even}} = [5, 3, 1, 2, 4, 5, 3, 1]
$$

$$
x_{\text{odd}} = [4, 2, 1, 3, 5, 4, 2]
$$


### Main loop of the reader kernel

The main work is performed by:

```cpp
for (uint32_t pair = 0; pair < pair_count; ++pair) {
    push_output_value(even_writer, read_padded_symmetric_value(..., pair * 2));
    push_output_value(odd_writer, read_padded_symmetric_value(..., pair * 2 + 1));
}
```

This loop walks through the padded logical sequence in pairs:

- `pair * 2` is the next even padded index
- `pair * 2 + 1` is the next odd padded index

For each pair:

- the even indexed padded sample is pushed into the even output writer
- the odd indexed padded sample is pushed into the odd output writer

If the padded length is odd, one final value remains after all complete pairs are processed:

```cpp
if (padded_length & 1U) {
    push_output_value(even_writer, read_padded_symmetric_value(..., padded_length - 1));
}
```

This last sample always belongs to the even stream, because the last valid index of an odd-length sequence is even.


### Output stick writers

The helpers `make_output_stick_writer()`, `push_output_value()`, and `flush_partial_output_stick()` are defined in [stick_writer.hpp](../tt-wavelet/kernels/primitives/stick_writer.hpp).

Their role is:

- reserve one output stick in the target circular buffer
- append scalar values into that stick one by one
- once the stick is full, push it to the CB
- if the final stick is only partially filled, zero-fill the unused lanes and push it as well

This means the logical output lengths may be `8` and `7`, but physically each output is still stored as an integer number of full sticks.


### Writer kernel

The writer kernel is simpler than the reader.

It waits until one even or odd stick is available in the corresponding CB, then writes that full stick to the destination DRAM buffer:

```cpp
cb_wait_front(cb_id_even, 1);
noc_async_write(get_read_ptr(cb_id_even), even_dst.get_noc_addr(even_written), stick_nbytes);
cb_pop_front(cb_id_even, 1);
```

It repeats the same logic for the odd stream until all expected even and odd sticks are written.

So the overall division of responsibility is:

- the **reader kernel** computes padding, split indices, caching, and packing into output sticks
- the **writer kernel** performs the final DRAM stores for those produced sticks

>Important Note
>The are space for optimization of padding and splitting logic but during basic implementation this logic was used as it simplier.


## What is `shift` in a lifting step

Before moving to the lifting steps, it is useful to define what `shift` means in the scheme.

Each lifting step applies a local stencil from one stream to another:

$$
y[i] = \text{base}[i] + \sum_{j=0}^{K-1} c_j \cdot \text{source}[i + j + \text{shift}]
$$

Here:

- `source` is the stream being read
- `base` is the stream being updated
- `c_j` are the stencil coefficients
- `shift` moves the stencil relative to the current base index

So:

- `shift = 0` means the default alignment
- `shift > 0` moves the stencil to the right and uses more future samples
- `shift < 0` moves the stencil to the left and uses more past samples

For example, the update step

```json
{
  "type": "update",
  "shift": -1,
  "coefficients": [a, b, c]
}
```

means:

$$
y[i] = \text{base}[i] + a \cdot \text{source}[i - 1] + b \cdot \text{source}[i] + c \cdot \text{source}[i + 1]
$$

So `shift = -1` centers this 3-tap stencil around the current position.

### How `shift` is handled in code

In the implementation, `shift` is not applied directly inside the compute kernel as `i + shift`.

Instead, the host-side planner in [tt_wavelet/src/lifting/plan.cpp](../tt-wavelet/tt_wavelet/src/lifting/plan.cpp) folds it into the logical stream geometry.
Each stream is tracked as:

```cpp
struct StreamState {
    int shift;
    size_t length;
};
```

For a predict or update step, `compute_step_geometry()` combines:

- the current stream shift
- the step-local `shift`
- the stencil width `k`

and computes:

$$
\text{conv\_shift} = \text{source.shift} + \text{kernel\_shift} + \min(\text{source.length}, k) - 1
$$

From this it derives:

- `out_shift`
- `source_offset`
- `base_offset`

So by the time runtime args are prepared, the step is already described by dense offsets rather than by the original `shift`.

The reader kernel in [lwt_reader.cpp](../tt-wavelet/kernels/dataflow/lwt_reader.cpp) then packs the exact source window needed by the stencil.
Conceptually, the physical output is computed as:

$$
\text{out}[p] = \text{base}[\text{base\_offset} + p] +
\sum_{j=0}^{k-1} h[j] \cdot \text{source}[\text{source\_offset} + p + k - 1 - j]
$$

That is why the reader uses:

```cpp
source_offset + source_logical_index
```

for the source stream, and:

```cpp
base_offset + output_base + col
```

for the base stream.

After this packing step, the compute kernel in [lwt_compute.cpp](../tt-wavelet/kernels/lwt_compute.cpp) only applies a dense stencil:

```cpp
hstencil_row<K>(h_coeffs, kDstInput0, kDstInput1, kDstOutput);
```

So at stencil time, `shift` is already baked into the prepared source window.
The compute kernel only sees aligned input data and coefficients.

>Important note
>`source_left_pad` in the reader path is not the lifting-scheme `shift`.
>It is only a technical left padding used to align the packed source window to the fixed hardware stencil width.

This `17 - k` padding is introduced when runtime args for the predict/update reader are prepared:

```cpp
reader_rt[off + 8] = stencil_source_left_pad(bundle.plan.packed_steps[i]);
```

where:

```cpp
stencil_source_left_pad(desc) = 17 - k
```

Then it is applied in [lwt_reader.cpp](../tt-wavelet/kernels/dataflow/lwt_reader.cpp) while packing the source tiles:

```cpp
source_logical_index = packed_index - source_left_pad;
```

So this padding appears at the reader-packing stage, just before the data is fed into `hstencil_row<K>(...)`.

For the reason why the horizontal stencil expects this `17 - k` alignment, see [HORIZONTAL_STENCIL.md](./HORIZONTAL_STENCIL.md).


## Concrete example

For the concrete walkthrough below, denote the preprocess outputs by:

$$
e_0 = [5,3,1,2,4,5,3,1]
$$

$$
o_0 = [4,2,1,3,5,4,2]
$$

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

Here the current stream shifts come directly from the initial delays:

- even starts with shift `1`
- odd starts with shift `2`

For this predict step the local step shift is `0`, so the convolution stage uses the current even shift without any additional translation.

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

So this step does not compute a new odd shift as `delay.odd + step.shift`.
Instead it:

1. starts from the current source shift `1`
2. applies the local step shift `0` inside `conv_shift`
3. computes the new logical support through overlap with the base stream

The resulting odd output shift is `out_shift = 2`, and that becomes the current odd shift for the next step.

### Dense compute equation

Using the same packed stencil notation, with `k = 1` we get:

$$
y[p] = o_0[p] - e_0[p+1], \qquad p \in [0,7)
$$

The indices come from the planner values:

- `p` comes from `base_offset + p` with `base_offset = 0`
- `p + 1` comes from `source_offset + p` with `source_offset = 1`

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

Reader runtime block:

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

The `src_left_pad = 16` value comes from:

$$
17 - k = 17 - 1 = 16
$$

So for this step the reader:

- reads source values from `even_ping` starting at dense index `1`
- reads base values from `odd_ping` starting at dense index `0`
- packs the tiles needed by `hstencil_row<1>(...)`

After step 0:

- `odd_pong` contains the new odd stream `o_1`
- logical odd state becomes:

$$
(\text{shift}=2,\; \text{length}=7)
$$

This `o_1` stream is then used as the source for the update step.


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

Here the current source shift is already `2`, as produced by Step 0.
So this update step does not restart from the original JSON delays.
Instead it takes the current source shift and applies the local step shift `-1` inside the convolution geometry.

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

Again, the new even shift is not computed as `delay.even + step.shift`.
The planner first forms the convolution support from the current source shift and the local step shift, then intersects it with the current base support.
That overlap produces `out_shift = 3`, which becomes the new current even shift.

### Dense compute equation

From the generic packed stencil equation:

$$
\text{out}[p] = \text{base}[\text{base\_offset} + p] +
\sum_{j=0}^{k-1} h[j] \cdot \text{source}[\text{source\_offset} + p + k - 1 - j]
$$

substituting:

- `base = e_0`
- `source = o_1`
- `k = 3`
- `source_offset = 0`
- `base_offset = 2`

gives:

$$
y[p] = e_0[2+p] + \frac{1}{16}o_1[p+2] + \frac{1}{2}o_1[p+1] - \frac{1}{16}o_1[p]
$$

for:

$$
p \in [0,5)
$$

The indices come directly from the planner offsets:

- `2 + p` comes from `base_offset + p`
- `p + 2`, `p + 1`, `p` come from `source_offset + p + k - 1 - j` with `k = 3`

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

Again, the reader performs dense slicing. It does not understand wavelet semantics.
It only knows:

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

This step is not part of the fused predict/update program.
It is executed separately in [lifting/device.cpp](../tt-wavelet/tt_wavelet/src/lifting/device.cpp) using:

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

For scale steps, the local step shift is always `0`.
Unlike predict/update, there is no convolution support to build here.
The scale operation preserves the current stream shift, so this step keeps the even stream at:

$$
(\text{shift}=3,\; \text{length}=5)
$$

### Reader and indices

Unlike the stencil path, the scale path has no convolution geometry.
After step 1, the active even stream already has logical length `5`, so the scale reader simply reads dense values:

$$
e_1[0], e_1[1], e_1[2], e_1[3], e_1[4]
$$

In [lwt_scale_reader.cpp](../tt-wavelet/kernels/dataflow/lwt_scale_reader.cpp), this is done through:

```cpp
logical_index = output_base + col
```

and the value is read if:

```cpp
logical_index < logical_length
```

So the scale reader does not apply `shift` and does not apply stencil padding.
It only reads the active dense prefix of the current stream.

### Device buffers used

Buffers:

- source DRAM buffer: `even_pong`
- output DRAM buffer: `even_ping`

Reader runtime block:

```text
[src_addr=even_pong,
 logical_length=5,
 output_sticks=1,
 scalar_packed=bitcast(sqrt(2))]
```

Important:

- the scale reader emits two tiles:
  - input tile
  - coefficient tile filled with the scalar repeated across row 0
- the multiply kernel is pure FP32 tile multiply
- the writer stores the scaled result back into row-major DRAM

After step 2:

- `even_ping` contains the scaled even stream `e_2`
- logical even state remains:

$$
(\text{shift}=3,\; \text{length}=5)
$$


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

For scale steps, the local step shift is again `0`.
So this step also preserves the current stream shift.
The odd stream therefore stays at:

$$
(\text{shift}=2,\; \text{length}=7)
$$

### Reader and indices

Just like `scale-even`, this path reads the active dense prefix of the current stream.
So the reader consumes:

$$
o_1[0], o_1[1], \dots, o_1[6]
$$

with no additional stencil padding and no new convolution geometry.

### Device buffers used

Buffers:

- source DRAM buffer: `odd_pong`
- output DRAM buffer: `odd_ping`

Reader runtime block:

```text
[src_addr=odd_pong,
 logical_length=7,
 output_sticks=1,
 scalar_packed=bitcast(1/sqrt(2))]
```

After step 3:

- `odd_ping` contains the scaled odd stream `o_2`
- logical odd state remains:

$$
(\text{shift}=2,\; \text{length}=7)
$$

At this point, the final active streams are:

- `even_ping`
- `odd_ping`


## Why ping-pong buffers are used

Each logical family has two physical buffers:

- even: `even_ping`, `even_pong`
- odd: `odd_ping`, `odd_pong`

The planner tracks which slot is currently active for each family.
When a step writes a new version of a stream, it writes into the toggled slot:

```cpp
with_toggled_slot(stream)
```

So the update rule is:

- read from the current active slot
- write to the other slot
- mark the written slot as the new active one

For the running example, the slot flow is:

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
```

This avoids read/write conflicts on the same logical stream.
In particular, a step never overwrites the buffer that is still being used as its input for that step.

### How this is handled in code

The physical buffers are allocated up front in [lifting/device.cpp](../tt-wavelet/tt_wavelet/src/lifting/device.cpp):

```cpp
auto even_ping = create_signal_mesh_buffer(...);
auto even_pong = create_signal_mesh_buffer(...);
auto odd_ping = create_signal_mesh_buffer(...);
auto odd_pong = create_signal_mesh_buffer(...);
```

and stored as:

```cpp
MeshBufferPair{ .ping = ..., .pong = ... }
```

The planner toggles slots with:

```cpp
with_toggled_slot(stream)
```

in [lifting/plan.cpp](../tt-wavelet/tt_wavelet/src/lifting/plan.cpp).
For example:

- predict writes to `with_toggled_slot(active.odd)`
- update writes to `with_toggled_slot(active.even)`

At runtime, the selected slot is converted into the actual backing buffer by:

```cpp
resolve_mesh_buffer(buffers, route.source)
resolve_output_mesh_buffer(buffers, route)
```

and those addresses are packed into the reader and writer runtime argument blocks.
So ping-pong is not just a conceptual model: it is the actual mechanism used to choose `src_addr`, `base_addr`, and `output_addr` for each step.


## Scope for optimization

In the current implementation, intermediate results are materialized back into DRAM-backed `MeshBuffer`s after each step.

So even inside the fused predict/update phase, the flow is still:

```text
reader -> compute -> writer -> DRAM
```

and then the next step reads that updated stream from DRAM again.

For example:

- predict writes `o_1` into `odd_pong`
- update then reads `odd_pong` back as its source

This is why the writer kernels exist even for intermediate streams: they publish the new active version of the stream into its backing buffer, so the next step can address it through the normal `TensorAccessor` path.

This design is simple and makes routing explicit, but it leaves room for optimization.
Potential future improvements include:

- keeping more intermediate streams on-chip instead of round-tripping through DRAM
- fusing more than one logical step into a tighter pipeline
- reducing the number of reader/writer boundaries between consecutive steps
- specializing paths where the next consumer can directly reuse freshly produced data

So the current path prioritizes correctness and clear buffer ownership first, while performance tuning can further reduce these extra DRAM writes and rereads.

### How this is handled in code

In the fused predict/update path, the reader loops over logical steps and, starting from step 1, explicitly waits until the previous DRAM write is published:

```cpp
if (step > 0) {
    cb_wait_front(cb_sync, 1);
    cb_pop_front(cb_sync, 1);
}
```

The writer then stores the produced output tile to the selected DRAM buffer and signals completion through the sync CB:

```cpp
noc_async_write(...)
noc_async_write_barrier()
cb_push_back(cb_sync, 1)
```

So the next step does not consume the previous result directly from the compute CBs.
It consumes it through the backing `MeshBuffer` after the writer has committed it.

This is the reason the current code writes intermediate results to DRAM again: the step-to-step handoff is implemented through buffer publication and reread, not through a fully on-chip chained pipeline.


# Kernel Arg Layout

Kernel arguments in this repo are passed as small flat `uint32_t` blocks, not as a custom `MeshBuffer` or large C++ runtime struct.

- compile-time args are attached in `CreateKernel(...)` and read inside kernels with `get_compile_time_arg_val(i)`
- runtime args are attached with `tt::tt_metal::SetRuntimeArgs(...)` and read with `get_arg_val<uint32_t>(i)`
- TT-Metal owns the actual argument storage; this code just builds `std::array<uint32_t, N>` or `std::vector<uint32_t>`

Tiny example:

```cpp
tt::tt_metal::SetRuntimeArgs(program, reader, core, std::array<uint32_t, 4>{
    static_cast<uint32_t>(src_buffer.address()),
    logical_length,
    output_stick_count,
    scalar_packed,
});

const uint32_t src_addr = get_arg_val<uint32_t>(0);
const uint32_t logical_length = get_arg_val<uint32_t>(1);
```

See:
- [tt-wavelet/tt_wavelet/src/pad_split/device.cpp](tt-wavelet/tt_wavelet/src/pad_split/device.cpp)
- [tt-wavelet/tt_wavelet/src/lifting/device.cpp](tt-wavelet/tt_wavelet/src/lifting/device.cpp)
- [tt-wavelet/kernels/dataflow/lwt_reader.cpp](tt-wavelet/kernels/dataflow/lwt_reader.cpp)

Common layouts:

- `pad_split` reader runtime args:

```text
[src_addr, input_length, padded_length, left_pad]
```

- fused `predict/update` reader runtime args:

```text
[num_steps,
 (src_addr, src_len, base_addr, base_len, out_len, out_sticks,
  src_off, base_off, src_left_pad) * N]
```

- fused `predict/update` writer runtime args:

```text
[num_steps, (output_addr, output_sticks) * N]
```

- `scale` reader runtime args:

```text
[src_addr, logical_length, output_stick_count, scalar_packed]
```

So if you are reading a kernel, the first thing to check is usually:

1. which `SetRuntimeArgs(...)` block builds its runtime layout on the host
2. which `get_arg_val(...)` indices the kernel consumes
3. which `get_compile_time_arg_val(...)` indices are CB IDs, stick sizes, or accessor metadata
