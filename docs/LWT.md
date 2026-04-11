# Overview

The doc focuses mostly on 1D example of Lifting Wavelet Transform (LWT) but the same principles apply to 2D and 3D cases.

Each lifting scheme can be represented as following json. Json is used for simplicity. For C++ version each lifting scheme will be represented as struct and unrolled during compile time.

The list of all lifting schemes can be view in this directory [lifting-schemes](../ttnn-wavelet/lifting_schemes)

For example we choose [bior1.3](../ttnn-wavelet/lifting_schemes/bior1.3.json)

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
            "coefficients": [-1.0]
        },
        {
            "type": "update",
            "shift": -1,
            "coefficients": [0.0625, 0.5, -0.0625]
        },
        {
            "type": "scale-even",
            "shift": 0,
            "coefficients": [1.4142135623730951]
        },
        {
            "type": "scale-odd",
            "shift": 0,
            "coefficients": [0.7071067811865475]
        }
    ]
}
```
The scheme consists of 4 steps:

Predict step

Update step

Scale-even step is always multiplication by scalar

Scale-odd step is always multiplication by scalar

Swap step can be present in some schemes, which swaps the even and odd samples


## Padding (Physical extension)

To perform convolution at the boundaries of the signal, we extend the signal beyond its original length.
We use symmetric extension.

Example (NumPy / PyWavelets)

```python
import numpy as np

signal = [1, 2, 3, 4, 5]
print(np.pad(signal, (2, 2), mode='symmetric'))
```

```
[2 1 1 2 3 4 5 5 4]
```

The signal is extended by mirroring values at both boundaries.


## Padding length

Padding length is defined as

$$
P = \text{tap \ size} - 1
$$

For bior1.3

$$
\text{tap \ size} = 6
$$

therefore

$$
P = 5
$$

The padded signal becomes

```
x =
[5 4 3 2 1 | 1 2 3 4 5 | 5 4 3 2 1]
```


## Even Odd Split

Even and odd split can be represented with the following code

```python
import numpy as np

signal = [1, 2, 3, 4, 5]
padded = np.pad(signal, (5, 5), mode='symmetric')
print(padded)
even = padded[::2]
odd  = padded[1::2]
print(even)
print(odd)
```

After padding the signal is split into two parity streams.

Given padded signal

$$
x[n]
$$

even stream

$$
e[n] = x[2n]
$$

odd stream

$$
o[n] = x[2n+1]
$$


### Example

```
x = [5 4 3 2 1 1 2 3 4 5 5 4 3 2 1]
```

Even samples

```
e = [5 3 1 2 4 5 3 1]
```

Odd samples

```
o = [4 2 1 3 5 4 2]
```


### Index mapping

$$
x[2n]   \rightarrow e[n]
$$

$$
x[2n+1] \rightarrow o[n]
$$



## Predict Step

The predict step updates odd samples using a stencil computed from the even samples

$$
d[n]=o[n]+\sum_{k=0}^{K-1}h_ke[n + s + k]
$$

where $o[n]$ are odd samples, $e[n]$ are even samples, $h_k$ are predict coefficients,
$K$ is the number of stencil taps, $s$ is the shift from the lifting scheme, and $n$ is the sample index.

This step produces the detail coefficients $d[n]$.



## Update Step

The update step modifies the even samples using the predicted detail values

$$
a[n]=e[n]+\sum_{k=0}^{K-1}g_kd[n + s + k]
$$

where $e[n]$ are even samples, $d[n]$ are detail coefficients from the predict step,
$g_k$ are update coefficients, $K$ is the number of stencil taps, and $s$ is the shift.

The update step depends on the output of the predict step, therefore predict must be executed first.



## General Stencil Form

Both predict and update steps share the same stencil structure

$$
f[n]=\text{base}[n]+\sum_{k=0}^{K-1}h_k\text{signal}[n+\delta_k]
$$

where $\text{base}$ is the value being updated, $\text{signal}$ is the input stream,
$h_k$ are stencil coefficients, $K$ is the number of taps, and $\delta_k = s + k$ are stencil offsets.

Predict

$$
\text{base} = o,
\qquad
\text{signal} = e,
\qquad
f = d
$$

Update

$$
\text{base} = e,
\qquad
\text{signal} = d,
\qquad
f = a
$$

## Other steps

For the given bior1.3 scheme, the remaining steps are scaling steps which simply multiply the even and odd streams by a scalar factor.

$$
a[n] \leftarrow a[n] \cdot s_e
$$

$$
d[n] \leftarrow d[n] \cdot s_o
$$

where $s_e$ and $s_o$ are the scaling factors for even and odd streams respectively.


# TT-Metal Path

For 1D LWT we use custom sfpi 1D horizontal stencil function defined here [sfpi](../tt-wavelet/kernels/stencil_sfpi.h)

>[!NOTE]
>The sfpi is currently in WIP it produces incorrect results. But the idea should reamin the same.

The code is now splitted between two set of files

Fisrt performs pad + split for even odd samples of the signal but does not call sfpi function.


Second is much more simplified to test the sfpi kernel. It performs our row major tilization which I will introduce below and then call sfpi function to perform only one step from the lifting scheme. The rest of the steps are not implemented yet.

## Row Major Layout

Here we consider the algorithm in row major layout. Even thought the whole path almost works in row-major for computation before calling SFPU the tile layout is required by hardware.

Let`s view the row major layout from the perspective of the next files

- [main](../tt-wavelet/stencil.cpp) - here host tilization is performed:

The sfpi kernel has the next calling convention

```cpp
calculate_stencil(
    const uint32_t h_packed[K], const uint input1, const uint input2, const uint output, const uint rows = 32)
```

Where `h_packed` is uint32_t coefficients of the step which being bit casted to floats.

- input1 - Halo dst index
- input2 - Input dst signal index
- output - Output dst signal index

So in main we create CirrucularBuffers for Halo, Input and Output. Each of them stores default tiles 32x32 of fp32.

The signal is already padded for 17 - K from the left. K stands for the length of each coefficients in the step. For bior1.3 K = 3 for update step

```json
        {
            "type": "update",
            "shift": -1,
            "coefficients": [0.0625, 0.5, -0.0625]
        },
```

Given some signal with length 32

```
x = [1, 2, 3, ..., 32]
```

The padded signal becomes

Halo
```
Halo:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, ]
```

Input
```
Input:[19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
```

They are aligned for 32 elements to fit in tile

In tile layout they will be represented as

```
Halo:
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, ]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
...
```
So the first row of the tile contains the first 32 elements of the halo, the second row and all others contains only zeros. The same applies to the input tile.

If we view on tile as faces each face has 16 rows
so in this layout our signal will be store in first two faces
face0row face1row0

For tilization we use

```cpp
auto tiled_halo = tilize_nfaces(halo_tile, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH);
auto tiled_input = tilize_nfaces(input_tile, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH);
```

This provides next row majsor layout

face0 row 0 view

```
Tiled Halo:
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2] - 16 elems
```

The next 15 row is zeros
15 rows $\times$ 16 elems = 240 zeros

and face1 row 0 view
after 240 zeros

```
[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
```

### Where padding `17 - k` comes from

The kernel computes a causal stencil

$$
g[i] = \sum_{j=0}^{k-1} h[j] \cdot f[i-j]
$$

but the original (unpadded) signal is $s$. We build the working signal as

$$
f = R_{17-k}s,
\qquad
f[i] = s[i-(17-k)]
$$

so

$$
g[i]
= \sum_{j=0}^{k-1} h[j] \cdot s[i-j-(17-k)]
$$

The first output index that can touch the first real sample $s[0]$ is obtained from

$$
i - (k-1) - (17-k) = 0
\Rightarrow i = 16
$$

So the first 16 outputs are halo-influenced warmup values, and valid signal-aligned output starts at index 16.
This is why we choose left padding `17 - k`: for any stencil size $1 < k < 18$, the kernel keeps the same alignment rule (valid region starts after the first 16 elements).

For the running example with $k=3$, the left pad is $17-3=14$, which is exactly why the halo begins with 14 zeros before `[1, 2, ...]`.

### How kernel code builds a cyclic shift operator

In [stencil_sfpi.h](../tt-wavelet/kernels/stencil_sfpi.h), the macro `STENCIL_ROTATE(a_reg, b_reg)` implements a 1-step right shift on two adjacent column groups.

At ISA level it is done in two phases:

1. `TTI_SFPSHFT2(..., SHFLROR1)` rotates each register cyclically (wrap-around inside the subvector).
2. A lane mask (prepared in `calculate_stencil_init`) enables only column 0, then `TTI_SFPMOV` patches column 0 of `b_reg` using data from `a_reg`.

Conceptually, this is a "cyclic primitive + boundary fix" pattern:

- the hardware gives cyclic rotation,
- the mask+move step fixes the seam between neighboring chunks,
- the result behaves like a 1-element shift across the pair `(a_reg, b_reg)` for the active output chunk.

Inside `calculate_stencil_body`, every odd tap performs this rotate before accumulating the odd-offset term. Repeating that per odd tap produces the needed $R_j$ offsets from the math section, while even taps use the current (non-shifted for that parity) registers directly.

So the code path mirrors the formulas:

$$
g_e = \sum h[2j]R_j f_e + \sum h[2j+1]R_{j+1}f_o,
\qquad
g_o = \sum h[2j]R_j f_o + \sum h[2j+1]R_j f_e
$$

with `ROTATE` being the concrete implementation of the incremental shift operator on device registers.

### Notation reference

In the formulas above:

- $f$ is the full input stream.
- $f_e$ is the even-index stream: $f_e[n] = f[2n]$.
- $f_o$ is the odd-index stream: $f_o[n] = f[2n+1]$.
- $g$ is the stencil output stream.
- $g_e$ is the even output: $g_e[n] = g[2n]$.
- $g_o$ is the odd output: $g_o[n] = g[2n+1]$.
- $h[t]$ is the stencil coefficient at tap index $t$.
- $k$ is the stencil length (number of taps), with $1 < k < 18$ in this kernel.
- $j$ is the summation index over tap pairs.
- $R_m$ is a right-shift operator by $m$ samples: $(R_m x)[i] = x[i-m]$.

This is exactly what the kernel implements: odd-tap iterations apply one incremental rotate step so the register view matches the required $R_j$ / $R_{j+1}$ offsets.


That is why the following row-major layout was selected.

So for example for signal of length 1024 there will be 32 tiles with only first row of signal.


## Whole LWT Path


1. Given signal of for example length 1024 (we split the signal onto batches of 32 samples 1024 / 32 = 32 batches) but we treat it as the whole signal

2. Physical Wavelet extension is applied [(symmetric padding)](#padding-physical-extension) but only for the whole signal not each batch separately. So for example for bior1.3 with tap size 6 we will have 5 elements of padding on the left and 5 elements of padding on the right of the whole signal.

3. Split signal into even and odd samples [(even odd split)](#even-odd-split) but again for the whole signal not for each batch separately. So we will have two streams of length 517 (512 from the original signal and 5 from padding) for even and odd samples.

4. Iterate thourgh the steps (here we mostly consider predict update steps, because scale and swap are simple operations)



```json
"steps": [
        {
            "type": "predict",
            "shift": 0,
            "coefficients": [-1.0]
        },
        {
            "type": "update",
            "shift": -1,
            "coefficients": [0.0625, 0.5, -0.0625]
        },
        {
            "type": "scale-even",
            "shift": 0,
            "coefficients": [1.4142135623730951]
        },
        {
            "type": "scale-odd",
            "shift": 0,
            "coefficients": [0.7071067811865475]
        }
    ]
```

For the predict step in this example, the coefficient length is 1:

```json
{
    "type": "predict",
    "shift": 0,
    "coefficients": [-1.0]
}
```

So $K = 1$, and the generic predict formula

$$
d[n] = o[n] + \sum_{k=0}^{K-1} h_k\,e[n+s+k]
$$

reduces to

$$
d[n] = o[n] + h_0\,e[n+s].
$$

For bior1.3 predict, $s = 0$ and $h_0 = -1$, therefore

$$
d[n] = o[n] - e[n].
$$

Interpretation: for each index $n$, we take one odd sample and subtract the aligned even sample.
In this special case there is no multi-tap window and no extra tap-dependent shift term.

For the update step, the coefficient length is 3:

```json
{
    "type": "update",
    "shift": -1,
    "coefficients": [0.0625, 0.5, -0.0625]
}
```
Here we need to perform a 3-tap stencil with coefficients $h_0 = 0.0625$, $h_1 = 0.5$, $h_2 = -0.0625$

and shift $s=-1$. So the update formula becomes

$$
a[n] = e[n] + 0.0625\,d[n-1] + 0.5\,d[n] - 0.0625\,d[n+1].
$$

For this update step, the input stream is details $d$.
So in the SFPU path we apply the same pre-shift rule to details:

$$
d_{\text{work}} = R_{17-K}d,
\qquad
d_{\text{work}}[i] = d[i-(17-K)].
$$

Then each tap multiplies a coefficient by the shifted details element:

$$
h[t]\cdot d_{\text{work}}[i-t]=
h[t]\cdot d[i-t-(17-K)].
$$

In simple words: because this is the update step, coefficients are multiplied by details, and those details must be pre-shifted by $17-K$ (implemented via left halo).
For this example $K=3$, so the required pre-shift is $17-3=14$.

As a concrete example, if details are

$$
d = [1, 2, \dots, 32]
$$

and $K=3$, the reader prepares the following halo/input split:


```
Halo:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, ]

Input:[19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
```

To summarize, the main bottleneck is stencil padding, and in practice it must be handled by a fused reader kernel.

Important ordering note: before the per-step stencil loop starts, the reader should do a one-time preprocessing pass:

1. Apply physical (symmetric) padding to the full signal.
2. Split the padded signal into even and odd streams.

In theory, this should also be part of the fused reader kernel path (initial stage), executed once per signal.
After that, predict/update steps run iteratively on the produced streams.

For large schemes (for example [coif17](../ttnn-wavelet/lifting_schemes/coif17.json) with `tap_size = 102`), the complexity comes from many predict/update iterations.
In each iteration we have a step-specific stencil:

$$
K_r = |\text{coefficients}_r|,
\qquad
p_r = 17 - K_r,
\qquad
x_r^{\text{work}} = R_{p_r}x_r.
$$

So the reader must be reconfigured per step (or at least per step pattern):

1. Select which stream is the stencil input for this step (`e` for predict, `d` for update).
2. Build halo for the current $p_r = 17-K_r$ and the step shift.
3. Feed tiles to compute as `halo + input` chunks.

Even when $K_r$ is often 2 in coif17 (so $p_r=15$ for many steps), this is still iterative because source streams and step shifts change, and each new step consumes results from the previous one.

Dataflow-wise, this forms a repeated pipeline:

1. Reader loads/constructs `CB_in` (with halo-adjusted data for current step).
2. Compute loads from `CB_in` to `Dst`, runs stencil, writes `Dst` result.
3. Compute packs `Dst` result to `CB_out`.
4. Writer commits output, and that output becomes input for the next step.

In short, the runtime loop is a ping-pong cycle like:

`CircularBuffer(input) -> Dst compute -> CircularBuffer(output) -> next-step input`.


```json
        {
            "type": "predict",
            "shift": -1,
            "coefficients": [
                -0.6628758009034809,
                0.5789465132124677
            ]
        },
        {
            "type": "update",
            "shift": 0,
            "coefficients": [
                -1.571044102359109,
                1.524344121543015
            ]
        },
        {
            "type": "predict",
            "shift": -1,
            "coefficients": [
                -0.5933442736062721,
                0.4921298057893261
            ]
        },
        {
            "type": "update",
            "shift": 0,
            "coefficients": [
                -1.7615108792441594,
                1.2033282051334329
            ]
        },
        {
            "type": "predict",
            "shift": -1,
            "coefficients": [
                -0.6252793863255618,
                0.40804332140548294
            ]
        },
```

This coif17 fragment requires multiple repeated stencil passes, where each pass reuses previous-step output and rebuilds padding/halo according to the current step configuration.

