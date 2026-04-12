# SFPU kernel for horizontal stencil

The stencil on SFPI assumes the following:
- The stencil filter $h$ has a length $1 < k < 18$.
- The input signal $f$ is padded with $17-k$ (any) elements on the left and on the right to fit into tile sizes.

Stencil uses the following formula:

$$
g[i] = \sum_{j=0}^{k-1} h[j] \cdot f[i-j]
$$

## Mathematical background

Define linear operator $R_k$ as follows, that is the shift to the right by $k$ elements:

$$
(R_kf)[i] = f[i - k]
$$

Then we can rewrite the stencil formula as follows:

$$
g = \sum_{j=0}^{k-1} h[j] \cdot R_j f
$$

That is, the stencil is a linear combination of shifted versions of the input signal $f$.

Taking into account the hardware limitations (i.e. processing even and odd columns seperately), we can split the stencil into two parts: one for even columns and one for odd columns.

Define: $f_e[i] = f[2i]$ and $f_o[i] = f[2i+1]$. Then we can rewrite the stencil as follows:

$$
g[2i] = \sum_{j=0}^{k-1} h[j] \cdot f[2i-j] = \sum_{j=0}^{\lfloor (k-1)/2 \rfloor} h[2j] \cdot f[2i-2j] + \sum_{j=0}^{\lfloor (k-2)/2 \rfloor} h[2j+1] \cdot f[2i-(2j+1)] =
\sum_{j=0}^{\lfloor (k-1)/2 \rfloor} h[2j] \cdot f_e[i-j] + \sum_{j=0}^{\lfloor (k-2)/2 \rfloor} h[2j+1] \cdot f_o[i-j-1]
$$

So

$$
g_e = \sum_{j=0}^{\lfloor (k-1)/2 \rfloor} h[2j] \cdot (R_j f_e)[i] + \sum_{j=0}^{\lfloor (k-2)/2 \rfloor} h[2j+1] \cdot (R_{j+1} f_o)[i]
$$

With the same reasoning we can derive the formula for odd columns:

$$
g_o = \sum_{j=0}^{\lfloor (k-1)/2 \rfloor} h[2j] \cdot (R_j f_o)[i] + \sum_{j=0}^{\lfloor (k-2)/2 \rfloor} h[2j+1] \cdot (R_j f_e)[i]
$$


### Padding

As was mentioned for our original signal $s$ we have that: $f = R_{17-k} s$. Consider $g[16]$:

$$
g[16] = \sum_{j=0}^{k-1} h[j] \cdot f[16-j] = \sum_{j=0}^{k-1} h[j] \cdot s[k-1-j]
$$

So when you substitute $j=k-1$ you get $s[0]$. That means that the actual result of stencil between $s$ and $h$ is $R_{16}g' = g$. I.e. the $g$ is the result of the stencil between $s$ and $h$ shifted to the right by 16 elements. To compensate for this, we skip the first 16 elements (first face) of the output and start computing directly from the second face ($g[t], t>16$), and what you get as the output is $g'$.

## Implementation details

Our kernel is the operator $g'[i:i+16] = g[i+16:i+32] = T(f[i:i+16], f[i+16:i+32], h)$, where $g$ and $f$ are 4 consecutive rows of the tensor (refer to the [Tenstorrent ISA Documentation]() for details on the SFPU registers and the tile layout).

We have implemented a 1-element shift operator for two consecutive stacks of columns (a, b). It produces a valid results for the second stack (b) while first stack is used as a halo for the elements to be shifted out to the right. You can do the same shift up to 8 elements for the same stacks (a, b).

It is implemented as follows (call it `ROTATE(a, b)`):

```c++
// Shift a to the right
SFPSHFT2(0, a, a, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
// Shift b to the right
SFPSHFT2(0, b, b, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
// Set LaneEnable=true for lanes with col=0 and LaneEnable=false for others
SFPSETCC(0, mask, 0, SFPSETCC_MOD1_LREG_NE0);
// Copy first column of a to the first column of b
SFPMOV(0, a, b, 0);
// Set all LaneEnable=true
SFPENCC(0, 0, 0, SFPENCC_MOD1_EU_R1);
```

The register `mask` is preloaded with the following values:

$$
\text{mask} = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

It can also be viewed as in figure below.

![](./figs/HorizontalRotate.svg)

The whole algorithm is implemented by alternating between doing the shift on even and odd columns and doing the multiplication and reduction for the stencil formula.

Recall the formulas:

$$
g_e = \sum_{j=0}^{\lfloor (k-1)/2 \rfloor} h[2j] \cdot (R_j f_e)[i] + \sum_{j=0}^{\lfloor (k-2)/2 \rfloor} h[2j+1] \cdot (R_{j+1} f_o)[i]
$$
$$
g_o = \sum_{j=0}^{\lfloor (k-1)/2 \rfloor} h[2j] \cdot (R_j f_o)[i] + \sum_{j=0}^{\lfloor (k-2)/2 \rfloor} h[2j+1] \cdot (R_j f_e)[i]
$$

It is implemented as follows (call it `STENCIL(f_e_0, f_o_0, f_e_1, f_o_1, g_e, g_o)`, with `h` being a compile-time (for kernel) array)
```c++
for (uint8_t j = 0; j < k; j++) {
    BROADCAST(h[j], tmp);
    if((j & 1) == 0) {
        // Even
        SFPMAD(f_e_1, tmp, g_e, 0);
        SFPMAD(f_o_1, tmp, g_o, 0);
    } else {
        // Odd
        SFPMAD(f_e_1, tmp, g_o, 0);
        ROTATE(f_o_0, f_o_1);
        SFPMAD(f_o_1, tmp, g_e, 0);
        ROTATE(f_e_0, f_e_1);
    }
}
```

With `BROADCAST(c, r)` defined as follows:
```c++
SFPLOADI(r, SFPLOADI_MOD0_UPPER, c >> 16);
SFPLOADI(r, SFPLOADI_MOD0_LOWER, c & 0xffff);
```

### Processing tiles

Above code works for two 4x16 blocks of the tile and outputs one 4x16 block of the output. On the figure below can be seen how choice of the input blocks (blue) influences where output block (red) is located. By sliding over the input blocks horizontally we compute whole tile of the output.

![](./figs/HorizontalStencilExamples.svg)

From tile 0 and tile 1 of the input, we compute tile 0 of the output. Then we can either add tile 2, and compute stenctil over tile 1 and tile 2 to produce tile 1 of the output, or we can just compute final stemcil over tile 1, whcih will produce half of the tile 1, with block 0 of face 1 (as show of figure) will not be valid.
