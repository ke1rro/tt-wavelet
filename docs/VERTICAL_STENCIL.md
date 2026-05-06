# SFPU kernel for vertical stencil

The following derivation and implementation is closely related to described in the [horizontal stencil](./HORIZONTAL_STENCIL.md). The main difference is that we are working with columns instead of rows and stencil computation is done in reversed order (bottom-to-top) rather than top-to-bottom. This is related to hardware limitations and capabilities.

The following assumptions are made:
- s

The formula for the stencil is following (g and f are chosen columns).

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
g = \sum_{j=0}^{k-1} h[j] \cdot R_j f = \sum_{j=-(k-1)}^{0} h[-j] \cdot R_{-j} f
$$

That is, the stencil is a linear combination of shifted versions of the input signal $f$.

But note that we compute valid stencil only, thus we take only $g'[i] = g[i + (k-1)] = (R_{-(k-1)}g)[i]$ and thus we can rewrite the formula as follows:

$$
g' = R_{-(k-1)} \sum_{j=-(k-1)}^{0} h[-j] \cdot R_{-j} f = \sum_{j=-(k-1)}^{0} h[-j] \cdot R_{-(k-1)-j} f = \sum_{j=0}^{k-1} h[(k-1)-j] \cdot R_{-j} f
$$

## Implementation details

There are 3 similar kernels for vertical stencil, that differ by the size of the filter that we can apply and as a consequence the number of rows that we can process at a time. The kernels are the following:
- $k < 6$: $g'[i:i+12] = T(f[i:i+16], h)$
- $k < 10$: $g'[i:i+8] = T(f[i:i+16], h)$
- $k < 14$: $g'[i:i+4] = T(f[i:i+16], h)$

In this documentation we only focus on the case $k < 10$, but the other cases can be derived in a similar way.

In the Tenstorrent setup we work with SRegisters (4x8), thus we can think about processing 4 elements of a column at a time. Kernel takes 4 such consecutive SRegisters and outputs (in case of $k<6$) 2 consecutive SRegisters of the stencil output.

As opposed to the horizontal stencil, we define $R_{-1}$ instead of $R_1$, a shift backward, which we will call `ROTATE()`, this operation rotates 4 first SRegisters, using following instructions:

```c++
SFPTRANSP();
SFPSHFT2(0, 0, 0, SFPSHFT2_MOD1_SUBVEC_CHAINED_COPY4);
SFPNOP();
SFPTRANSP();
```

In the figure below, the above function is visualized.

![](./figs/VerticalRotate.svg)

Recall the rewritten stencil formula:

$$
g'[i:i+8] = \sum_{j=0}^{k-1} h[(k-1)-j] \cdot (R_{-j} f)[i:i+8]
$$

Negative shifts require $k-1$ additional rows of the input column, so we need $f[i:i+8+(k-1)]$ as input. Since we assumed that $k<10$, we need at most $f[i:i+8+(9-1)] = f[i:i+16]$ as input, which is exactly what we get as an input of an operator.

By doing $R_{-1}$ shift we can get $R_{-j}$ for $j=0,1,...,k-1$ and thus compute the stencil output as a linear combination of these shifted versions of the input column.

The code of the kernel is as follows:

```c++
// Note: f0, f1, f2 and f3 are LReg0, LReg1, LReg2 and LReg3 respectively, as required by the SFPTRANSP
for (uint8_t j = 0; j < k; j++) {
    BROADCAST(h[(k-1)-j], tmp);
    SFPMAD(f0, tmp, g0, 0);
    SFPMAD(f1, tmp, g1, 0);
    ROTATE();
}
```

With `BROADCAST(c, r)` defined as follows:
```c++
SFPLOADI(r, SFPLOADI_MOD0_UPPER, c >> 16);
SFPLOADI(r, SFPLOADI_MOD0_LOWER, c & 0xffff);
```

Note that the vertical stencil does not require handling even/odd columns explicitly (DRegisters are processed as a set of SRegisters), does not require additional padding due to bottom-to-top processing and does not require masked movements, thus is more efficient than the horizontal stencil.
