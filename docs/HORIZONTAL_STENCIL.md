# Horizontal FP32 SFPU stencil

The production stencil supports compile-time `K=1..17` and computes valid convolution in the lifting arithmetic order:

$$
g[i] = b[i] + \sum_{j=0}^{K-1} h[j] f[i+K-1-j].
$$

The reader supplies `17-K` left alignment positions so the same 17-tap register window works for every route. This is an internal alignment rule, not materialization of an entire padded signal.

## Narrow-page window

One group uses four source `32x16` pages and three base pages. The fourth source page is the first page shifted by one 48-element logical row, supplying the right halo needed by the three output pages. The compute kernel loads even and odd columns into separate SFPU registers and accumulates the two parities independently.

For shifted source `f`, define `f_e[i]=f[2i]` and `f_o[i]=f[2i+1]`. Even coefficients multiply the matching parity. Odd coefficients cross parity, requiring a one-lane shift and a halo from the preceding register:

$$
g_e = b_e + \sum_j h[2j]R_j f_e + \sum_j h[2j+1]R_{j+1}f_o,
$$

$$
g_o = b_o + \sum_j h[2j]R_j f_o + \sum_j h[2j+1]R_jf_e.
$$

The maximum eight shifts between two 8-lane subvectors gives the `K<=17` bound.

![Horizontal stencil register windows](figs/HorizontalStencilExamples.svg)

## Architecture-specific rotate

TT-Metal defines exactly one official JIT macro. [`horizontal_stencil_sfpi.h`](../tt-wavelet/kernels/sfpi/horizontal_stencil_sfpi.h) branches directly on it.

- On Wormhole (`ARCH_WORMHOLE`), `SHFLSHR1` retains the documented erratum behavior used to inject the cross-register lane-zero halo after rotating the preceding register.
- On Blackhole (`ARCH_BLACKHOLE`), that erratum is fixed. Both registers are rotated, lane zero is selected from `LREG15` lane positions, and a masked move injects the halo explicitly.

Never compile the Wormhole rotate for Blackhole.

![Horizontal one-element rotate](figs/HorizontalRotate.svg)

## Scale integration

[`scale_sfpi.h`](../tt-wavelet/kernels/sfpi/scale_sfpi.h) owns FP32 scale primitives. For inline inverse scaling, source and/or base registers are multiplied before stencil accumulation. Forward terminal scaling multiplies the last updated output registers after accumulation. This keeps coefficient multiplication in SFPU and preserves operation order without materializing an avoidable local route.

## Required validation

Exercise `K=1`, `K=2`, shipped `K=9`, and synthetic `K=17`, including aligned and `+1 FP32` offsets, both layouts, odd/even lengths, and the 3,072-element chunk boundary. A successful host build is insufficient because SFPI is JIT compiled on first device use.
