# Vertical FP32 SFPU stencil

The production vertical stencil primitive remains available in [`vertical_stencil_sfpi.h`](../tt-wavelet/kernels/sfpi/vertical_stencil_sfpi.h). The standalone verification executable and its test-only kernels are preserved only in `archive/vertical_stencil_test` and are excluded from production builds.

For a column signal `f` and filter `h`, the valid output is:

$$
g[i] = \sum_{j=0}^{K-1} h[j] f[i-j].
$$

The implementation processes the window bottom-to-top. A transpose, chained four-register copy, and transpose back shift four SFPU registers upward by one row:

```cpp
SFPTRANSP();
SFPSHFT2(0, 0, 0, SFPSHFT2_MOD1_SUBVEC_CHAINED_COPY4);
SFPNOP();
SFPTRANSP();
```

![Vertical one-element rotate](figs/VerticalRotate.svg)

Register capacity selects one of three compile-time output heights:

- `K < 6`: 12 valid rows;
- `K < 10`: 8 valid rows;
- `K < 14`: 4 valid rows.

Unlike the horizontal stencil, columns do not need even/odd decomposition or an explicit cross-register masked move. This primitive is not part of the current 1D LWT/ILWT executable, but it is retained as production kernel infrastructure rather than verification-only code.
