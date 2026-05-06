# Lifting Wavelet Transform over SpliceChain

**Input**: 2x SpliceChain (odd and even), Nx kernels, scheme description
**Output**: 2x SpliceChain (odd and even)

Lifting wavelet transform (LWT) is a method of computing wavelet transform using a sequence of simple steps, called lifting steps. There are 5 lifting steps:

- Predict: $odd = odd + even * kernel$
- Update: $even = even + odd * kernel$
- Swap: $even \leftrightarrow odd$
- ScaleEven: $even = even \cdot c$
- ScaleOdd: $odd = odd \cdot c$

The $*$ operation is a multiplication of two Laurent polynomials. In our case $odd$, $even$ and $kernel$ are Laurent polynomials.

In the code they are stored as a tuple $(\text{array}, \text{degree})$, where degree is a degree of a monomial with the first coefficient from the array.

Array is actually stored as a SpliceChain in memory, except kernels.

### Addition

Addition will map to the lowest degree, so if we have $A + B$, the polynomial with the highest degree will be extended on the left with random (garbage) elements. Not zeros, because at the end global slice will be used, which will remove the elements that were affected by these garbage elements.

This is implemented using [SHIFT](./SHIFT.md) operation with [RECOVER1](./RECOVER1.md) being next to recover Splice format of the data.

Then elementwise addition of Splices of two signals is done, this will be the result of addition, and degree is the $min(a, b)$ of degrees of input signals.

### Convolution/multiplication

Multiplication by constant is trivial (elementwise multiplication).

Multiplication by kernel is done using [HORIZONTAL_STENCIL](./HORIZONTAL_STENCIL.md). The degree of the result is $a + k + d - 1$, where $a$ is the degree of the signal, $d$ is the degree of the kernel and $k$ is the size of the kernel.

Note that convolution is commutative with shifts, because it is LTI. Thus we can do shift and recover1 before the convolution.

At the same time convolution produces only 6 faces of the output, last two (right-most) are garbage, thus we apply [RECOVER2](./RECOVER2.md) to recover Splice format of the output.

### Final tree

In the final we get the following formula for the predict step as an example:

If degree of odd is smaller than degree of even, then:

Apply `SHIFT` to even, then `RECOVER1` to recover Splice format, then `HORIZONTAL_STENCIL` with kernel, then `RECOVER2` to recover Splice format of the output, and add it to odd:

```
// Forward pass
pred <- SHIFT(even, degree of even - degree of odd)
pred <- RECOVER1(pred, tmp)
pred <- HORIZONTAL_STENCIL(pred, kernel)

// Backward pass
pred <- RECOVER2(pred, tmp)
odd <- odd + pred
```

If degree of odd is bigger than degree of even, then:

First apply `HORIZONTAL_STENCIL` with kernel to even, then `RECOVER2` to recover Splice format of the output, then `SHIFT` to the odd, then `RECOVER1` to recover Splice format, and add them add:

```
// Forward pass
odd <- SHIFT(odd, degree of odd - degree of even)
odd <- RECOVER1(odd, tmp)
pred <- HORIZONTAL_STENCIL(even, kernel)

// Backward pass
pred <- RECOVER2(pred, tmp)
odd <- odd + pred
```

But note that `RECOVER2` requires backward pass (reversed processing of SpliceChain), because tmp is passed from the next Splice to the previoues, while `RECOVER1` does it in forward direction.
