# Shift operation

**Input**: 1x Splice, K (number of elements to shift)
**Output**: 1x Splice

Shift operation shifts each row of the splice to the right by a given number of elements $K$. It assumes $K < 16$. The shifted elements are filled with garbage values.

This is implemented using the idea of `_horizontal_stencil_rotate` from [HORIZONTAL STENCIL](./HORIZONTAL_STENCIL.md) documentation. It is basically applied to each DRegister of the Splice.

This operation is implemented using SFPU instructions and works over one Splice in the Dst register and does shift operation in-place.

Useful SFPI instructions:
- SFPLOAD
- SFPSTORE
- SFPMOV (masked)
- SFPSETCC
- SFPENCC
- SFPSFHT2
- SFPNOP
