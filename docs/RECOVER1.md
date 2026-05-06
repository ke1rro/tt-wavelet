# Recover1 operation

**Input**: 1x Splice, 1x DRegister (template)
**Output**: 1x Splice, 1x DRegister (template)

This operation assumes execution over SpliceChain from left to right and stores additional data in template to be used in latter Splices.

This operation is used to recover Splice format of the data after SHIFT operation, which produces Splice with only 6 valid faces, and first two faces are garbage (in the first column).

It takes the last two faces (last column), shifts it down by 1 using `_vertical_stencil_rotate` three times, which is described in [VERTICAL STENCIL](./VERTICAL_STENCIL.md) documentation, and stores the result in the first two faces of the Splice.

The temporary DRegister is copied into the first DRegister of the Splice, but only first row is active. So first 16 elements of the Splice are copied from temporary DRegister. At the same time, the last 16 elements of the last DRegister of a Splice (which are out of the boundary after shift down) are copied into first 16 elements of the temporary DRegister to be used later.

Useful SFPI instructions:
- SFPLOAD
- SFPSTORE
- SFPMOV (masked)
- SFPSETCC
- SFPENCC
- SFPTRANSP
- SFPSFHT2
- SFPNOP
