# Recover2 operation

**Input**: 1x Splice, 1x DRegister (template)
**Output**: 1x Splice, 1x DRegister (template)

This operation assumes execution over SpliceChain from right toleft and stores additional data in template to be used in leftmost Splices.

This operation is used to recover Splice format of the data after STENCIL/CONVOLUTION operation, which produces Splice with only 6 valid faces, and last two faces are garbage (in the last column).

It takes the first two faces (first column), shifts it up by 1 using `_vertical_stencil_rotate` one time, which is described in [VERTICAL STENCIL](./VERTICAL_STENCIL.md) documentation, and stores the result in the last two faces of the Splice.

The temporary DRegister is copied into the last DRegister of the Splice, but only last row is active. So last 16 elements of the Splice are copied from temporary DRegister. At the same time, the fisrt 16 elements of the first DRegister of a Splice (which are out of the boundary after shift up) are copied into last 16 elements of the temporary DRegister to be used later.

Useful SFPI instructions:
- SFPLOAD
- SFPSTORE
- SFPMOV (masked)
- SFPSETCC
- SFPENCC
- SFPTRANSP
- SFPSFHT2
- SFPNOP
