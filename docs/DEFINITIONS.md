

- `SingleRegister` (aka `LRegister`/`LReg`, `Register` and `SRegister`/`SReg`) is a 4x8 matrix - an LReg register of SFPU, can be either even columns of `DoubleRegister` or odd columns.
- `DoubleRegister` (aka `DRegister`/`DReg`) is a 4x16 matrix - two `SingleRegister`s (one for even and one for odd columns)
- `Face` is a 16x16 matrix - 4 `DoubleRegister`s stacked vertically
- `Tile` is a 32x32 matrix - 4 `Face`s stored in row-major layout
- `Splice` is a 32x64 matrix - 2 `Tile`s stacked horizontally with last 16 elements of each row being equalt to the first 16 elements of the next row.
- `SpliceChain` (aka `SChain`) is a sequence of `Splice`s such that 16 last elements of the last row of the previous `Splice` are equal to the first 16 elements of the first row of the next `Splice`.
