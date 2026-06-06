

- `SingleRegister` (aka `LRegister`/`LReg`, `Register` and `SRegister`/`SReg`) is a 4x8 matrix - an LReg register of SFPU, can be either even columns of `DoubleRegister` or odd columns.
- `DoubleRegister` (aka `DRegister`/`DReg`) is a 4x16 matrix - two `SingleRegister`s (one for even and one for odd columns)
- `Face` is a 16x16 matrix - 4 `DoubleRegister`s stacked vertically
- `Tile` is a 32x32 matrix - 4 `Face`s stored in row-major layout
- `Stick` is 1x32 matrix - one row of a `Tile`

- `HStick` or `HalfStick` is 1x16 matrix - one row of a `Face`


