# Definitions

- **Single register** (`LReg` or `SReg`): a `4x8` SFPU register. It holds either the even or odd columns of a double register.
- **Double register** (`DReg`): a `4x16` matrix formed by two single registers.
- **Face**: a `16x16` matrix formed by four vertically stacked double registers.
- **Tile**: a `32x32` matrix formed by four faces in TT tile order.
- **Narrow tile**: a `32x16` page formed by two faces. FP32 LWT compute uses this native page shape.
- **Stick**: one `1x32` row, or 128 bytes in FP32.
- **Half-stick**: one `1x16` row, or 64 bytes in FP32.
- **Group**: 1,536 logical stream values, represented by three FP32 narrow tiles.
- **Workspace slot**: one of the three local `A`, `B`, and `Scratch` streams owned by a worker core.
