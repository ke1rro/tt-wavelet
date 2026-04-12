# Row tensor to tile mapping for stencil kernel

## Problem overview

1D tensor can be seen as an 1xN 2D tensor. Such tensors are usually stored in row-major layout for memory efficiency instead of using tile layout, where it would have been padded with zeros to have y-dimensions of 32 with waste of 31*N elements.

Nevertheless computation on TensixCoprocessor is done on tiles, thus requiring us to convert row-major data to tiles. The one way is to use only first row of the tile to carry actual data, thus we would need to process N/32 tiles.

![](./figs/NaiveRowMajor.svg)

On the figure above data will occupy `tile0/face0/block0/row0`, `tile0/face1/block0/row0`, `tile1/face0/block0/row0`, `tile1/face1/block0/row0` and so on.

Our [horizontal stencil](./HORIZONTAL_STENCIL.md) processes 4x16 blocks at a time. Sine only first row of the tile is active, we can only process 2 blocks per tile (`face0/block0`, `face1/block0`), thus we would need to process a total of N/16 blocks. Hovewer in that case we only have efficiency of 25%, because 3/4 rows of the block are not used.

## Proposed solution

To improve efficiency we propose the following method for mapping row-major data to tiles.

As discussed in the [horizontal stencil documentation](./HORIZONTAL_STENCIL.md) we can compute one block of the output using 2 consecutive blocks of the input. If we work in terms of rows, to compute 1x16 elements of the output `g[i:i+16]` we need `f[i:i+16]` and `f[i+16:i+32]`. In the figure below you can see how we can map a row tensor of length 208 to 2 tiles with 2 blocks per tile, which are used at full capacity.

![](./figs/AdvancedRowMajor.svg)

The idea is basically to think of two tiles as 4x64 tensor, where last 16 elements of the row `i` equal to the first 16 columns of the row `i+1`. In the figure you can see how result of the stencil `G` is computed from signal `S`. By collecting result as shown using arrows we can inverse map the output back to row-major layout.

This idea can be easily extended to more tiles or blocks per tiles. For example, we can add more tiles to the right, making the "tensor" wider, or we can add more blocks to the bottom, making the "tensor" taller.

Note that making the "tensor" wider is more efficient, because it does not require increasing the number of copied rows.

## Efficiency analysis

In this way for 2 tiles with all blocks active we can fit 1552 samples of the input signal. 2 full tiles consist of 8 face, given 4 blocks per face, we have 32 blocks in total. Thus we need to process N/776 tiles or N/48.5 blocks, which is a significant improvement over the naive method and the efficiency is around 75.8% (1552/2048).

By increasing number of tiles (horizontal scaling) for example to 4, we can fit 3600 samples of the input signal, which gives us efficiency of around 87.9% (3600/4096).

## Implementation details

Work in progress, stay tuned!
