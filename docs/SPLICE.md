# Row tensor to Tiles mapping for stencil kernel

## Problem overview

1D tensor can be seen as an 1xN 2D tensor. Such tensors are usually stored in row-major layout for memory efficiency instead of using Tile layout, where it would have been padded with zeros to have y-dimensions of 32 with waste of 31*N elements.

Nevertheless computation on TensixCoprocessor is done on Tiles, thus requiring us to convert row-major data to Tiles. The one way is to use only first row of the Tile to carry actual data, thus we would need to process N/32 Tiles.

![](./figs/NaiveRowMajor.svg)

On the figure above data will occupy `tile0/face0/dreg0/row0`, `tile0/face1/dreg0/row0`, `tile1/face0/dreg0/row0`, `tile1/face1/dreg0/row0` and so on.

Our [horizontal stencil](./HORIZONTAL_STENCIL.md) processes one DRegister(4x16) at a time. Since only first row of the Tile is active, we can only process 2 top DRegisters per Tile (`face0/dreg0`, `face1/dreg0`), thus we would need to process a total of N/16 DRegisters. Hovewer in that case we only have efficiency of 25%, because 3/4 rows of the DRegister are not used.

## Proposed solution

To improve efficiency we propose the following method for mapping row-major data to Tiles.

As discussed in the [horizontal stencil documentation](./HORIZONTAL_STENCIL.md) we can compute one DRegister of the output using 2 consecutive DRegisters of the input. If we work in terms of rows, to compute 1x16 elements of the output `g[i:i+16]` we need `f[i:i+16]` and `f[i+16:i+32]`. In the figure below you can see how we can map a row tensor of length 208 to 2 Tiles with 2 DRegisters per Tile, which are used at full capacity.

![](./figs/AdvancedRowMajor.svg)

On the figure above `S` is an input signal-array already padded in the left with $17-k$ elements as required by [horizontal stencil](./HORIZONTAL_STENCIL.md) and `G` is the output array of the stencil.

The idea is basically to think of two Tiles as 4x64 tensor, where last 16 elements of the row `i` equal to the first 16 columns of the row `i+1`. In the figure you can see how result of the stencil `G` is computed from signal `S`. By collecting result as shown using arrows we can inverse map the output back to row-major layout.

This idea can be easily extended to more Tiles or DRegisters per Tiles. For example, we can add more Tiles to the right, making the "tensor" wider, or we can add more blocks to the bottom, making the "tensor" taller.

Note that making the "tensor" wider is more efficient, because it does not require increasing the number of copied rows.

## Accepted solution

We have decided to stick with the 2 Tiles with full DRegister usage. Thus we define Splice which is two Tiles stacked together with the above strategy of copying data. Since not every signal can be contained in a single Splice, we define a SpliceChain - a sequence of Splices such that last 16 elements of the previous Splice are equal to the first 16 elements of the next Splice.

## Efficiency analysis

In this way for a Splice with all DRegisters active we can fit 1552 samples of the input signal. Splice consists of 8 Faces, given 4 DRegisters per Face, we have 32 DRegisters in total. Thus we need to process N/776 Tiles or N/48.5 DRegisters, which is a significant improvement over the naive method and the efficiency is around 75.8% (1552/2048).
