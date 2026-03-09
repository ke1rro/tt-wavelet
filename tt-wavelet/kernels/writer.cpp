// tt_metal/programming_examples/eltwise_sfpu/kernels/dataflow/write_tile.cpp
#include <cstdint>

void kernel_main() {
    uint32_t c_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);

    // The circular buffer that we are going to read from and write to DRAM
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out0);

    // Address of the output buffer
    constexpr auto out0_args = TensorAccessorArgs<0>();
    const auto out0 = TensorAccessor(out0_args, c_addr, tile_size_bytes);

    // Loop over all the tiles and write them to the output buffer
    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_out0, 1);
        uint32_t cb_out0_addr = get_read_ptr(cb_out0);
        // write the tile to DRAM
        noc_async_write_tile(i, out0, cb_out0_addr);
        noc_async_write_barrier();
        // Mark the tile as consumed
        cb_pop_front(cb_out0, 1);
    }
}