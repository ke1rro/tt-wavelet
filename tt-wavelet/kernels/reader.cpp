// tt_metal/programming_examples/eltwise_sfpu/kernels/dataflow/read_tile.cpp
#include <cstdint>

void kernel_main() {
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;

    const uint32_t tile_size_bytes = get_tile_size(cb_in0);
    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, in0_addr, tile_size_bytes);

    // Read in the data from the source buffer and write to the circular buffer
    // in a loop.
    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_reserve_back(cb_in0, 1);
        uint32_t cb_in0_addr = get_write_ptr(cb_in0);
        noc_async_read_tile(i, in0, cb_in0_addr);

        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);
    }
}
