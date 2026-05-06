#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t case_count = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t tile_nbytes = get_tile_size(cb_input);
    constexpr auto src_args = TensorAccessorArgs<1>();
    const auto src = TensorAccessor(src_args, src_addr, tile_nbytes);

    for (uint32_t test_case = 0; test_case < case_count; ++test_case) {
        cb_reserve_back(cb_input, 3);
        const uint32_t cb_addr = get_write_ptr(cb_input);
        const uint32_t tile_base = test_case * 3;

        noc_async_read_tile(tile_base + 0, src, cb_addr + 0 * tile_nbytes);
        noc_async_read_tile(tile_base + 1, src, cb_addr + 1 * tile_nbytes);
        noc_async_read_tile(tile_base + 2, src, cb_addr + 2 * tile_nbytes);
        noc_async_read_barrier();

        cb_push_back(cb_input, 3);
    }
}
