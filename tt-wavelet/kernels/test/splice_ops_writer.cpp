#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t case_count = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_output = get_compile_time_arg_val(0);
    constexpr uint32_t tile_nbytes = get_tile_size(cb_output);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    const auto dst = TensorAccessor(dst_args, dst_addr, tile_nbytes);

    for (uint32_t test_case = 0; test_case < case_count; ++test_case) {
        cb_wait_front(cb_output, 3);
        const uint32_t cb_addr = get_read_ptr(cb_output);
        const uint32_t tile_base = test_case * 3;

        noc_async_write_tile(tile_base + 0, dst, cb_addr + 0 * tile_nbytes);
        noc_async_write_tile(tile_base + 1, dst, cb_addr + 1 * tile_nbytes);
        noc_async_write_tile(tile_base + 2, dst, cb_addr + 2 * tile_nbytes);
        noc_async_write_barrier();

        cb_pop_front(cb_output, 3);
    }
}
