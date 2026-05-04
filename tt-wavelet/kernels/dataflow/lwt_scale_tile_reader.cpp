#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_page_count = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t tile_nbytes = get_tile_size(cb_input);
    constexpr auto src_args = TensorAccessorArgs<1>();
    const auto src = TensorAccessor(src_args, src_addr, tile_nbytes);

    for (uint32_t page = 0; page < tile_page_count; ++page) {
        cb_reserve_back(cb_input, 1);
        noc_async_read_tile(page, src, get_write_ptr(cb_input));
        noc_async_read_barrier();
        cb_push_back(cb_input, 1);
    }
}
