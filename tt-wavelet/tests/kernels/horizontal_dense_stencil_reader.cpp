#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_source0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_source1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_tile_size(cb_source0);
    constexpr auto input_args = TensorAccessorArgs<3>();
    const auto input = TensorAccessor(input_args, input_addr, tile_bytes);

    cb_reserve_back(cb_source0, 1);
    noc_async_read_tile(0, input, get_write_ptr(cb_source0));
    noc_async_read_barrier();
    cb_push_back(cb_source0, 1);

    cb_reserve_back(cb_source1, 1);
    noc_async_read_tile(1, input, get_write_ptr(cb_source1));
    noc_async_read_barrier();
    cb_push_back(cb_source1, 1);

    cb_reserve_back(cb_base, 1);
    noc_async_read_tile(2, input, get_write_ptr(cb_base));
    noc_async_read_barrier();
    cb_push_back(cb_base, 1);
}
