#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_halo = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input = get_compile_time_arg_val(1);

    constexpr uint32_t t_size_b = get_tile_size(cb_halo);

    constexpr auto input_args = TensorAccessorArgs<2>();
    const auto input = TensorAccessor(input_args, input_addr, t_size_b);

    cb_reserve_back(cb_halo, 1);
    // fisrt tile halo from dram to cb halo 0
    noc_async_read_tile(0, input, get_write_ptr(cb_halo));
    noc_async_read_barrier();
    cb_push_back(cb_halo, 1);

    // second tile input from dram to cb input 1
    cb_reserve_back(cb_input, 1);
    noc_async_read_tile(1, input, get_write_ptr(cb_input));
    noc_async_read_barrier();
    cb_push_back(cb_input, 1);
}