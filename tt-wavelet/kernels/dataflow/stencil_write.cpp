#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t cb_output = get_compile_time_arg_val(0);
    constexpr uint32_t t_size_b = get_tile_size(cb_output);
    const auto output_args = TensorAccessorArgs<1>();
    const auto output = TensorAccessor(output_args, output_addr, t_size_b);

    cb_wait_front(cb_output, 1);
    const uint32_t l1_read_addr = get_read_ptr(cb_output);
    noc_async_write_tile(0, output, l1_read_addr);
    noc_async_write_barrier();
    cb_pop_front(cb_output, 1);

    //     // Temporary debug path: dump two tiles from cb_output.
    // for (uint32_t tile_id = 0; tile_id < 2; ++tile_id) {
    //     cb_wait_front(cb_output, 1);
    //     const uint32_t l1_read_addr = get_read_ptr(cb_output);
    //     noc_async_write_tile(tile_id, output, l1_read_addr);
    //     noc_async_write_barrier();
    //     cb_pop_front(cb_output, 1);
    // }
}
