#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_tile_size(cb_output);
    constexpr auto output_args = TensorAccessorArgs<1>();
    const auto output = TensorAccessor(output_args, output_addr, tile_bytes);

    cb_wait_front(cb_output, 1);
    noc_async_write_tile(0, output, get_read_ptr(cb_output));
    noc_async_write_barrier();
    cb_pop_front(cb_output, 1);
}
