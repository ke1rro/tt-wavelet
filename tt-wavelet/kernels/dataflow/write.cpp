
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out0);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out_writer = TensorAccessor(out_args, out_addr, tile_size_bytes);

    cb_wait_front(cb_out0, 1);
    uint32_t l1_read_addr = get_read_ptr(cb_out0);

    noc_async_write_tile(0, out_writer, l1_read_addr);
    noc_async_write_barrier();

    cb_pop_front(cb_out0, 1);
}