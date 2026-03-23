#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    uint32_t base_addr = get_args_val<uint32_t>(0);
    uint32_t in0_addr = get_args_val<uint32_t>(1);
    uint32_t in1_addr = get_args_val<uint32_t>(2);

    constexpr uint32_t base_cb = tt::CBIndex::c_0;
    constexpr uint32_t in0_cb = tt::CBIndex::c_1;
    constexpr uint32_t in1_cb = tt::CBIndex::c_2;

    const uint32_t tile_size_bytes = get_tile_size(base_cb);

    constexpr auto base_args = TensorAccessorArgs<0>();
    const auto base_in = TensorAccessor(base_args, base_addr, tile_size_bytes);
    constexpr auto in0_args = TensorAccessorArgs<base_args.next_compile_time_args_offset()>();
    constexpr auto in0 = TensorAccessor(in0_args, in0_addr, tile_size_bytes);
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    constexpr auto in1 = TensorAccessor(in1_args, in1_addr, tile_size_bytes);

    cb_reserve_back(base_cb, 1);
    cb_reserve_back(in0_cb, 1);
    cb_reserve_back(in1_cb, 1);

    noc_async_read_tile(0, base_in, get_write_ptr(base_cb));
    noc_async_read_tile(1, in0, get_write_ptr(in0_cb));
    noc_async_read_tile(2, in1, get_write_ptr(in1_cb));

    noc_async_read_barrier();

    cb_push_back(base_cb, 1);
    cb_push_back(in0_cb, 1);
    cb_push_back(in1_cb, 1);
}