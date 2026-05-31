#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pad_split_1d_writer_utils.hpp"

void kernel_main() {
    const uint32_t even_addr = get_arg_val<uint32_t>(0);
    const uint32_t odd_addr = get_arg_val<uint32_t>(1);
    const uint32_t even_stick_count = get_arg_val<uint32_t>(2);
    const uint32_t odd_stick_count = get_arg_val<uint32_t>(3);
    const uint32_t even_stick_begin = get_arg_val<uint32_t>(4);
    const uint32_t odd_stick_begin = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_even = get_named_compile_time_arg_val("cb_even");
    constexpr uint32_t cb_odd = get_named_compile_time_arg_val("cb_odd");
    constexpr auto even_dst_args = TensorAccessorArgs<0>();
    constexpr auto odd_dst_args = TensorAccessorArgs<even_dst_args.next_compile_time_args_offset()>();
    const auto even_dst = TensorAccessor(even_dst_args, even_addr, ttwv::device_protocol::kStickBytes);
    const auto odd_dst = TensorAccessor(odd_dst_args, odd_addr, ttwv::device_protocol::kStickBytes);

    uint32_t even_written = 0;
    uint32_t odd_written = 0;

    while (even_written < even_stick_count || odd_written < odd_stick_count) {
        if (even_written < even_stick_count) {
            cb_wait_front(cb_even, 1);
            const uint64_t noc_addr = even_dst.get_noc_addr(even_stick_begin + even_written);
            noc_async_write(get_read_ptr(cb_even), noc_addr, ttwv::device_protocol::kStickBytes);
            noc_async_write_barrier();
            cb_pop_front(cb_even, 1);
            even_written++;
        }
        if (odd_written < odd_stick_count) {
            cb_wait_front(cb_odd, 1);
            const uint64_t noc_addr = odd_dst.get_noc_addr(odd_stick_begin + odd_written);
            noc_async_write(get_read_ptr(cb_odd), noc_addr, ttwv::device_protocol::kStickBytes);
            noc_async_write_barrier();
            cb_pop_front(cb_odd, 1);
            odd_written++;
        }
    }
}
