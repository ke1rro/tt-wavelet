#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t even_addr = get_arg_val<uint32_t>(0);
    const uint32_t odd_addr = get_arg_val<uint32_t>(1);
    const uint32_t even_stick_count = get_arg_val<uint32_t>(2);
    const uint32_t odd_stick_count = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_even = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_odd = get_compile_time_arg_val(1);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();
    const auto even_dst = TensorAccessor(dst_args, even_addr, stick_nbytes);
    const auto odd_dst = TensorAccessor(dst_args, odd_addr, stick_nbytes);

    uint32_t even_written = 0;
    uint32_t odd_written = 0;

    while (even_written < even_stick_count || odd_written < odd_stick_count) {
        if (even_written < even_stick_count) {
            cb_wait_front(cb_id_even, 1);
            const uint64_t noc_addr = even_dst.get_noc_addr(even_written);
            noc_async_write(get_read_ptr(cb_id_even), noc_addr, stick_nbytes);
            noc_async_write_barrier();
            cb_pop_front(cb_id_even, 1);
            even_written++;
        }
        if (odd_written < odd_stick_count) {
            cb_wait_front(cb_id_odd, 1);
            const uint64_t noc_addr = odd_dst.get_noc_addr(odd_written);
            noc_async_write(get_read_ptr(cb_id_odd), noc_addr, stick_nbytes);
            noc_async_write_barrier();
            cb_pop_front(cb_id_odd, 1);
            odd_written++;
        }
    }
}
