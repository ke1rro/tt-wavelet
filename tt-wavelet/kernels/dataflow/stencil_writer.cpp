// stencil_writer.cpp
//
// Writer dataflow kernel for the stencil compute pipeline.
//
// Drains the output CB (row-major sticks produced by the compute kernel
// after untilization) and writes them to the output DRAM buffer.
//
// Compile-time args:
//   0: cb_out       — CB index for output sticks
//   1: stick_nbytes — byte size of one stick (aligned)
//
// Runtime args:
//   0: dst_addr       — DRAM base address of output buffer
//   1: num_output_sticks — total number of sticks to write

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_output_sticks = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();
    const auto dst = TensorAccessor(dst_args, dst_addr, stick_nbytes);

    for (uint32_t stick = 0; stick < num_output_sticks; stick++) {
        cb_wait_front(cb_out, 1);
        const uint64_t noc_addr = dst.get_noc_addr(stick);
        noc_async_write(get_read_ptr(cb_out), noc_addr, stick_nbytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
