// lwt_writer.cpp
//
// Writer kernel for the 1D LWT (lifting wavelet transform) pipeline.
//
// Drains the output circular buffer produced by the compute kernel and
// stores each stick sequentially into a DRAM destination buffer.
//
// Runtime args:
//   0: dst_addr    — base address of the DRAM output buffer
//   1: stick_count — total number of sticks to write
//
// Compile-time args:
//   0: cb_id_out   — index of the output CB to drain
//   1: stick_nbytes — byte size of one stick (must be NOC-aligned)
//   2+: TensorAccessorArgs for the destination buffer

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t stick_count = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();
    const auto dst = TensorAccessor(dst_args, dst_addr, stick_nbytes);

    for (uint32_t i = 0; i < stick_count; ++i) {
        cb_wait_front(cb_id_out, 1);
        const uint64_t noc_addr = dst.get_noc_addr(i);
        noc_async_write(get_read_ptr(cb_id_out), noc_addr, stick_nbytes);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, 1);
    }
}
