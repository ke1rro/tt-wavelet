#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t kHalfStickElements = 16;
constexpr uint32_t kHalfStickBytes = kHalfStickElements * sizeof(float);
constexpr uint32_t kSecondHalfOffsetBytes = 256 * sizeof(float);

}  // namespace

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_stick_count = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_output = get_compile_time_arg_val(0);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();
    const auto dst = TensorAccessor(dst_args, output_addr, stick_nbytes);

    for (uint32_t stick = 0; stick < output_stick_count; ++stick) {
        cb_wait_front(cb_output, 1);
        const uint32_t output_tile = get_read_ptr(cb_output);
        const uint64_t noc_addr = dst.get_noc_addr(stick);

        noc_async_write(output_tile, noc_addr, kHalfStickBytes);
        noc_async_write(output_tile + kSecondHalfOffsetBytes, noc_addr + kHalfStickBytes, kHalfStickBytes);
        noc_async_write_barrier();

        cb_pop_front(cb_output, 1);
    }
}
