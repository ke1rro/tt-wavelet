#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "lwt_readers_utils.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t logical_length = get_arg_val<uint32_t>(1);
    const uint32_t output_stick_count = get_arg_val<uint32_t>(2);
    const uint32_t scalar_packed = get_arg_val<uint32_t>(3);
    union {
        uint32_t bits;
        float value;
    } scalar_bits{scalar_packed};
    const float scalar = scalar_bits.value;

    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_coeff = get_compile_time_arg_val(1);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb_src_cache = get_compile_time_arg_val(3);
    constexpr uint32_t stick_width = get_compile_time_arg_val(4);
    constexpr auto accessor_args = TensorAccessorArgs<5>();
    const auto src = TensorAccessor(accessor_args, src_addr, stick_nbytes);

    ttwv::kernels::primitives::StickReadCache src_cache{
        cb_src_cache, stick_nbytes, stick_width, ttwv::kernels::primitives::kInvalidStick, false};

    for (uint32_t stick = 0; stick < output_stick_count; ++stick) {
        cb_reserve_back(cb_input, 1);
        auto* input_tile = reinterpret_cast<float*>(get_write_ptr(cb_input));
        ttwv::kernels::primitives::clear_tile(input_tile);

        cb_reserve_back(cb_coeff, 1);
        auto* coeff_tile = reinterpret_cast<float*>(get_write_ptr(cb_coeff));
        ttwv::kernels::primitives::clear_tile(coeff_tile);

        const uint32_t output_base = stick * stick_width;
        for (uint32_t col = 0; col < stick_width; ++col) {
            const uint32_t logical_index = output_base + col;
            const float input_value = logical_index < logical_length
                                          ? ttwv::kernels::primitives::read_source_value(src, src_cache, logical_index)
                                          : 0.0F;
            ttwv::kernels::primitives::store_row0_value(input_tile, col, input_value);
            ttwv::kernels::primitives::store_row0_value(coeff_tile, col, scalar);
        }

        cb_push_back(cb_input, 1);
        cb_push_back(cb_coeff, 1);
    }

    ttwv::kernels::primitives::release_cache(src_cache);
}
