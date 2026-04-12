#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "../utils/row_major_tile.hpp"
#include "../utils/stick_read_cache.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t base_addr = get_arg_val<uint32_t>(1);
    const uint32_t source_length = get_arg_val<uint32_t>(2);
    const uint32_t output_length = get_arg_val<uint32_t>(3);
    const uint32_t output_stick_count = get_arg_val<uint32_t>(4);
    const uint32_t k = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_input0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(3);
    constexpr uint32_t cb_src_cache = get_compile_time_arg_val(4);
    constexpr uint32_t cb_base_cache = get_compile_time_arg_val(5);
    constexpr uint32_t stick_width = get_compile_time_arg_val(6);
    constexpr auto accessor_args = TensorAccessorArgs<7>();
    const auto src = TensorAccessor(accessor_args, src_addr, stick_nbytes);
    const auto base = TensorAccessor(accessor_args, base_addr, stick_nbytes);

    ttwv::kernels::utils::StickReadCache src_cache{
        cb_src_cache, stick_nbytes, stick_width, ttwv::kernels::utils::kInvalidStick, false};
    ttwv::kernels::utils::StickReadCache base_cache{
        cb_base_cache, stick_nbytes, stick_width, ttwv::kernels::utils::kInvalidStick, false};

    const int32_t left_pad = ttwv::kernels::utils::stencil_left_pad(k);

    for (uint32_t stick = 0; stick < output_stick_count; ++stick) {
        cb_reserve_back(cb_input0, 1);
        auto* input0_tile = reinterpret_cast<float*>(get_write_ptr(cb_input0));
        ttwv::kernels::utils::clear_tile(input0_tile);

        cb_reserve_back(cb_input1, 1);
        auto* input1_tile = reinterpret_cast<float*>(get_write_ptr(cb_input1));
        ttwv::kernels::utils::clear_tile(input1_tile);

        cb_reserve_back(cb_base, 1);
        auto* base_tile = reinterpret_cast<float*>(get_write_ptr(cb_base));
        ttwv::kernels::utils::clear_tile(base_tile);

        const uint32_t output_base = stick * stick_width;

        for (uint32_t col = 0; col < stick_width; ++col) {
            const int32_t shifted_index = static_cast<int32_t>(output_base + col) - left_pad;
            const float source_value =
                shifted_index >= 0 && shifted_index < static_cast<int32_t>(source_length)
                    ? ttwv::kernels::utils::read_source_value(src, src_cache, static_cast<uint32_t>(shifted_index))
                    : 0.0F;
            ttwv::kernels::utils::store_row0_value(input0_tile, col, source_value);
        }

        for (uint32_t col = 0; col < 16; ++col) {
            const int32_t shifted_index = static_cast<int32_t>(output_base + stick_width + col) - left_pad;
            const float source_value =
                shifted_index >= 0 && shifted_index < static_cast<int32_t>(source_length)
                    ? ttwv::kernels::utils::read_source_value(src, src_cache, static_cast<uint32_t>(shifted_index))
                    : 0.0F;
            ttwv::kernels::utils::store_row0_value(input1_tile, col, source_value);
        }

        for (uint32_t col = 0; col < stick_width; ++col) {
            const uint32_t output_index = output_base + col;
            const float base_value = output_index < output_length
                                         ? ttwv::kernels::utils::read_source_value(base, base_cache, output_index)
                                         : 0.0F;
            ttwv::kernels::utils::store_row0_value(base_tile, col, base_value);
        }

        cb_push_back(cb_input0, 1);
        cb_push_back(cb_input1, 1);
        cb_push_back(cb_base, 1);
    }

    ttwv::kernels::utils::release_cache(src_cache);
    ttwv::kernels::utils::release_cache(base_cache);
}
