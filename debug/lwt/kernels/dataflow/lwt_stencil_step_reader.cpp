#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "primitives/tile_row_major.hpp"
#include "primitives/stick_cache.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t base_addr = get_arg_val<uint32_t>(1);
    const uint32_t source_length = get_arg_val<uint32_t>(2);
    const uint32_t base_length = get_arg_val<uint32_t>(3);
    const uint32_t output_length = get_arg_val<uint32_t>(4);
    const uint32_t output_stick_count = get_arg_val<uint32_t>(5);
    const uint32_t source_offset = get_arg_val<uint32_t>(6);
    const uint32_t base_offset = get_arg_val<uint32_t>(7);
    const uint32_t source_left_pad = get_arg_val<uint32_t>(8);

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

    ttwv::kernels::primitives::StickReadCache src_cache{
        cb_src_cache, stick_nbytes, stick_width, ttwv::kernels::primitives::kInvalidStick, false};
    ttwv::kernels::primitives::StickReadCache base_cache{
        cb_base_cache, stick_nbytes, stick_width, ttwv::kernels::primitives::kInvalidStick, false};

    for (uint32_t stick = 0; stick < output_stick_count; ++stick) {
        cb_reserve_back(cb_input0, 1);
        auto* input0_tile = reinterpret_cast<float*>(get_write_ptr(cb_input0));
        ttwv::kernels::primitives::clear_tile(input0_tile);

        cb_reserve_back(cb_input1, 1);
        auto* input1_tile = reinterpret_cast<float*>(get_write_ptr(cb_input1));
        ttwv::kernels::primitives::clear_tile(input1_tile);

        cb_reserve_back(cb_base, 1);
        auto* base_tile = reinterpret_cast<float*>(get_write_ptr(cb_base));
        ttwv::kernels::primitives::clear_tile(base_tile);

        const uint32_t output_base = stick * stick_width;

        for (uint32_t col = 0; col < stick_width; ++col) {
            const int32_t packed_index = static_cast<int32_t>(output_base + col);
            const int32_t source_logical_index = packed_index - static_cast<int32_t>(source_left_pad);
            const float source_value =
                source_logical_index >= 0 && static_cast<uint32_t>(source_logical_index) < source_length
                    ? ttwv::kernels::primitives::read_source_value(
                          src, src_cache, source_offset + static_cast<uint32_t>(source_logical_index))
                    : 0.0F;
            ttwv::kernels::primitives::store_row0_value(input0_tile, col, source_value);
        }

        for (uint32_t col = 0; col < 16; ++col) {
            const int32_t packed_index = static_cast<int32_t>(output_base + stick_width + col);
            const int32_t source_logical_index = packed_index - static_cast<int32_t>(source_left_pad);
            const float source_value =
                source_logical_index >= 0 && static_cast<uint32_t>(source_logical_index) < source_length
                    ? ttwv::kernels::primitives::read_source_value(
                          src, src_cache, source_offset + static_cast<uint32_t>(source_logical_index))
                    : 0.0F;
            ttwv::kernels::primitives::store_row0_value(input1_tile, col, source_value);
        }

        for (uint32_t col = 0; col < stick_width; ++col) {
            const uint32_t base_index = base_offset + output_base + col;
            const float base_value = base_index < base_length
                                         ? ttwv::kernels::primitives::read_source_value(base, base_cache, base_index)
                                         : 0.0F;
            ttwv::kernels::primitives::store_row0_value(base_tile, col, base_value);
        }

        cb_push_back(cb_input0, 1);
        cb_push_back(cb_input1, 1);
        cb_push_back(cb_base, 1);
    }

    ttwv::kernels::primitives::release_cache(src_cache);
    ttwv::kernels::primitives::release_cache(base_cache);
}
