#include <cstdint>

#include "../primitives/stick_cache.hpp"
#include "../primitives/tile_row_major.hpp"
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t num_steps = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_src_tile0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_src_tile1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base_tile = get_compile_time_arg_val(2);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(3);
    constexpr uint32_t cb_src_cache = get_compile_time_arg_val(4);
    constexpr uint32_t cb_base_cache = get_compile_time_arg_val(5);
    constexpr uint32_t stick_width = get_compile_time_arg_val(6);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(7);
    constexpr auto accessor_args = TensorAccessorArgs<8>();

    for (uint32_t step = 0; step < num_steps; ++step) {
        // Wait for previous step's DRAM writes to complete before reading updated streams.
        if (step > 0) {
            cb_wait_front(cb_sync, 1);
            cb_pop_front(cb_sync, 1);
        }

        // Per-step runtime args packed at offset 1 + step * 9.
        const uint32_t arg_base = 1 + step * 9;
        const uint32_t src_addr = get_arg_val<uint32_t>(arg_base + 0);
        const uint32_t source_length = get_arg_val<uint32_t>(arg_base + 1);
        const uint32_t base_addr = get_arg_val<uint32_t>(arg_base + 2);
        const uint32_t base_length = get_arg_val<uint32_t>(arg_base + 3);
        const uint32_t output_stick_count = get_arg_val<uint32_t>(arg_base + 5);
        const uint32_t source_offset = get_arg_val<uint32_t>(arg_base + 6);
        const uint32_t base_offset = get_arg_val<uint32_t>(arg_base + 7);
        const uint32_t source_left_pad = get_arg_val<uint32_t>(arg_base + 8);

        const auto src = TensorAccessor(accessor_args, src_addr, stick_nbytes);
        const auto base = TensorAccessor(accessor_args, base_addr, stick_nbytes);

        ttwv::kernels::primitives::StickReadCache src_cache{
            cb_src_cache, stick_nbytes, stick_width, ttwv::kernels::primitives::kInvalidStick, false};
        ttwv::kernels::primitives::StickReadCache base_cache{
            cb_base_cache, stick_nbytes, stick_width, ttwv::kernels::primitives::kInvalidStick, false};

        for (uint32_t stick = 0; stick < output_stick_count; ++stick) {
            const uint32_t output_base = stick * stick_width;

            // Pack source tile 0 (32 elements)
            cb_reserve_back(cb_src_tile0, 1);
            auto* src_tile0 = reinterpret_cast<float*>(get_write_ptr(cb_src_tile0));
            ttwv::kernels::primitives::clear_tile(src_tile0);

            for (uint32_t col = 0; col < stick_width; ++col) {
                const int32_t packed_index = static_cast<int32_t>(output_base + col);
                const int32_t source_logical_index = packed_index - static_cast<int32_t>(source_left_pad);
                const float source_value =
                    source_logical_index >= 0 && static_cast<uint32_t>(source_logical_index) < source_length
                        ? ttwv::kernels::primitives::read_source_value(
                              src, src_cache, source_offset + static_cast<uint32_t>(source_logical_index))
                        : 0.0F;
                ttwv::kernels::primitives::store_row0_value(src_tile0, col, source_value);
            }

            // Pack source tile 1 (next 16 elements for stencil overlap)
            cb_reserve_back(cb_src_tile1, 1);
            auto* src_tile1 = reinterpret_cast<float*>(get_write_ptr(cb_src_tile1));
            ttwv::kernels::primitives::clear_tile(src_tile1);

            for (uint32_t col = 0; col < 16; ++col) {
                const int32_t packed_index = static_cast<int32_t>(output_base + stick_width + col);
                const int32_t source_logical_index = packed_index - static_cast<int32_t>(source_left_pad);
                const float source_value =
                    source_logical_index >= 0 && static_cast<uint32_t>(source_logical_index) < source_length
                        ? ttwv::kernels::primitives::read_source_value(
                              src, src_cache, source_offset + static_cast<uint32_t>(source_logical_index))
                        : 0.0F;
                ttwv::kernels::primitives::store_row0_value(src_tile1, col, source_value);
            }

            // Pack base tile (32 elements)
            cb_reserve_back(cb_base_tile, 1);
            auto* base_tile = reinterpret_cast<float*>(get_write_ptr(cb_base_tile));
            ttwv::kernels::primitives::clear_tile(base_tile);

            for (uint32_t col = 0; col < stick_width; ++col) {
                const uint32_t base_index = base_offset + output_base + col;
                const float base_value = base_index < base_length ? ttwv::kernels::primitives::read_source_value(
                                                                        base, base_cache, base_index)
                                                                  : 0.0F;
                ttwv::kernels::primitives::store_row0_value(base_tile, col, base_value);
            }

            cb_push_back(cb_src_tile0, 1);
            cb_push_back(cb_src_tile1, 1);
            cb_push_back(cb_base_tile, 1);
        }

        ttwv::kernels::primitives::release_cache(src_cache);
        ttwv::kernels::primitives::release_cache(base_cache);
    }
}
