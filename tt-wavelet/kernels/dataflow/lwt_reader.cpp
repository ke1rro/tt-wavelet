#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "lwt_readers_utils.hpp"

namespace {

constexpr uint32_t kBlockElements = 16;
constexpr uint32_t kRowsPerGroup = 32;
constexpr uint32_t kInputTilesPerGroup = 2;
constexpr uint32_t kInputBlocksPerRow = kInputTilesPerGroup * 2;
constexpr uint32_t kOutputBlocksPerRow = kInputBlocksPerRow - 1;
constexpr uint32_t kGroupOutputElements = kRowsPerGroup * kOutputBlocksPerRow * kBlockElements;

template <typename SrcAccessor>
float read_dense_or_zero(
    const SrcAccessor& src,
    ttwv::kernels::primitives::StickReadCache& cache,
    const uint32_t logical_length,
    const uint32_t logical_index) {
    return logical_index < logical_length ? ttwv::kernels::primitives::read_source_value(src, cache, logical_index)
                                          : 0.0F;
}

}  // namespace

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
        const uint32_t output_length = get_arg_val<uint32_t>(arg_base + 4);
        const uint32_t output_group_count = get_arg_val<uint32_t>(arg_base + 5);
        const uint32_t source_offset = get_arg_val<uint32_t>(arg_base + 6);
        const uint32_t base_offset = get_arg_val<uint32_t>(arg_base + 7);
        const uint32_t source_left_pad = get_arg_val<uint32_t>(arg_base + 8);

        const auto src = TensorAccessor(accessor_args, src_addr, stick_nbytes);
        const auto base = TensorAccessor(accessor_args, base_addr, stick_nbytes);

        ttwv::kernels::primitives::StickReadCache src_cache{
            cb_src_cache, stick_nbytes, stick_width, ttwv::kernels::primitives::kInvalidStick, false};
        ttwv::kernels::primitives::StickReadCache base_cache{
            cb_base_cache, stick_nbytes, stick_width, ttwv::kernels::primitives::kInvalidStick, false};

        for (uint32_t group = 0; group < output_group_count; ++group) {
            const uint32_t group_base = group * kGroupOutputElements;

            cb_reserve_back(cb_src_tile0, 1);
            auto* src_tile0 = reinterpret_cast<float*>(get_write_ptr(cb_src_tile0));
            ttwv::kernels::primitives::clear_tile(src_tile0);

            cb_reserve_back(cb_src_tile1, 1);
            auto* src_tile1 = reinterpret_cast<float*>(get_write_ptr(cb_src_tile1));
            ttwv::kernels::primitives::clear_tile(src_tile1);

            for (uint32_t tile = 0; tile < kInputTilesPerGroup; ++tile) {
                auto* src_tile = tile == 0 ? src_tile0 : src_tile1;
                for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
                    for (uint32_t local_block = 0; local_block < 2; ++local_block) {
                        const uint32_t block_id = tile * 2 + local_block;
                        for (uint32_t lane = 0; lane < kBlockElements; ++lane) {
                            const int32_t packed_index = static_cast<int32_t>(
                                group_base + (row * kOutputBlocksPerRow + block_id) * kBlockElements + lane);
                            const int32_t source_logical_index = packed_index - static_cast<int32_t>(source_left_pad);

                            float source_value = 0.0F;
                            if (source_logical_index >= 0) {
                                const uint32_t source_index =
                                    source_offset + static_cast<uint32_t>(source_logical_index);
                                source_value = read_dense_or_zero(src, src_cache, source_length, source_index);
                            }

                            ttwv::kernels::primitives::store_tile_value(
                                src_tile, row, local_block * kBlockElements + lane, source_value);
                        }
                    }
                }
            }

            cb_reserve_back(cb_base_tile, 2);
            const uint32_t base_tiles_addr = get_write_ptr(cb_base_tile);
            auto* base_full_tile = reinterpret_cast<float*>(base_tiles_addr);
            auto* base_tail_tile =
                reinterpret_cast<float*>(base_tiles_addr + ttwv::kernels::primitives::kTileScalars * sizeof(float));
            ttwv::kernels::primitives::clear_tile(base_full_tile);
            ttwv::kernels::primitives::clear_tile(base_tail_tile);

            for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
                for (uint32_t block = 0; block < kOutputBlocksPerRow; ++block) {
                    for (uint32_t lane = 0; lane < kBlockElements; ++lane) {
                        const uint32_t output_index =
                            group_base + (row * kOutputBlocksPerRow + block) * kBlockElements + lane;
                        const uint32_t base_index = base_offset + output_index;
                        const float base_value = output_index < output_length
                                                     ? read_dense_or_zero(base, base_cache, base_length, base_index)
                                                     : 0.0F;

                        if (block < 2) {
                            ttwv::kernels::primitives::store_tile_value(
                                base_full_tile, row, block * kBlockElements + lane, base_value);
                        } else {
                            ttwv::kernels::primitives::store_tile_value(base_tail_tile, row, lane, base_value);
                        }
                    }
                }
            }

            cb_push_back(cb_src_tile0, 1);
            cb_push_back(cb_src_tile1, 1);
            cb_push_back(cb_base_tile, 2);
        }

        ttwv::kernels::primitives::release_cache(src_cache);
        ttwv::kernels::primitives::release_cache(base_cache);
    }
}
