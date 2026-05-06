#include <cstdint>

#include "../../tt_wavelet/include/device_protocol/lwt_config.hpp"
#include "api/dataflow/dataflow_api.h"
#include "lwt_readers_utils.hpp"

namespace {

constexpr uint32_t kStepPredict = 0;
constexpr uint32_t kStepUpdate = 1;
constexpr uint32_t kStepScaleEven = 2;
constexpr uint32_t kStepScaleOdd = 3;

constexpr uint32_t kBlockElements = 16;
constexpr uint32_t kRowsPerGroup = 32;
constexpr uint32_t kInputTilesPerGroup = 2;
constexpr uint32_t kInputBlocksPerRow = kInputTilesPerGroup * 2;
constexpr uint32_t kOutputBlocksPerRow = kInputBlocksPerRow - 1;
constexpr uint32_t kGroupOutputElements = kRowsPerGroup * kOutputBlocksPerRow * kBlockElements;

template <typename SrcAccessor>
ALWI float read_dense_or_zero(
    const SrcAccessor& src,
    ttwv::kernels::primitives::StickReadCache& cache,
    const uint32_t logical_length,
    const uint32_t logical_index) {
    return logical_index < logical_length ? ttwv::kernels::primitives::read_source_value(src, cache, logical_index)
                                          : 0.0F;
}

template <typename ConfigAccessor>
ALWI const uint32_t* load_route_config(
    const ConfigAccessor& config,
    const uint32_t cb_config,
    const uint32_t config_page_bytes,
    const uint32_t route_index) {
    cb_reserve_back(cb_config, 1);
    noc_async_read(config.get_noc_addr(route_index), get_write_ptr(cb_config), config_page_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_config, 1);
    cb_wait_front(cb_config, 1);
    return reinterpret_cast<const uint32_t*>(get_read_ptr(cb_config));
}

template <typename DataAccessor>
ALWI void emit_predict_update_tiles(
    const DataAccessor& src,
    const DataAccessor& base,
    const uint32_t cb_src_tile0,
    const uint32_t cb_src_tile1,
    const uint32_t cb_base_tile,
    const uint32_t stick_nbytes,
    const uint32_t stick_width,
    const uint32_t cb_src_cache,
    const uint32_t cb_base_cache,
    const uint32_t source_length,
    const uint32_t base_length,
    const uint32_t output_length,
    const uint32_t global_group_begin,
    const uint32_t local_group_count,
    const uint32_t source_offset,
    const uint32_t base_offset,
    const uint32_t source_left_pad) {
    ttwv::kernels::primitives::StickReadCache src_cache{
        cb_src_cache, stick_nbytes, stick_width, 1, ttwv::kernels::primitives::kInvalidStick, 0, false};
    ttwv::kernels::primitives::StickReadCache base_cache{
        cb_base_cache, stick_nbytes, stick_width, 1, ttwv::kernels::primitives::kInvalidStick, 0, false};

    for (uint32_t local_group = 0; local_group < local_group_count; ++local_group) {
        const uint32_t global_group = global_group_begin + local_group;
        const uint32_t group_base = global_group * kGroupOutputElements;

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
                            const uint32_t source_index = source_offset + static_cast<uint32_t>(source_logical_index);
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

template <typename DataAccessor>
ALWI void emit_scale_tiles(
    const DataAccessor& src,
    const uint32_t cb_scale_tile,
    const uint32_t stick_nbytes,
    const uint32_t stick_width,
    const uint32_t cb_src_cache,
    const uint32_t source_length,
    const uint32_t global_group_begin,
    const uint32_t local_group_count) {
    ttwv::kernels::primitives::StickReadCache src_cache{
        cb_src_cache, stick_nbytes, stick_width, 1, ttwv::kernels::primitives::kInvalidStick, 0, false};

    for (uint32_t local_group = 0; local_group < local_group_count; ++local_group) {
        const uint32_t global_group = global_group_begin + local_group;
        const uint32_t group_base = global_group * kGroupOutputElements;

        cb_reserve_back(cb_scale_tile, 2);
        const uint32_t scale_tiles_addr = get_write_ptr(cb_scale_tile);
        auto* scale_full_tile = reinterpret_cast<float*>(scale_tiles_addr);
        auto* scale_tail_tile =
            reinterpret_cast<float*>(scale_tiles_addr + ttwv::kernels::primitives::kTileScalars * sizeof(float));
        ttwv::kernels::primitives::clear_tile(scale_full_tile);
        ttwv::kernels::primitives::clear_tile(scale_tail_tile);

        for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
            for (uint32_t block = 0; block < kOutputBlocksPerRow; ++block) {
                for (uint32_t lane = 0; lane < kBlockElements; ++lane) {
                    const uint32_t output_index =
                        group_base + (row * kOutputBlocksPerRow + block) * kBlockElements + lane;
                    const float value = read_dense_or_zero(src, src_cache, source_length, output_index);

                    if (block < 2) {
                        ttwv::kernels::primitives::store_tile_value(
                            scale_full_tile, row, block * kBlockElements + lane, value);
                    } else {
                        ttwv::kernels::primitives::store_tile_value(scale_tail_tile, row, lane, value);
                    }
                }
            }
        }

        cb_push_back(cb_scale_tile, 2);
    }

    ttwv::kernels::primitives::release_cache(src_cache);
}

}  // namespace

void kernel_main() {
    const uint32_t route_config_addr = get_arg_val<uint32_t>(0);
    const uint32_t route_count = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_config = get_compile_time_arg_val(0);
    constexpr uint32_t config_page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t cb_src_tile0 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_src_tile1 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_base_tile = get_compile_time_arg_val(4);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(5);
    constexpr uint32_t cb_src_cache = get_compile_time_arg_val(6);
    constexpr uint32_t cb_base_cache = get_compile_time_arg_val(7);
    constexpr uint32_t stick_width = get_compile_time_arg_val(8);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(9);
    constexpr auto config_args = TensorAccessorArgs<10>();
    constexpr auto data_args = TensorAccessorArgs<config_args.next_compile_time_args_offset()>();

    const auto config = TensorAccessor(config_args, route_config_addr, config_page_bytes);

    for (uint32_t route_index = 0; route_index < route_count; ++route_index) {
        if (route_index > 0) {
            cb_wait_front(cb_sync, 1);
            cb_pop_front(cb_sync, 1);
        }

        const uint32_t* route = load_route_config(config, cb_config, config_page_bytes, route_index);
        const uint32_t route_type = route[ttwv::device_protocol::kRouteType];
        const uint32_t source_addr = route[ttwv::device_protocol::kRouteSourceAddr];
        const uint32_t source_length = route[ttwv::device_protocol::kRouteSourceLength];
        const uint32_t base_addr = route[ttwv::device_protocol::kRouteBaseAddr];
        const uint32_t base_length = route[ttwv::device_protocol::kRouteBaseLength];
        const uint32_t output_length = route[ttwv::device_protocol::kRouteOutputLength];
        const uint32_t source_offset = route[ttwv::device_protocol::kRouteSourceOffset];
        const uint32_t base_offset = route[ttwv::device_protocol::kRouteBaseOffset];
        const uint32_t source_left_pad = route[ttwv::device_protocol::kRouteSourceLeftPad];
        const uint32_t route_range_arg_base = 2 + route_index * 2;
        const uint32_t global_group_begin = get_arg_val<uint32_t>(route_range_arg_base);
        const uint32_t local_group_count = get_arg_val<uint32_t>(route_range_arg_base + 1);

        const auto src = TensorAccessor(data_args, source_addr, stick_nbytes);

        if (route_type == kStepPredict || route_type == kStepUpdate) {
            const auto base = TensorAccessor(data_args, base_addr, stick_nbytes);
            emit_predict_update_tiles(
                src,
                base,
                cb_src_tile0,
                cb_src_tile1,
                cb_base_tile,
                stick_nbytes,
                stick_width,
                cb_src_cache,
                cb_base_cache,
                source_length,
                base_length,
                output_length,
                global_group_begin,
                local_group_count,
                source_offset,
                base_offset,
                source_left_pad);
        } else if (route_type == kStepScaleEven || route_type == kStepScaleOdd) {
            emit_scale_tiles(
                src,
                cb_base_tile,
                stick_nbytes,
                stick_width,
                cb_src_cache,
                source_length,
                global_group_begin,
                local_group_count);
        }

        cb_pop_front(cb_config, 1);
    }
}
