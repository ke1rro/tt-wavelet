#include <cstdint>

#include "../../tt_wavelet/include/lifting/step.hpp"
#include "api/dataflow/dataflow_api.h"
#include "lwt_readers_utils.hpp"
#include "lwt_tile_row_major_utils.hpp"

namespace {

constexpr uint32_t kStepPredict = static_cast<uint32_t>(ttwv::StepType::kPredict);
constexpr uint32_t kStepUpdate = static_cast<uint32_t>(ttwv::StepType::kUpdate);
constexpr uint32_t kStepScaleEven = static_cast<uint32_t>(ttwv::StepType::kScaleEven);
constexpr uint32_t kStepScaleOdd = static_cast<uint32_t>(ttwv::StepType::kScaleOdd);

constexpr uint32_t kBlockElements = ttwv::device_protocol::kLwtHalfStickElements;
constexpr uint32_t kRowsPerGroup = ttwv::device_protocol::kLwtRowsPerGroup;
constexpr uint32_t kInputTilesPerGroup = 2;
constexpr uint32_t kOutputBlocksPerRow = ttwv::device_protocol::kLwtOutputBlocksPerRow;
constexpr uint32_t kGroupOutputElements = ttwv::device_protocol::kLwtGroupOutputElements;

template <typename ConfigAccessor>
ALWI const uint32_t* load_config_page(
    const ConfigAccessor& config, const uint32_t config_addr, const uint32_t cb_config, const uint32_t page_index) {
    const auto page_accessor = TensorAccessor(config, config_addr, ttwv::device_protocol::kRouteConfigPageBytes);
    cb_reserve_back(cb_config, 1);
    noc_async_read(
        page_accessor.get_noc_addr(page_index), get_write_ptr(cb_config), ttwv::device_protocol::kRouteConfigPageBytes);
    noc_async_read_barrier();
    cb_push_back(cb_config, 1);
    cb_wait_front(cb_config, 1);
    return reinterpret_cast<const uint32_t*>(get_read_ptr(cb_config));
}

template <typename InputAccessor>
ALWI void initialize_cone(
    const InputAccessor& input,
    const uint32_t even_addr,
    const uint32_t odd_addr,
    const uint32_t cb_input_cache,
    const uint32_t input_length,
    const uint32_t left_pad,
    const uint32_t even_begin,
    const uint32_t even_length,
    const uint32_t odd_begin,
    const uint32_t odd_length) {
    ttwv::kernels::primitives::StickReadCache input_cache{
        cb_input_cache,
        ttwv::device_protocol::kStickBytes,
        ttwv::kStickWidth,
        ttwv::device_protocol::kLwtCacheStickCount,
        ttwv::kernels::primitives::kInvalidStick,
        0,
        false};
    auto* even_dst = reinterpret_cast<volatile tt_l1_ptr float*>(even_addr);
    auto* odd_dst = reinterpret_cast<volatile tt_l1_ptr float*>(odd_addr);
    uint32_t even_written = 0;
    uint32_t odd_written = 0;

    const uint32_t even_end = even_begin + even_length;
    const uint32_t odd_end = odd_begin + odd_length;
    const uint32_t split_begin = even_begin < odd_begin ? even_begin : odd_begin;
    const uint32_t split_end = even_end > odd_end ? even_end : odd_end;
    for (uint32_t split_index = split_begin; split_index < split_end; ++split_index) {
        if (split_index >= even_begin && split_index < even_end) {
            const uint32_t padded_index = 2U * split_index;
            const float value = ttwv::kernels::primitives::read_padded_symmetric_value(
                input, input_cache, input_length, left_pad, padded_index);
            even_dst[even_written++] = value;
        }
        if (split_index >= odd_begin && split_index < odd_end) {
            const uint32_t padded_index = 2U * split_index + 1U;
            const float value = ttwv::kernels::primitives::read_padded_symmetric_value(
                input, input_cache, input_length, left_pad, padded_index);
            odd_dst[odd_written++] = value;
        }
    }

    ttwv::kernels::primitives::release_cache(input_cache);
}

ALWI float read_local_or_zero(
    const volatile tt_l1_ptr float* src, const uint32_t logical_end, const uint32_t logical_index) {
    return logical_index < logical_end ? src[logical_index] : 0.0F;
}

ALWI void emit_predict_update_tiles(
    const uint32_t source_addr,
    const uint32_t base_addr,
    const uint32_t cb_src_tile0,
    const uint32_t cb_src_tile1,
    const uint32_t cb_base_tile,
    const uint32_t source_end,
    const uint32_t base_end,
    const uint32_t output_length,
    const uint32_t group_count,
    const uint32_t source_offset,
    const uint32_t base_offset,
    const uint32_t source_left_pad) {
    const auto* src = reinterpret_cast<volatile tt_l1_ptr float*>(source_addr);
    const auto* base = reinterpret_cast<volatile tt_l1_ptr float*>(base_addr);

    for (uint32_t group = 0; group < group_count; ++group) {
        const uint32_t group_base = group * kGroupOutputElements;
        cb_reserve_back(cb_src_tile0, 1);
        auto* src_tile0 = reinterpret_cast<float*>(get_write_ptr(cb_src_tile0));
        cb_reserve_back(cb_src_tile1, 1);
        auto* src_tile1 = reinterpret_cast<float*>(get_write_ptr(cb_src_tile1));

        for (uint32_t tile = 0; tile < kInputTilesPerGroup; ++tile) {
            auto* src_tile = tile == 0 ? src_tile0 : src_tile1;
            for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
                for (uint32_t local_block = 0; local_block < 2; ++local_block) {
                    const uint32_t block_id = tile * 2 + local_block;
                    auto* tile_block =
                        src_tile + ttwv::kernels::primitives::tile_offset(row, local_block * kBlockElements);
                    for (uint32_t lane = 0; lane < kBlockElements; ++lane) {
                        const int32_t packed_index = static_cast<int32_t>(
                            group_base + (row * kOutputBlocksPerRow + block_id) * kBlockElements + lane);
                        const int32_t source_local_index = packed_index - static_cast<int32_t>(source_left_pad);
                        float value = 0.0F;
                        if (source_local_index >= 0) {
                            value = read_local_or_zero(
                                src, source_end, source_offset + static_cast<uint32_t>(source_local_index));
                        }
                        tile_block[lane] = value;
                    }
                }
            }
        }

        cb_reserve_back(cb_base_tile, 2);
        const uint32_t base_tiles_addr = get_write_ptr(cb_base_tile);
        auto* base_full_tile = reinterpret_cast<float*>(base_tiles_addr);
        auto* base_tail_tile =
            reinterpret_cast<float*>(base_tiles_addr + ttwv::kernels::primitives::kTileScalars * sizeof(float));
        for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
            for (uint32_t block = 0; block < kOutputBlocksPerRow; ++block) {
                auto* tile_block =
                    block < 2 ? base_full_tile + ttwv::kernels::primitives::tile_offset(row, block * kBlockElements)
                              : base_tail_tile + ttwv::kernels::primitives::tile_offset(row, 0);
                for (uint32_t lane = 0; lane < kBlockElements; ++lane) {
                    const uint32_t output_index =
                        group_base + (row * kOutputBlocksPerRow + block) * kBlockElements + lane;
                    tile_block[lane] = output_index < output_length
                                           ? read_local_or_zero(base, base_end, base_offset + output_index)
                                           : 0.0F;
                }
            }
        }

        cb_push_back(cb_src_tile0, 1);
        cb_push_back(cb_src_tile1, 1);
        cb_push_back(cb_base_tile, 2);
    }
}

ALWI void emit_scale_tiles(
    const uint32_t source_addr,
    const uint32_t cb_scale_tile,
    const uint32_t source_end,
    const uint32_t source_offset,
    const uint32_t output_length,
    const uint32_t group_count) {
    const auto* src = reinterpret_cast<volatile tt_l1_ptr float*>(source_addr);

    for (uint32_t group = 0; group < group_count; ++group) {
        const uint32_t group_base = group * kGroupOutputElements;
        cb_reserve_back(cb_scale_tile, 2);
        const uint32_t scale_tiles_addr = get_write_ptr(cb_scale_tile);
        auto* scale_full_tile = reinterpret_cast<float*>(scale_tiles_addr);
        auto* scale_tail_tile =
            reinterpret_cast<float*>(scale_tiles_addr + ttwv::kernels::primitives::kTileScalars * sizeof(float));
        for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
            for (uint32_t block = 0; block < kOutputBlocksPerRow; ++block) {
                auto* tile_block =
                    block < 2 ? scale_full_tile + ttwv::kernels::primitives::tile_offset(row, block * kBlockElements)
                              : scale_tail_tile + ttwv::kernels::primitives::tile_offset(row, 0);
                for (uint32_t lane = 0; lane < kBlockElements; ++lane) {
                    const uint32_t output_index =
                        group_base + (row * kOutputBlocksPerRow + block) * kBlockElements + lane;
                    tile_block[lane] = output_index < output_length
                                           ? read_local_or_zero(src, source_end, source_offset + output_index)
                                           : 0.0F;
                }
            }
        }
        cb_push_back(cb_scale_tile, 2);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_length = get_arg_val<uint32_t>(1);
    const uint32_t left_pad = get_arg_val<uint32_t>(2);
    const uint32_t initial_even_addr = get_arg_val<uint32_t>(3);
    const uint32_t initial_odd_addr = get_arg_val<uint32_t>(4);
    const uint32_t chunk_config_addr = get_arg_val<uint32_t>(5);
    const uint32_t route_config_addr = get_arg_val<uint32_t>(6);
    const uint32_t chunk_begin = get_arg_val<uint32_t>(7);
    const uint32_t chunk_count = get_arg_val<uint32_t>(8);
    const uint32_t route_count = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_config = get_compile_time_arg_val(0);
    constexpr uint32_t cb_src_tile0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_src_tile1 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_base_tile = get_compile_time_arg_val(3);
    constexpr uint32_t cb_input_cache = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(5);
    constexpr auto config_args = TensorAccessorArgs<6>();
    constexpr auto input_args = TensorAccessorArgs<config_args.next_compile_time_args_offset()>();

    const auto input = TensorAccessor(input_args, input_addr, ttwv::device_protocol::kStickBytes);

    bool first_local_route = true;
    for (uint32_t local_chunk = 0; local_chunk < chunk_count; ++local_chunk) {
        if (!first_local_route) {
            cb_wait_front(cb_sync, 1);
            cb_pop_front(cb_sync, 1);
        }

        const uint32_t global_chunk = chunk_begin + local_chunk;
        const uint32_t* chunk = load_config_page(config_args, chunk_config_addr, cb_config, global_chunk);
        const uint32_t initial_even_begin = chunk[ttwv::device_protocol::kConeInitialEvenBegin];
        const uint32_t initial_even_length = chunk[ttwv::device_protocol::kConeInitialEvenLength];
        const uint32_t initial_odd_begin = chunk[ttwv::device_protocol::kConeInitialOddBegin];
        const uint32_t initial_odd_length = chunk[ttwv::device_protocol::kConeInitialOddLength];
        cb_pop_front(cb_config, 1);

        initialize_cone(
            input,
            initial_even_addr,
            initial_odd_addr,
            cb_input_cache,
            input_length,
            left_pad,
            initial_even_begin,
            initial_even_length,
            initial_odd_begin,
            initial_odd_length);

        for (uint32_t route_index = 0; route_index < route_count; ++route_index) {
            if (route_index > 0) {
                cb_wait_front(cb_sync, 1);
                cb_pop_front(cb_sync, 1);
            }
            const uint32_t config_index = global_chunk * route_count + route_index;
            const uint32_t* route = load_config_page(config_args, route_config_addr, cb_config, config_index);
            const uint32_t route_type = route[ttwv::device_protocol::kRouteType];
            const uint32_t source_addr = route[ttwv::device_protocol::kRouteSourceAddr];
            const uint32_t source_end = route[ttwv::device_protocol::kRouteSourceLength];
            const uint32_t base_addr = route[ttwv::device_protocol::kRouteBaseAddr];
            const uint32_t base_end = route[ttwv::device_protocol::kRouteBaseLength];
            const uint32_t output_length = route[ttwv::device_protocol::kRouteOutputLength];
            const uint32_t source_offset = route[ttwv::device_protocol::kRouteSourceOffset];
            const uint32_t base_offset = route[ttwv::device_protocol::kRouteBaseOffset];
            const uint32_t source_left_pad = route[ttwv::device_protocol::kRouteSourceLeftPad];
            const uint32_t group_count = route[ttwv::device_protocol::kRouteGroupCount];

            if (route_type == kStepPredict || route_type == kStepUpdate) {
                emit_predict_update_tiles(
                    source_addr,
                    base_addr,
                    cb_src_tile0,
                    cb_src_tile1,
                    cb_base_tile,
                    source_end,
                    base_end,
                    output_length,
                    group_count,
                    source_offset,
                    base_offset,
                    source_left_pad);
            } else if (route_type == kStepScaleEven || route_type == kStepScaleOdd) {
                emit_scale_tiles(source_addr, cb_base_tile, source_end, source_offset, output_length, group_count);
            }
            cb_pop_front(cb_config, 1);
            first_local_route = false;
        }
    }
}
