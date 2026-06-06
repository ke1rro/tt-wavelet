#ifndef TTWV_LWT_SCHEME_HEADER
#error "TTWV_LWT_SCHEME_HEADER must identify the generated lifting scheme header"
#endif

#ifndef TTWV_LWT_SCHEME_TYPE
#error "TTWV_LWT_SCHEME_TYPE must identify the generated lifting scheme type"
#endif

#include <cstdint>

#include "../../tt_wavelet/include/device_protocol/lwt_config.hpp"
#include "../../tt_wavelet/include/lifting/static_scheme.hpp"
#include "../primitives/indexing.hpp"
#include "../primitives/stick_cache.hpp"
#include "../primitives/tile_row_major.hpp"
#include "api/dataflow/dataflow_api.h"
#include TTWV_LWT_SCHEME_HEADER

namespace ttwv::kernels {

namespace row_major = ttwv::kernels::primitives;

namespace {

constexpr uint32_t kBlockElements = ttwv::device_protocol::kLwtHalfStickElements;
constexpr uint32_t kRowsPerGroup = ttwv::device_protocol::kLwtRowsPerGroup;
constexpr uint32_t kInputTilesPerGroup = 2;
constexpr uint32_t kOutputBlocksPerRow = ttwv::device_protocol::kLwtOutputBlocksPerRow;
constexpr uint32_t kGroupOutputElements = ttwv::device_protocol::kLwtGroupOutputElements;
constexpr uint32_t kStepConfigWords = 4;
constexpr uint32_t kCommonArgCount = 11;

struct LocalStream {
    uint32_t addr;
    int32_t lo;
    uint32_t length;
};

ALWI int32_t i32_arg(const uint32_t index) { return static_cast<int32_t>(get_arg_val<uint32_t>(index)); }

template <typename SrcAccessor>
ALWI float read_input_value(
    const SrcAccessor& src,
    ttwv::kernels::primitives::StickReadCache& cache,
    const uint32_t input_length,
    const int32_t logical_index) {
    const uint32_t source_index = ttwv::kernels::primitives::symmetric_index(logical_index, input_length);
    return ttwv::kernels::primitives::read_source_value(src, cache, source_index, input_length);
}

ALWI bool contains(const LocalStream& stream, const int32_t logical_index) {
    return stream.length > 0 && logical_index >= stream.lo &&
           logical_index < stream.lo + static_cast<int32_t>(stream.length);
}

ALWI float stream_value_or_zero(const LocalStream& stream, const int32_t logical_index) {
    if (!contains(stream, logical_index)) {
        return 0.0F;
    }
    const auto* values = reinterpret_cast<const float*>(stream.addr);
    return values[static_cast<uint32_t>(logical_index - stream.lo)];
}

ALWI void swap_streams(LocalStream& lhs, LocalStream& rhs) {
    const LocalStream tmp = lhs;
    lhs = rhs;
    rhs = tmp;
}

template <typename SrcAccessor>
ALWI void fill_initial_stream(
    const SrcAccessor& src,
    ttwv::kernels::primitives::StickReadCache& cache,
    const uint32_t input_length,
    const int32_t stream_lo,
    const uint32_t stream_length,
    const int32_t delay,
    const int32_t parity,
    const int32_t pad,
    const uint32_t dst_addr) {
    auto* dst = reinterpret_cast<float*>(dst_addr);
    for (uint32_t index = 0; index < stream_length; ++index) {
        const int32_t logical_index = stream_lo + static_cast<int32_t>(index);
        const int32_t dense_index = logical_index - delay;
        const int32_t padded_index = 2 * dense_index + parity;
        dst[index] = read_input_value(src, cache, input_length, padded_index - pad);
    }
}

template <typename Step>
ALWI void emit_predict_update_tiles(
    const LocalStream& source,
    const LocalStream& base,
    const uint32_t cb_src_tile0,
    const uint32_t cb_src_tile1,
    const uint32_t cb_base_tile,
    const int32_t target_group_begin,
    const uint32_t target_group_count) {
    for (uint32_t local_group = 0; local_group < target_group_count; ++local_group) {
        const int32_t group = target_group_begin + static_cast<int32_t>(local_group);
        const int32_t group_base = group * static_cast<int32_t>(kGroupOutputElements);

        cb_reserve_back(cb_src_tile0, 1);
        auto* src_tile0 = reinterpret_cast<float*>(get_write_ptr(cb_src_tile0));

        cb_reserve_back(cb_src_tile1, 1);
        auto* src_tile1 = reinterpret_cast<float*>(get_write_ptr(cb_src_tile1));

        for (uint32_t tile = 0; tile < kInputTilesPerGroup; ++tile) {
            auto* src_tile = tile == 0 ? src_tile0 : src_tile1;
            for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
                for (uint32_t local_block = 0; local_block < 2; ++local_block) {
                    const uint32_t block_id = tile * 2 + local_block;
                    auto* tile_block = src_tile + row_major::tile_offset(row, local_block * kBlockElements);
                    for (uint32_t lane = 0; lane < kBlockElements; ++lane) {
                        const int32_t output_logical =
                            group_base +
                            static_cast<int32_t>((row * kOutputBlocksPerRow + block_id) * kBlockElements + lane);
                        const int32_t source_logical = output_logical - Step::shift - 16;
                        tile_block[lane] = stream_value_or_zero(source, source_logical);
                    }
                }
            }
        }

        cb_reserve_back(cb_base_tile, 2);
        const uint32_t base_tiles_addr = get_write_ptr(cb_base_tile);
        auto* base_full_tile = reinterpret_cast<float*>(base_tiles_addr);
        auto* base_tail_tile = reinterpret_cast<float*>(base_tiles_addr + row_major::kTileScalars * sizeof(float));

        for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
            for (uint32_t block = 0; block < kOutputBlocksPerRow; ++block) {
                auto* tile_block = block < 2 ? base_full_tile + row_major::tile_offset(row, block * kBlockElements)
                                             : base_tail_tile + row_major::tile_offset(row, 0);
                for (uint32_t lane = 0; lane < kBlockElements; ++lane) {
                    const int32_t output_logical =
                        group_base + static_cast<int32_t>((row * kOutputBlocksPerRow + block) * kBlockElements + lane);
                    tile_block[lane] = stream_value_or_zero(base, output_logical);
                }
            }
        }

        cb_push_back(cb_src_tile0, 1);
        cb_push_back(cb_src_tile1, 1);
        cb_push_back(cb_base_tile, 2);
    }
}

template <typename Scheme, uint32_t StepIndex, uint32_t PuIndex>
ALWI void run_reader_steps(
    const uint32_t cb_src_tile0,
    const uint32_t cb_src_tile1,
    const uint32_t cb_base_tile,
    const uint32_t cb_sync,
    LocalStream& even_cur,
    LocalStream& even_free,
    LocalStream& odd_cur,
    LocalStream& odd_free) {
    if constexpr (StepIndex < Scheme::num_steps) {
        using Step = SchemeStep<Scheme, StepIndex>;
        if constexpr (Step::type == StepType::kSwap) {
            swap_streams(even_cur, odd_cur);
            swap_streams(even_free, odd_free);
            run_reader_steps<Scheme, StepIndex + 1, PuIndex>(
                cb_src_tile0, cb_src_tile1, cb_base_tile, cb_sync, even_cur, even_free, odd_cur, odd_free);
        } else if constexpr (Step::type == StepType::kPredict || Step::type == StepType::kUpdate) {
            if constexpr (PuIndex > 0) {
                cb_wait_front(cb_sync, 1);
                cb_pop_front(cb_sync, 1);
            }

            constexpr uint32_t arg_base = kCommonArgCount + PuIndex * kStepConfigWords;
            const int32_t target_lo = i32_arg(arg_base);
            const uint32_t target_length = get_arg_val<uint32_t>(arg_base + 1);
            const int32_t target_group_begin = i32_arg(arg_base + 2);
            const uint32_t target_group_count = get_arg_val<uint32_t>(arg_base + 3);

            if constexpr (Step::type == StepType::kPredict) {
                emit_predict_update_tiles<Step>(
                    even_cur,
                    odd_cur,
                    cb_src_tile0,
                    cb_src_tile1,
                    cb_base_tile,
                    target_group_begin,
                    target_group_count);
                odd_free.lo = target_lo;
                odd_free.length = target_length;
                swap_streams(odd_cur, odd_free);
            } else {
                emit_predict_update_tiles<Step>(
                    odd_cur,
                    even_cur,
                    cb_src_tile0,
                    cb_src_tile1,
                    cb_base_tile,
                    target_group_begin,
                    target_group_count);
                even_free.lo = target_lo;
                even_free.length = target_length;
                swap_streams(even_cur, even_free);
            }

            run_reader_steps<Scheme, StepIndex + 1, PuIndex + 1>(
                cb_src_tile0, cb_src_tile1, cb_base_tile, cb_sync, even_cur, even_free, odd_cur, odd_free);
        } else {
            run_reader_steps<Scheme, StepIndex + 1, PuIndex>(
                cb_src_tile0, cb_src_tile1, cb_base_tile, cb_sync, even_cur, even_free, odd_cur, odd_free);
        }
    }
}

}  // namespace

template <typename Scheme>
void chunk_lwt_reader() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_length = get_arg_val<uint32_t>(1);
    const int32_t init_even_lo = i32_arg(7);
    const uint32_t init_even_length = get_arg_val<uint32_t>(8);
    const int32_t init_odd_lo = i32_arg(9);
    const uint32_t init_odd_length = get_arg_val<uint32_t>(10);

    constexpr uint32_t cb_src_tile0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_src_tile1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base_tile = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(3);
    constexpr uint32_t cb_input_cache = get_compile_time_arg_val(4);
    constexpr uint32_t cb_even_a = get_compile_time_arg_val(5);
    constexpr uint32_t cb_even_b = get_compile_time_arg_val(6);
    constexpr uint32_t cb_odd_a = get_compile_time_arg_val(7);
    constexpr uint32_t cb_odd_b = get_compile_time_arg_val(8);
    constexpr auto input_args = TensorAccessorArgs<9>();

    const auto input = TensorAccessor(input_args, input_addr, ttwv::device_protocol::kStickBytes);

    ttwv::kernels::primitives::StickReadCache cache{
        cb_input_cache,
        ttwv::device_protocol::kStickBytes,
        ttwv::kStickWidth,
        ttwv::device_protocol::kPadSplitCacheStickCount,
        ttwv::kernels::primitives::kInvalidStick,
        0,
        false};

    const uint32_t even_a_addr = get_write_ptr(cb_even_a);
    const uint32_t even_b_addr = get_write_ptr(cb_even_b);
    const uint32_t odd_a_addr = get_write_ptr(cb_odd_a);
    const uint32_t odd_b_addr = get_write_ptr(cb_odd_b);

    constexpr int32_t pad = static_cast<int32_t>(Scheme::tap_size - 1);
    fill_initial_stream(
        input, cache, input_length, init_even_lo, init_even_length, Scheme::delay_even, 0, pad, even_a_addr);
    fill_initial_stream(
        input, cache, input_length, init_odd_lo, init_odd_length, Scheme::delay_odd, 1, pad, odd_a_addr);
    ttwv::kernels::primitives::release_cache(cache);

    LocalStream even_cur{.addr = even_a_addr, .lo = init_even_lo, .length = init_even_length};
    LocalStream even_free{.addr = even_b_addr, .lo = 0, .length = 0};
    LocalStream odd_cur{.addr = odd_a_addr, .lo = init_odd_lo, .length = init_odd_length};
    LocalStream odd_free{.addr = odd_b_addr, .lo = 0, .length = 0};

    run_reader_steps<Scheme, 0, 0>(
        cb_src_tile0, cb_src_tile1, cb_base_tile, cb_sync, even_cur, even_free, odd_cur, odd_free);
}

}  // namespace ttwv::kernels

void kernel_main() { ttwv::kernels::chunk_lwt_reader<TTWV_LWT_SCHEME_TYPE>(); }
