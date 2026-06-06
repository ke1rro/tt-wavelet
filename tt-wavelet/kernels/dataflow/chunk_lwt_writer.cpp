#ifndef TTWV_LWT_SCHEME_HEADER
#error "TTWV_LWT_SCHEME_HEADER must identify the generated lifting scheme header"
#endif

#ifndef TTWV_LWT_SCHEME_TYPE
#error "TTWV_LWT_SCHEME_TYPE must identify the generated lifting scheme type"
#endif

#include <cstdint>

#include "../../tt_wavelet/include/device_protocol/lwt_config.hpp"
#include "../../tt_wavelet/include/lifting/static_scheme.hpp"
#include "../primitives/tile_row_major.hpp"
#include "api/dataflow/dataflow_api.h"
#include TTWV_LWT_SCHEME_HEADER

namespace ttwv::kernels {

namespace row_major = ttwv::kernels::primitives;

namespace {

constexpr uint32_t kBlockElements = ttwv::device_protocol::kLwtHalfStickElements;
constexpr uint32_t kRowsPerGroup = ttwv::device_protocol::kLwtRowsPerGroup;
constexpr uint32_t kOutputBlocksPerRow = ttwv::device_protocol::kLwtOutputBlocksPerRow;
constexpr uint32_t kGroupOutputElements = ttwv::device_protocol::kLwtGroupOutputElements;
constexpr uint32_t kStepConfigWords = 4;
constexpr uint32_t kCommonArgCount = 15;

struct LocalStream {
    uint32_t addr;
    int32_t lo;
    uint32_t length;
};

struct Interval {
    int32_t lo;
    uint32_t length;
};

ALWI int32_t i32_arg(const uint32_t index) { return static_cast<int32_t>(get_arg_val<uint32_t>(index)); }

ALWI bool is_empty(const Interval interval) { return interval.length == 0; }

ALWI int32_t interval_hi(const Interval interval) { return interval.lo + static_cast<int32_t>(interval.length) - 1; }

ALWI int32_t max_i32(const int32_t lhs, const int32_t rhs) { return lhs > rhs ? lhs : rhs; }

ALWI int32_t min_i32(const int32_t lhs, const int32_t rhs) { return lhs < rhs ? lhs : rhs; }

ALWI uint32_t ceil_div_u32(const uint32_t value, const uint32_t divisor) {
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
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

ALWI void store_stream_value(LocalStream& stream, const int32_t logical_index, const float value) {
    if (!contains(stream, logical_index)) {
        return;
    }
    auto* values = reinterpret_cast<float*>(stream.addr);
    values[static_cast<uint32_t>(logical_index - stream.lo)] = value;
}

ALWI void swap_streams(LocalStream& lhs, LocalStream& rhs) {
    const LocalStream tmp = lhs;
    lhs = rhs;
    rhs = tmp;
}

ALWI float float_from_bits(const uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } value{bits};
    return value.f;
}

ALWI void write_output_group_to_stream(
    LocalStream& target, const uint32_t cb_output, const int32_t group, const uint32_t tile_nbytes) {
    cb_wait_front(cb_output, 2);
    const uint32_t output_full_tile = get_read_ptr(cb_output);
    const uint32_t output_tail_tile = output_full_tile + tile_nbytes;
    const int32_t group_base = group * static_cast<int32_t>(kGroupOutputElements);

    for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
        for (uint32_t block = 0; block < kOutputBlocksPerRow; ++block) {
            const uint32_t tile_addr = block < 2 ? output_full_tile : output_tail_tile;
            const uint32_t tile_col = block < 2 ? block * kBlockElements : 0;
            const auto* tile_values = reinterpret_cast<const float*>(tile_addr);
            for (uint32_t lane = 0; lane < kBlockElements; ++lane) {
                const int32_t logical_index =
                    group_base + static_cast<int32_t>((row * kOutputBlocksPerRow + block) * kBlockElements + lane);
                const float value = tile_values[row_major::tile_offset(row, tile_col + lane)];
                store_stream_value(target, logical_index, value);
            }
        }
    }

    cb_pop_front(cb_output, 2);
}

template <typename DstAccessor>
ALWI void write_final_stream(
    const DstAccessor& dst,
    const uint32_t stick_l1_addr,
    const uint32_t final_length,
    const int32_t final_shift,
    const Interval owned,
    const LocalStream& stream,
    const float scale) {
    if (is_empty(owned) || final_length == 0) {
        return;
    }

    const int32_t owned_hi = interval_hi(owned);
    const int32_t stream_index_lo = max_i32(0, owned.lo - final_shift);
    const int32_t stream_index_hi = min_i32(static_cast<int32_t>(final_length) - 1, owned_hi - final_shift);
    if (stream_index_hi < stream_index_lo) {
        return;
    }

    const uint32_t first_stick = static_cast<uint32_t>(stream_index_lo) / ttwv::kStickWidth;
    const uint32_t last_stick = static_cast<uint32_t>(stream_index_hi) / ttwv::kStickWidth;
    auto* stick = reinterpret_cast<float*>(stick_l1_addr);

    for (uint32_t stick_id = first_stick; stick_id <= last_stick; ++stick_id) {
        const uint32_t stick_base_index = stick_id * ttwv::kStickWidth;
        for (uint32_t lane = 0; lane < ttwv::kStickWidth; ++lane) {
            const uint32_t stream_index = stick_base_index + lane;
            const int32_t logical_index = final_shift + static_cast<int32_t>(stream_index);
            const bool in_owned = logical_index >= owned.lo && logical_index <= owned_hi && stream_index < final_length;
            if (!in_owned) {
                continue;
            }
            stick[lane] = stream_value_or_zero(stream, logical_index) * scale;
            noc_async_write(
                stick_l1_addr + lane * static_cast<uint32_t>(sizeof(float)),
                dst.get_noc_addr(stick_id) + lane * static_cast<uint32_t>(sizeof(float)),
                sizeof(float));
        }
    }
}

template <typename Scheme, uint32_t StepIndex, uint32_t PuIndex>
ALWI void run_writer_steps(
    const uint32_t cb_output,
    const uint32_t cb_sync,
    LocalStream& even_cur,
    LocalStream& even_free,
    LocalStream& odd_cur,
    LocalStream& odd_free,
    float& even_scale,
    float& odd_scale) {
    if constexpr (StepIndex < Scheme::num_steps) {
        using Step = SchemeStep<Scheme, StepIndex>;
        if constexpr (Step::type == StepType::kSwap) {
            swap_streams(even_cur, odd_cur);
            swap_streams(even_free, odd_free);
            const float tmp = even_scale;
            even_scale = odd_scale;
            odd_scale = tmp;
            run_writer_steps<Scheme, StepIndex + 1, PuIndex>(
                cb_output, cb_sync, even_cur, even_free, odd_cur, odd_free, even_scale, odd_scale);
        } else if constexpr (Step::type == StepType::kPredict || Step::type == StepType::kUpdate) {
            constexpr uint32_t arg_base = kCommonArgCount + PuIndex * kStepConfigWords;
            const int32_t target_lo = i32_arg(arg_base);
            const uint32_t target_length = get_arg_val<uint32_t>(arg_base + 1);
            const int32_t target_group_begin = i32_arg(arg_base + 2);
            const uint32_t target_group_count = get_arg_val<uint32_t>(arg_base + 3);
            const uint32_t tile_nbytes = get_tile_size(cb_output);

            if constexpr (Step::type == StepType::kPredict) {
                odd_free.lo = target_lo;
                odd_free.length = target_length;
                for (uint32_t local_group = 0; local_group < target_group_count; ++local_group) {
                    write_output_group_to_stream(
                        odd_free, cb_output, target_group_begin + static_cast<int32_t>(local_group), tile_nbytes);
                }
                swap_streams(odd_cur, odd_free);
            } else {
                even_free.lo = target_lo;
                even_free.length = target_length;
                for (uint32_t local_group = 0; local_group < target_group_count; ++local_group) {
                    write_output_group_to_stream(
                        even_free, cb_output, target_group_begin + static_cast<int32_t>(local_group), tile_nbytes);
                }
                swap_streams(even_cur, even_free);
            }

            cb_reserve_back(cb_sync, 1);
            cb_push_back(cb_sync, 1);

            run_writer_steps<Scheme, StepIndex + 1, PuIndex + 1>(
                cb_output, cb_sync, even_cur, even_free, odd_cur, odd_free, even_scale, odd_scale);
        } else if constexpr (Step::type == StepType::kScaleEven) {
            static_assert(Step::k == 1, "ScaleEven steps must have exactly one coefficient");
            even_scale *= float_from_bits(Step::coeff_bits[0]);
            run_writer_steps<Scheme, StepIndex + 1, PuIndex>(
                cb_output, cb_sync, even_cur, even_free, odd_cur, odd_free, even_scale, odd_scale);
        } else if constexpr (Step::type == StepType::kScaleOdd) {
            static_assert(Step::k == 1, "ScaleOdd steps must have exactly one coefficient");
            odd_scale *= float_from_bits(Step::coeff_bits[0]);
            run_writer_steps<Scheme, StepIndex + 1, PuIndex>(
                cb_output, cb_sync, even_cur, even_free, odd_cur, odd_free, even_scale, odd_scale);
        }
    }
}

}  // namespace

template <typename Scheme>
void chunk_lwt_writer() {
    const uint32_t approx_addr = get_arg_val<uint32_t>(0);
    const uint32_t detail_addr = get_arg_val<uint32_t>(1);
    const int32_t final_even_shift = i32_arg(3);
    const uint32_t final_even_length = get_arg_val<uint32_t>(4);
    const int32_t final_odd_shift = i32_arg(5);
    const uint32_t final_odd_length = get_arg_val<uint32_t>(6);
    const Interval final_even_owned{.lo = i32_arg(7), .length = get_arg_val<uint32_t>(8)};
    const Interval final_odd_owned{.lo = i32_arg(9), .length = get_arg_val<uint32_t>(10)};
    const int32_t init_even_lo = i32_arg(11);
    const uint32_t init_even_length = get_arg_val<uint32_t>(12);
    const int32_t init_odd_lo = i32_arg(13);
    const uint32_t init_odd_length = get_arg_val<uint32_t>(14);

    constexpr uint32_t cb_output = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(1);
    constexpr uint32_t cb_even_a = get_compile_time_arg_val(2);
    constexpr uint32_t cb_even_b = get_compile_time_arg_val(3);
    constexpr uint32_t cb_odd_a = get_compile_time_arg_val(4);
    constexpr uint32_t cb_odd_b = get_compile_time_arg_val(5);
    constexpr uint32_t cb_approx_stick = get_compile_time_arg_val(6);
    constexpr uint32_t cb_detail_stick = get_compile_time_arg_val(7);
    constexpr auto approx_args = TensorAccessorArgs<8>();
    constexpr auto detail_args = TensorAccessorArgs<approx_args.next_compile_time_args_offset()>();

    const auto approx = TensorAccessor(approx_args, approx_addr, ttwv::device_protocol::kStickBytes);
    const auto detail = TensorAccessor(detail_args, detail_addr, ttwv::device_protocol::kStickBytes);

    LocalStream even_cur{.addr = get_write_ptr(cb_even_a), .lo = init_even_lo, .length = init_even_length};
    LocalStream even_free{.addr = get_write_ptr(cb_even_b), .lo = 0, .length = 0};
    LocalStream odd_cur{.addr = get_write_ptr(cb_odd_a), .lo = init_odd_lo, .length = init_odd_length};
    LocalStream odd_free{.addr = get_write_ptr(cb_odd_b), .lo = 0, .length = 0};
    float even_scale = 1.0F;
    float odd_scale = 1.0F;

    run_writer_steps<Scheme, 0, 0>(cb_output, cb_sync, even_cur, even_free, odd_cur, odd_free, even_scale, odd_scale);

    const uint32_t approx_stick_addr = get_write_ptr(cb_approx_stick);
    const uint32_t detail_stick_addr = get_write_ptr(cb_detail_stick);
    write_final_stream(
        approx, approx_stick_addr, final_even_length, final_even_shift, final_even_owned, even_cur, even_scale);
    write_final_stream(
        detail, detail_stick_addr, final_odd_length, final_odd_shift, final_odd_owned, odd_cur, odd_scale);
    noc_async_write_barrier();
}

}  // namespace ttwv::kernels

void kernel_main() { ttwv::kernels::chunk_lwt_writer<TTWV_LWT_SCHEME_TYPE>(); }
