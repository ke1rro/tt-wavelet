#include <array>
#include <cstdint>
#include <utility>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "lwt_compute_utils.hpp"

namespace {

constexpr uint32_t kArgsPerStep = 1 + ttwv::device_protocol::step_desc_word_count;
constexpr uint32_t kDstEven0 = 0;
constexpr uint32_t kDstEven1 = 1;
constexpr uint32_t kDstOdd0 = 2;
constexpr uint32_t kDstOdd1 = 3;
constexpr uint32_t kDstTmp0 = 4;
constexpr uint32_t kDstTmp1 = 5;
constexpr uint32_t kDstTmpU = 6;

constexpr uint8_t kStepTypePredict = 0;
constexpr uint8_t kStepTypeUpdate = 1;
constexpr uint8_t kStepTypeScaleEven = 2;
constexpr uint8_t kStepTypeScaleOdd = 3;
constexpr uint8_t kStepTypeSwap = 4;

template <uint32_t K>
inline void step_predict(
    const uint32_t cb_even,
    const uint32_t cb_odd,
    uint32_t& even_shift,
    uint32_t& odd_shift,
    const uint32_t splice_number,
    const uint32_t kernel_shift,
    const std::array<uint32_t, K>& h_coeffs) {
    static_assert(K > 1 && K <= 17, "K must be in the range [2, 17]");

    uint32_t predict_shift = even_shift + kernel_shift + K - 1;

    cb_wait_front(cb_even, splice_number * 2);
    cb_wait_front(cb_odd, splice_number * 2);
    cb_reserve_back(cb_even, splice_number * 2);
    cb_reserve_back(cb_odd, splice_number * 2);

    // Forward
    for (uint32_t splice = 0; splice < splice_number; splice++) {
        tile_regs_acquire();

        copy_tile_to_dst_init_short(cb_even);
        copy_tile(cb_even, splice * 2, kDstEven0);
        copy_tile(cb_even, splice * 2 + 1, kDstEven1);
        copy_tile_to_dst_init_short(cb_odd);
        copy_tile(cb_odd, splice * 2, kDstOdd0);
        copy_tile(cb_odd, splice * 2 + 1, kDstOdd1);

        splice_ops_init();
        if (predict_shift < odd_shift) {
            // Branch 1: add to even samples
            shift_splice(kDstEven0, kDstEven1, odd_shift - predict_shift);
            recover1_splice(kDstEven0, kDstEven1, kDstTmpU);
        } else {
            // Branch 2: add to odd samples
            shift_splice(kDstOdd0, kDstOdd1, predict_shift - odd_shift);
            recover1_splice(kDstOdd0, kDstOdd1, kDstTmpU);
        }

        // predict <- even * h
        hstencil_spline<K>(h_coeffs, kDstEven0, kDstEven1, kDstTmp0, kDstTmp1);

        // odd <- predict + odd
        add_binary_tile_init();
        add_binary_tile(kDstTmp0, kDstOdd0, kDstOdd0);
        add_binary_tile(kDstTmp1, kDstOdd1, kDstOdd1);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(kDstOdd0, cb_odd, splice * 2);
        pack_tile(kDstOdd1, cb_odd, splice * 2 + 1);
        pack_tile(kDstEven0, cb_even, splice * 2);
        pack_tile(kDstEven1, cb_even, splice * 2 + 1);

        tile_regs_release();
    }

    cb_pop_front(cb_even, splice_number * 2);
    cb_pop_front(cb_odd, splice_number * 2);
    cb_push_back(cb_even, splice_number * 2);
    cb_push_back(cb_odd, splice_number * 2);

    cb_wait_front(cb_even, splice_number * 2);
    cb_wait_front(cb_odd, splice_number * 2);

    // Backward
    for (uint32_t splice = splice_number + 1; splice > 0; splice--) {
        tile_regs_acquire();

        copy_tile_to_dst_init_short(cb_odd);
        copy_tile(cb_odd, (splice - 1) * 2, kDstOdd0);
        copy_tile(cb_odd, (splice - 1) * 2 + 1, kDstOdd1);

        splice_ops_init();
        recover2_splice(kDstOdd0, kDstOdd1, kDstTmpU);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(kDstOdd0, cb_odd, (splice - 1) * 2);
        pack_tile(kDstOdd1, cb_odd, (splice - 1) * 2 + 1);

        tile_regs_release();
    }

    cb_pop_front(cb_odd, splice_number * 2);
    cb_push_back(cb_odd, splice_number * 2);

    if (predict_shift < odd_shift) {
        odd_shift = predict_shift;
    }
}

inline void step_scale(
    const uint32_t cb_even,
    const uint32_t splice_number,
    const uint32_t s_packed) {

    cb_wait_front(cb_even, splice_number * 2);
    cb_reserve_back(cb_even, splice_number * 2);

    for (uint32_t splice = 0; splice < splice_number; splice++) {
        tile_regs_acquire();

        copy_tile_to_dst_init_short(cb_even);
        copy_tile(cb_even, splice * 2, kDstEven0);
        copy_tile(cb_even, splice * 2 + 1, kDstEven1);

        binop_with_scalar_tile_init();
        mul_unary_tile(kDstEven0, s_packed);
        mul_unary_tile(kDstEven1, s_packed);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(kDstEven0, cb_even, splice * 2);
        pack_tile(kDstEven1, cb_even, splice * 2 + 1);

        tile_regs_release();
    }

    cb_pop_front(cb_even, splice_number * 2);
    cb_push_back(cb_even, splice_number * 2);
}

template <uint32_t arg_base, uint32_t num_steps>
inline void unroll_step(
    const uint32_t cb_even,
    const uint32_t cb_odd,
    const bool swapped,
    const uint32_t cb_out,
    const uint32_t splice_number,
    const uint32_t even_shift,
    const uint32_t odd_shift,
) {

    if constexpr (num_steps > 0) {
        constexpr uint32_t kMeta = get_compile_time_arg_val(arg_base);
        constexpr uint8_t kType = kMeta & 0x7;

        if constexpr(kType == kStepTypeScaleEven) {
            step_scale(cb_even, splice_number, get_compile_time_arg_val(arg_base + 1));
            unroll_step<arg_base + 2, num_steps - 1>(cb_even, cb_odd, swapped, cb_out, splice_number, even_shift, odd_shift);
        } else if constexpr(kType == kStepTypeScaleOdd) {
            step_scale(cb_odd, splice_number, get_compile_time_arg_val(arg_base + 1));
            unroll_step<arg_base + 2, num_steps - 1>(cb_even, cb_odd, swapped, cb_out, splice_number, even_shift, odd_shift);
        } else if constexpr(kType == kStepTypeSwap) {
            unroll_step<arg_base + 1, num_steps - 1>(cb_odd, cb_even, !swapped, cb_out, splice_number, odd_shift, even_shift);
        } else {
            constexpr uint32_t K = (kMeta >> 3);
            constexpr uint32_t kernel_shift = get_compile_time_arg_val(arg_base + 1);

            std::array<uint32_t, K> h_coeffs{};
#pragma unroll 17
            for (uint32_t j = 0; j < K; ++j) {
                h_coeffs[j] = get_compile_time_arg_val(arg_base + 2 + j);
            }

            if (kType == kStepTypePredict) {
                step_predict<K>(cb_even, cb_odd, even_shift, odd_shift, splice_number, kernel_shift, h_coeffs);
            } else if (kType == kStepTypeUpdate) {
                step_predict<K>(cb_odd, cb_even, odd_shift, even_shift, splice_number, kernel_shift, h_coeffs);
            }

            unroll_step<arg_base + 2 + K, num_steps - 1>(cb_even, cb_odd, swapped, cb_out, splice_number, even_shift, odd_shift);
        }
    } else {
        cb_reserve_back(cb_out, 1);

        uint32_t write_ptr = get_write_ptr(cb_out);  

        volatile uint32_t* ptr32 = (volatile uint32_t*)write_ptr;  
        ptr32[0] = even_shift;  
        ptr32[1] = odd_shift;

        volatile uint8_t* ptr8 = (volatile uint8_t*)(write_ptr + 8);  
        ptr8[0] = swapped ? 1 : 0;

        cb_push_back(cb_out, 1);
    }
}

}

void kernel_main() {
    constexpr uint32_t cb_even = get_named_compile_time_arg_val("cb_even");
    constexpr uint32_t cb_odd = get_named_compile_time_arg_val("cb_odd");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t even_shift = get_named_compile_time_arg_val("even_delay");
    constexpr uint32_t odd_shift = get_named_compile_time_arg_val("odd_delay");
    constexpr uint32_t num_steps = get_named_compile_time_arg_val("num_steps");

    const uint32_t splice_number = get_arg_val<uint32_t>(0);

    ckernel::init_sfpu(cb_even, cb_odd);

    unroll_step<0, num_steps>(cb_even, cb_odd, false, cb_out, splice_number, even_shift, odd_shift);
}
