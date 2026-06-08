#include <array>
#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "lwt_compute_utils.hpp"

namespace ttwv::kernels {

constexpr uint32_t kDstInput0 = 0;
constexpr uint32_t kDstInput1 = 1;
constexpr uint32_t kDstOutput = 2;
constexpr uint32_t kDstTailOutput = 3;

template <uint32_t K>
void step_predict_update(
    const uint32_t cb_input,
    const uint32_t cb_base,
    const uint32_t cb_output,
    const std::array<uint32_t, K> h_coeffs,
    const uint32_t splice_number) {
    static_assert(K > 1, "Predict/update steps must have at least 2 coefficients");
    static_assert(K <= ttwv::device_protocol::kStepCoeffCapacity, "Step coefficient count exceeds device capacity");

    for (uint32_t splice = 0; splice < splice_number; ++splice) {
        tile_regs_acquire();

        cb_wait_front(cb_input, 2);
        copy_tile_to_dst_init_short(cb_input);
        copy_tile(cb_input, 0, kDstInput0);
        copy_tile(cb_input, 1, kDstInput1);
        cb_pop_front(cb_input, 2);

        cb_wait_front(cb_base, 2);
        copy_tile_to_dst_init_short(cb_base);
        copy_tile(cb_base, 0, kDstOutput);
        copy_tile(cb_base, 1, kDstTailOutput);
        cb_pop_front(cb_base, 2);

        hstencil_init();
        hstencil_plus_base_tile<K>(
            h_coeffs, kDstInput0, kDstInput1, kDstOutput, kDstTailOutput, kDstOutput, kDstTailOutput);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_output, 2);
        pack_tile(kDstOutput, cb_output, 0);
        pack_tile(kDstTailOutput, cb_output, 1);
        cb_push_back(cb_output, 2);

        tile_regs_release();
    }
}

void step_scale(
    const uint32_t cb_input,
    const uint32_t cb_output,
    const uint32_t scalar_packed,
    const uint32_t splice_number) {
    for (uint32_t splice = 0; splice < splice_number; ++splice) {
        tile_regs_acquire();

        cb_wait_front(cb_input, 2);
        copy_tile_to_dst_init_short(cb_input);
        copy_tile(cb_input, 0, kDstOutput);
        copy_tile(cb_input, 1, kDstTailOutput);
        cb_pop_front(cb_input, 2);

        scale_tile(kDstOutput, scalar_packed);
        scale_tile(kDstTailOutput, scalar_packed);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_output, 2);
        pack_tile(kDstOutput, cb_output, 0);
        pack_tile(kDstTailOutput, cb_output, 1);
        cb_push_back(cb_output, 2);

        tile_regs_release();
    }
}


template <uint32_t arg_base, uint32_t num_steps>
inline void unroll_step(
    const uint32_t cb_input,
    const uint32_t cb_base,
    const uint32_t cb_output,
) {

    if constexpr (num_steps > 0) {
        constexpr uint32_t kMeta = get_compile_time_arg_val(arg_base);
        constexpr uint8_t kType = kMeta & 0x7;

        constexpr uint32_t splice_number = get_compile_time_arg_val(arg_base + 1);

        if constexpr(kType == kStepTypeScaleEven || kType == kStepTypeScaleOdd) {
            constexpr uint32_t scalar_packed = get_compile_time_arg_val(arg_base + 2);

            step_scale(cb_input, cb_output, scalar_packed, splice_number);
            unroll_step<arg_base + 3, num_steps - 1>(cb_input, cb_base, cb_output);
        } else if constexpr(kType == kStepTypeSwap) {
            unroll_step<arg_base + 2, num_steps - 1>(cb_input, cb_base, cb_output);
        } else if constexpr(kType == kStepTypePredict || kType == kStepTypeUpdate) {
            constexpr uint32_t K = (kMeta >> 3);

            std::array<uint32_t, K> h_coeffs{};
#pragma unroll 17
            for (uint32_t j = 0; j < K; ++j) {
                h_coeffs[j] = get_compile_time_arg_val(arg_base + 2 + j);
            }

            step_predict_update<K>(cb_input, cb_base, cb_output, h_coeffs, splice_number);
            unroll_step<arg_base + 2 + K, num_steps - 1>(cb_input, cb_base, cb_output);
        } else {
            static_assert(false, "Unsupported step type");
        }
    }
}

} // namespace ttwv::kernels

void kernel_main() {
    constexpr uint32_t cb_input = get_named_compile_time_arg_val("cb_input");
    constexpr uint32_t cb_base = get_named_compile_time_arg_val("cb_base");
    constexpr uint32_t cb_output = get_named_compile_time_arg_val("cb_output");
    constexpr uint32_t num_steps = get_named_compile_time_arg_val("num_steps");

    ckernel::init_sfpu(cb_base, cb_output);

    ttwv::kernels::unroll_step<0, num_steps>(cb_input, cb_base, cb_output);
}
