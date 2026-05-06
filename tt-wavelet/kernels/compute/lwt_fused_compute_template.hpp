#pragma once

#include <array>
#include <cstdint>

#include "../../tt_wavelet/include/lifting/static_scheme.hpp"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "lwt_compute_utils.hpp"

namespace ttwv::kernels {

constexpr uint32_t kDstInput0 = 0;
constexpr uint32_t kDstInput1 = 1;
constexpr uint32_t kDstOutput = 2;
constexpr uint32_t kDstTailOutput = 3;
constexpr uint32_t kDstScale = 0;

template <uint32_t K>
inline void run_predict_update_step(
    const uint32_t cb_input0,
    const uint32_t cb_input1,
    const uint32_t cb_base,
    const uint32_t cb_output,
    const std::array<uint32_t, K> h_coeffs,
    const uint32_t output_group_count) {
    static_assert(K > 0, "Predict/update steps must have at least one coefficient");
    static_assert(K <= ttwv::device_protocol::step_coeff_capacity, "Step coefficient count exceeds device capacity");

    for (uint32_t group = 0; group < output_group_count; ++group) {
        tile_regs_acquire();

        cb_wait_front(cb_input0, 1);
        copy_tile_to_dst_init_short(cb_input0);
        copy_tile(cb_input0, 0, kDstInput0);
        cb_pop_front(cb_input0, 1);

        cb_wait_front(cb_input1, 1);
        copy_tile_to_dst_init_short(cb_input1);
        copy_tile(cb_input1, 0, kDstInput1);
        cb_pop_front(cb_input1, 1);

        cb_wait_front(cb_base, 1);
        copy_tile_to_dst_init_short(cb_base);
        copy_tile(cb_base, 0, kDstOutput);
        cb_pop_front(cb_base, 1);

        cb_wait_front(cb_base, 1);
        copy_tile_to_dst_init_short(cb_base);
        copy_tile(cb_base, 0, kDstTailOutput);
        cb_pop_front(cb_base, 1);

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

inline void run_scale_step(
    const uint32_t cb_input,
    const uint32_t cb_output,
    const uint32_t scalar_packed,
    const uint32_t output_group_count) {
    for (uint32_t group = 0; group < output_group_count; ++group) {
        cb_wait_front(cb_input, 2);
        cb_reserve_back(cb_output, 2);

        for (uint32_t tile = 0; tile < 2; ++tile) {
            tile_regs_acquire();

            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, tile, kDstScale);

            scale_tile(kDstScale, scalar_packed);

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(kDstScale, cb_output, tile);
            tile_regs_release();
        }

        cb_pop_front(cb_input, 2);
        cb_push_back(cb_output, 2);
    }
}

template <typename Scheme, uint32_t StepIndex, uint32_t ExecutableIndex>
inline void run_static_steps(
    const uint32_t cb_input0, const uint32_t cb_input1, const uint32_t cb_base, const uint32_t cb_output) {
    if constexpr (StepIndex < Scheme::num_steps) {
        using Step = SchemeStep<Scheme, StepIndex>;
        if constexpr (Step::type == StepType::kSwap) {
            run_static_steps<Scheme, StepIndex + 1, ExecutableIndex>(cb_input0, cb_input1, cb_base, cb_output);
        } else if constexpr (Step::type == StepType::kPredict || Step::type == StepType::kUpdate) {
            const uint32_t output_group_count = get_arg_val<uint32_t>(1 + ExecutableIndex);
            run_predict_update_step<Step::k>(
                cb_input0, cb_input1, cb_base, cb_output, Step::coeff_bits, output_group_count);
            run_static_steps<Scheme, StepIndex + 1, ExecutableIndex + 1>(cb_input0, cb_input1, cb_base, cb_output);
        } else if constexpr (Step::type == StepType::kScaleEven || Step::type == StepType::kScaleOdd) {
            static_assert(Step::k == 1, "Scale steps must have exactly one coefficient");
            const uint32_t output_group_count = get_arg_val<uint32_t>(1 + ExecutableIndex);
            run_scale_step(cb_base, cb_output, Step::coeff_bits[0], output_group_count);
            run_static_steps<Scheme, StepIndex + 1, ExecutableIndex + 1>(cb_input0, cb_input1, cb_base, cb_output);
        } else {
            static_assert(Step::type == StepType::kSwap, "Unsupported static lifting step type");
        }
    }
}

template <typename Scheme>
void lwt_fused_compute() {
    constexpr uint32_t cb_input0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output = get_compile_time_arg_val(3);

    ckernel::init_sfpu(cb_base, cb_output);
    run_static_steps<Scheme, 0, 0>(cb_input0, cb_input1, cb_base, cb_output);
}

}  // namespace ttwv::kernels
