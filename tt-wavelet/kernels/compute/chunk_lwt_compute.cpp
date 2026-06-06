#ifndef TTWV_LWT_SCHEME_HEADER
#error "TTWV_LWT_SCHEME_HEADER must identify the generated lifting scheme header"
#endif

#ifndef TTWV_LWT_SCHEME_TYPE
#error "TTWV_LWT_SCHEME_TYPE must identify the generated lifting scheme type"
#endif

#include <array>
#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "lwt_compute_utils.hpp"
#include TTWV_LWT_SCHEME_HEADER

namespace ttwv::kernels {

constexpr uint32_t kDstInput0 = 0;
constexpr uint32_t kDstInput1 = 1;
constexpr uint32_t kDstOutput = 2;
constexpr uint32_t kDstTailOutput = 3;

template <uint32_t K>
inline void run_predict_update_step(
    const uint32_t cb_input0,
    const uint32_t cb_input1,
    const uint32_t cb_base,
    const uint32_t cb_output,
    const std::array<uint32_t, K> h_coeffs,
    const uint32_t output_group_count) {
    static_assert(K > 0, "Predict/update steps must have at least one coefficient");
    static_assert(K <= ttwv::device_protocol::kStepCoeffCapacity, "Step coefficient count exceeds device capacity");

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

template <typename Scheme, uint32_t StepIndex, uint32_t PuIndex>
inline void run_static_predict_update_steps(
    const uint32_t cb_input0, const uint32_t cb_input1, const uint32_t cb_base, const uint32_t cb_output) {
    if constexpr (StepIndex < Scheme::num_steps) {
        using Step = SchemeStep<Scheme, StepIndex>;
        if constexpr (Step::type == StepType::kPredict || Step::type == StepType::kUpdate) {
            const uint32_t output_group_count = get_arg_val<uint32_t>(PuIndex);
            run_predict_update_step<Step::k>(
                cb_input0, cb_input1, cb_base, cb_output, Step::coeff_bits, output_group_count);
            run_static_predict_update_steps<Scheme, StepIndex + 1, PuIndex + 1>(
                cb_input0, cb_input1, cb_base, cb_output);
        } else {
            run_static_predict_update_steps<Scheme, StepIndex + 1, PuIndex>(cb_input0, cb_input1, cb_base, cb_output);
        }
    }
}

template <typename Scheme>
void chunk_lwt_compute() {
    constexpr uint32_t cb_input0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output = get_compile_time_arg_val(3);

    ckernel::init_sfpu(cb_base, cb_output);
    run_static_predict_update_steps<Scheme, 0, 0>(cb_input0, cb_input1, cb_base, cb_output);
}

}  // namespace ttwv::kernels

void kernel_main() { ttwv::kernels::chunk_lwt_compute<TTWV_LWT_SCHEME_TYPE>(); }
