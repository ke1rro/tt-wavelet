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

#ifndef TTWV_FUSE_TERMINAL_SCALE
#define TTWV_FUSE_TERMINAL_SCALE 0
#endif

namespace ttwv::kernels {

constexpr uint32_t kDstInput0 = 0;
constexpr uint32_t kDstInput1 = 1;
constexpr uint32_t kDstOutput = 2;
constexpr uint32_t kDstTailOutput = 3;
constexpr uint32_t kDstScale = 0;
constexpr bool kFuseTerminalScale = TTWV_FUSE_TERMINAL_SCALE != 0;

template <typename Scheme, uint32_t Index = 0>
constexpr uint32_t last_predict_update_step_index() noexcept {
    if constexpr (Index >= Scheme::num_steps) {
        return Scheme::num_steps;
    } else {
        constexpr uint32_t later = last_predict_update_step_index<Scheme, Index + 1>();
        using Step = SchemeStep<Scheme, Index>;
        if constexpr (later < Scheme::num_steps) {
            return later;
        } else if constexpr (Step::type == StepType::kPredict || Step::type == StepType::kUpdate) {
            return Index;
        } else {
            return Scheme::num_steps;
        }
    }
}

template <typename Scheme, uint32_t Index>
constexpr uint32_t swap_count_from() noexcept {
    if constexpr (Index >= Scheme::num_steps) {
        return 0;
    } else {
        using Step = SchemeStep<Scheme, Index>;
        return (Step::type == StepType::kSwap ? 1U : 0U) + swap_count_from<Scheme, Index + 1>();
    }
}

template <typename Scheme>
constexpr StepType fused_terminal_scale_type() noexcept {
    constexpr uint32_t last_step = last_predict_update_step_index<Scheme>();
    static_assert(last_step < Scheme::num_steps, "ConeStreamed terminal-scale fusion requires predict/update");
    using LastStep = SchemeStep<Scheme, last_step>;
    constexpr bool target_even = LastStep::type == StepType::kUpdate;
    constexpr bool swapped = (swap_count_from<Scheme, last_step + 1>() & 1U) != 0;
    constexpr bool final_even = target_even != swapped;
    return final_even ? StepType::kScaleEven : StepType::kScaleOdd;
}

template <typename Scheme, StepType ScaleType, uint32_t Index = 0>
constexpr uint32_t terminal_scale_bits() noexcept {
    if constexpr (Index >= Scheme::num_steps) {
        return 0;
    } else {
        using Step = SchemeStep<Scheme, Index>;
        if constexpr (Step::type == ScaleType) {
            static_assert(Step::k == 1, "Terminal scale must contain exactly one coefficient");
            return Step::coeff_bits[0];
        } else {
            return terminal_scale_bits<Scheme, ScaleType, Index + 1>();
        }
    }
}

template <uint32_t K, bool FuseTerminalScale, uint32_t TerminalScalePacked>
inline void run_predict_update_step(
    const uint32_t cb_input0,
    const uint32_t cb_input1,
    const uint32_t cb_base,
    const uint32_t cb_output,
    const std::array<uint32_t, K> h_coeffs,
    const uint32_t output_group_count) {
    static_assert(K > 0, "Predict/update steps must have at least one coefficient");
    static_assert(K <= device_protocol::kStepCoeffCapacity, "Step coefficient count exceeds device capacity");

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
        if constexpr (FuseTerminalScale) {
            scale_tile(kDstOutput, TerminalScalePacked);
            scale_tile(kDstTailOutput, TerminalScalePacked);
        }

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

template <typename Scheme, bool FuseTerminalScale, uint32_t StepIndex, uint32_t ExecutableIndex>
inline void run_static_steps(
    const uint32_t cb_input0,
    const uint32_t cb_input1,
    const uint32_t cb_base,
    const uint32_t cb_output,
    const uint32_t runtime_arg_base) {
    if constexpr (StepIndex < Scheme::num_steps) {
        using Step = SchemeStep<Scheme, StepIndex>;
        if constexpr (Step::type == StepType::kSwap) {
            run_static_steps<Scheme, FuseTerminalScale, StepIndex + 1, ExecutableIndex>(
                cb_input0, cb_input1, cb_base, cb_output, runtime_arg_base);
        } else if constexpr (Step::type == StepType::kPredict || Step::type == StepType::kUpdate) {
            const uint32_t output_group_count = get_arg_val<uint32_t>(runtime_arg_base + ExecutableIndex);
            constexpr uint32_t last_predict_update = last_predict_update_step_index<Scheme>();
            constexpr bool fuse_terminal_scale = FuseTerminalScale && StepIndex == last_predict_update;
            constexpr StepType scale_type = fused_terminal_scale_type<Scheme>();
            constexpr uint32_t scale_bits = terminal_scale_bits<Scheme, scale_type>();
            run_predict_update_step<Step::k, fuse_terminal_scale, scale_bits>(
                cb_input0, cb_input1, cb_base, cb_output, Step::coeff_bits, output_group_count);
            run_static_steps<Scheme, FuseTerminalScale, StepIndex + 1, ExecutableIndex + 1>(
                cb_input0, cb_input1, cb_base, cb_output, runtime_arg_base);
        } else if constexpr (Step::type == StepType::kScaleEven || Step::type == StepType::kScaleOdd) {
            if constexpr (FuseTerminalScale) {
                if constexpr (Step::type == fused_terminal_scale_type<Scheme>()) {
                    run_static_steps<Scheme, FuseTerminalScale, StepIndex + 1, ExecutableIndex>(
                        cb_input0, cb_input1, cb_base, cb_output, runtime_arg_base);
                } else {
                    static_assert(Step::k == 1, "Scale steps must have exactly one coefficient");
                    const uint32_t output_group_count = get_arg_val<uint32_t>(runtime_arg_base + ExecutableIndex);
                    run_scale_step(cb_base, cb_output, Step::coeff_bits[0], output_group_count);
                    run_static_steps<Scheme, FuseTerminalScale, StepIndex + 1, ExecutableIndex + 1>(
                        cb_input0, cb_input1, cb_base, cb_output, runtime_arg_base);
                }
            } else {
                static_assert(Step::k == 1, "Scale steps must have exactly one coefficient");
                const uint32_t output_group_count = get_arg_val<uint32_t>(runtime_arg_base + ExecutableIndex);
                run_scale_step(cb_base, cb_output, Step::coeff_bits[0], output_group_count);
                run_static_steps<Scheme, FuseTerminalScale, StepIndex + 1, ExecutableIndex + 1>(
                    cb_input0, cb_input1, cb_base, cb_output, runtime_arg_base);
            }
        }
    }
}

template <typename Scheme>
void lwt_cone_compute() {
    constexpr uint32_t cb_input0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output = get_compile_time_arg_val(3);
    constexpr uint32_t route_count = executable_step_count<Scheme>() - (kFuseTerminalScale ? 1U : 0U);
    const uint32_t chunk_count = get_arg_val<uint32_t>(0);

    ckernel::init_sfpu(cb_base, cb_output);
    for (uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
        run_static_steps<Scheme, kFuseTerminalScale, 0, 0>(
            cb_input0, cb_input1, cb_base, cb_output, 1 + chunk * route_count);
    }
}

}  // namespace ttwv::kernels

void kernel_main() { ttwv::kernels::lwt_cone_compute<TTWV_LWT_SCHEME_TYPE>(); }
