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

// Seven 32x16 FP32 tiles fit in the four 32x32 slots available under
// tile_regs_acquire().  Place outputs on even narrow-tile indices so the
// standard packer can address them through W indices 0, 1, and 2.
constexpr uint32_t kDstBase0 = 0;
constexpr uint32_t kDstSource0 = 1;
constexpr uint32_t kDstBase1 = 2;
constexpr uint32_t kDstSource1 = 3;
constexpr uint32_t kDstBase2 = 4;
constexpr uint32_t kDstSource2 = 5;
constexpr uint32_t kDstSource3 = 6;
constexpr uint32_t kPackBase0 = 0;
constexpr uint32_t kPackBase1 = 1;
constexpr uint32_t kPackBase2 = 2;
constexpr uint32_t kDstScale = 0;
constexpr bool kFuseTerminalScale = TTWV_FUSE_TERMINAL_SCALE != 0;

#ifdef TRISC_MATH
ALWI void copy_narrow_tile_math(const uint32_t dst_tile_index) {
    math::math_unpack_to_dest_math_ready();
    math::set_dst_write_addr<DstTileShape::Tile32x16, UnpackDestination::DestReg>(dst_tile_index);
    math::math_unpack_to_dest_tile_ready();
}
#endif

// copy_tile_init_short() still configures the unpack MOP from the 32x16 CB
// metadata (two faces).  Only the FP32 math-side Dst address needs overriding:
// Metalium's generic direct-copy path currently hardcodes a 32x32 Dst stride.
ALWI void copy_narrow_tile(const uint32_t cb, const uint32_t input_tile_index, const uint32_t dst_tile_index) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        cb, input_tile_index)));
    MATH((copy_narrow_tile_math(dst_tile_index)));
}

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

        cb_wait_front(cb_input0, 2);
        copy_tile_to_dst_init_short(cb_input0);
        copy_narrow_tile(cb_input0, 0, kDstSource0);
        copy_narrow_tile(cb_input0, 1, kDstSource1);
        cb_pop_front(cb_input0, 2);

        cb_wait_front(cb_input1, 2);
        copy_tile_to_dst_init_short(cb_input1);
        copy_narrow_tile(cb_input1, 0, kDstSource2);
        copy_narrow_tile(cb_input1, 1, kDstSource3);
        cb_pop_front(cb_input1, 2);

        cb_wait_front(cb_base, 3);
        copy_tile_to_dst_init_short(cb_base);
        copy_narrow_tile(cb_base, 0, kDstBase0);
        copy_narrow_tile(cb_base, 1, kDstBase1);
        copy_narrow_tile(cb_base, 2, kDstBase2);
        cb_pop_front(cb_base, 3);

        hstencil_init();
        hstencil_plus_base_narrow_tiles<K>(
            h_coeffs, kDstSource0, kDstSource1, kDstSource2, kDstSource3, kDstBase0, kDstBase1, kDstBase2);

        if constexpr (FuseTerminalScale) {
            scale_narrow_tile(kDstBase0, TerminalScalePacked);
            scale_narrow_tile(kDstBase1, TerminalScalePacked);
            scale_narrow_tile(kDstBase2, TerminalScalePacked);
        }

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_output, 3);
        pack_tile(kPackBase0, cb_output, 0);
        pack_tile(kPackBase1, cb_output, 1);
        pack_tile(kPackBase2, cb_output, 2);
        cb_push_back(cb_output, 3);

        tile_regs_release();
    }
}

inline void run_scale_step(
    const uint32_t cb_input,
    const uint32_t cb_output,
    const uint32_t scalar_packed,
    const uint32_t output_group_count) {
    for (uint32_t group = 0; group < output_group_count; ++group) {
        cb_wait_front(cb_input, 3);
        cb_reserve_back(cb_output, 3);

        for (uint32_t tile = 0; tile < 3; ++tile) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_input);
            copy_narrow_tile(cb_input, tile, kDstScale);
            scale_narrow_tile(kDstScale, scalar_packed);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(kDstScale, cb_output, tile);
            tile_regs_release();
        }

        cb_pop_front(cb_input, 3);
        cb_push_back(cb_output, 3);
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
