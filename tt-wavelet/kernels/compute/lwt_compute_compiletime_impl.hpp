#pragma once

#include <array>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <utility>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "lwt_compute_utils.hpp"

namespace ttwv::kernels::lwt {

constexpr uint32_t kArgsPerStep = 2;
constexpr uint32_t kDstInput0 = 0;
constexpr uint32_t kDstInput1 = 1;
constexpr uint32_t kDstOutput = 2;
constexpr uint32_t kDstBase = 3;

[[nodiscard]] inline uint32_t pack_float(const float v) {
    union {
        float f;
        uint32_t u;
    } raw{v};
    return raw.u;
}

template <uint32_t K>
inline void run_step_with_coeffs(
    const uint32_t cb_input0,
    const uint32_t cb_input1,
    const uint32_t cb_base,
    const uint32_t cb_output,
    const std::array<uint32_t, K>& h_coeffs,
    const uint32_t output_stick_count) {
    for (uint32_t stick = 0; stick < output_stick_count; ++stick) {
        tile_regs_acquire();

        cb_wait_front(cb_input0, 1);
        copy_tile_to_dst_init_short(cb_input0);
        copy_tile(cb_input0, 0, kDstInput0);
        cb_pop_front(cb_input0, 1);

        cb_wait_front(cb_input1, 1);
        copy_tile_to_dst_init_short(cb_input1);
        copy_tile(cb_input1, 0, kDstInput1);
        cb_pop_front(cb_input1, 1);

        hstencil_init();
        hstencil_row<K>(h_coeffs, kDstInput0, kDstInput1, kDstOutput);

        add_binary_tile_init();
        cb_wait_front(cb_base, 1);
        copy_tile_to_dst_init_short(cb_base);
        copy_tile(cb_base, 0, kDstBase);
        add_binary_tile(kDstOutput, kDstBase, kDstOutput);
        cb_pop_front(cb_base, 1);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_output, 1);
        pack_tile(kDstOutput, cb_output);
        cb_push_back(cb_output, 1);

        tile_regs_release();
    }
}

template <typename StepT>
inline void run_step_for_static(
    const uint32_t cb_input0,
    const uint32_t cb_input1,
    const uint32_t cb_base,
    const uint32_t cb_output,
    const StepT& step,
    const uint32_t output_stick_count) {
    constexpr uint32_t K = static_cast<uint32_t>(StepT::n_coeffs);
    static_assert(K > 0, "predict/update steps must have coefficients");

    std::array<uint32_t, K> h_coeffs{};
#pragma unroll 17
    for (uint32_t j = 0; j < K; ++j) {
        h_coeffs[j] = pack_float(step.coefficients[j]);
    }

    run_step_with_coeffs<K>(cb_input0, cb_input1, cb_base, cb_output, h_coeffs, output_stick_count);
}

template <int WaveletId, uint32_t StepIdx, uint32_t NumSteps>
struct WaveletStepRunner {
    static inline bool run(
        const uint32_t target_step_index,
        const uint32_t cb_input0,
        const uint32_t cb_input1,
        const uint32_t cb_base,
        const uint32_t cb_output,
        const uint32_t output_stick_count) {
        if (target_step_index == StepIdx) {
            // Compile-time step access: coefficients become embedded immediates.
            constexpr const auto& step =
                ttwv::get_scheme_by_id<WaveletId>().template step<StepIdx>();
            using StepT = typename std::decay<decltype(step)>::type;
            // Scale/swap steps are never dispatched to compute by the host.
            if constexpr (StepT::type == ttwv::StepType::kPredict ||
                          StepT::type == ttwv::StepType::kUpdate) {
                run_step_for_static(cb_input0, cb_input1, cb_base, cb_output, step, output_stick_count);
                return true;
            }
            return false;
        }
        if constexpr (StepIdx + 1 < NumSteps) {
            return WaveletStepRunner<WaveletId, StepIdx + 1, NumSteps>::run(
                target_step_index, cb_input0, cb_input1, cb_base, cb_output, output_stick_count);
        }
        return false;
    }
};

template <int WaveletId>
inline bool run_wavelet_step(
    const uint32_t full_step_index,
    const uint32_t cb_input0,
    const uint32_t cb_input1,
    const uint32_t cb_base,
    const uint32_t cb_output,
    const uint32_t output_stick_count) {
    using SchemeT = typename std::decay<decltype(ttwv::get_scheme_by_id<WaveletId>())>::type;
    constexpr uint32_t num_steps = static_cast<uint32_t>(SchemeT::num_steps);
    return WaveletStepRunner<WaveletId, 0, num_steps>::run(
        full_step_index, cb_input0, cb_input1, cb_base, cb_output, output_stick_count);
}

template <int CandidateWaveletId, int MaxWaveletId>
struct SingleWaveletDispatch {
    static inline bool run(
        const uint32_t wavelet_id,
        const uint32_t full_step_index,
        const uint32_t cb_input0,
        const uint32_t cb_input1,
        const uint32_t cb_base,
        const uint32_t cb_output,
        const uint32_t output_stick_count) {
        if constexpr (CandidateWaveletId < MaxWaveletId) {
            if (wavelet_id == static_cast<uint32_t>(CandidateWaveletId)) {
                return run_wavelet_step<CandidateWaveletId>(
                    full_step_index, cb_input0, cb_input1, cb_base, cb_output, output_stick_count);
            }
        }
        return false;
    }
};

template <int BeginWaveletId, int MaxWaveletId, size_t... Offsets>
inline bool dispatch_wavelet_step_range_impl(
    const uint32_t wavelet_id,
    const uint32_t full_step_index,
    const uint32_t cb_input0,
    const uint32_t cb_input1,
    const uint32_t cb_base,
    const uint32_t cb_output,
    const uint32_t output_stick_count,
    std::index_sequence<Offsets...>) {

    return (false || ... || SingleWaveletDispatch<
        BeginWaveletId + static_cast<int>(Offsets), MaxWaveletId
    >::run(wavelet_id, full_step_index, cb_input0, cb_input1, cb_base, cb_output, output_stick_count));
}

template <int ChunkIndex, int WaveletChunkSize, int MaxWaveletId>
inline void kernel_main_chunk() {
    static_assert(ChunkIndex >= 0, "Chunk index must be non-negative");
    static_assert(WaveletChunkSize > 0, "Wavelet chunk size must be positive");

    constexpr uint32_t cb_input0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output = get_compile_time_arg_val(3);

    const uint32_t wavelet_id = get_arg_val<uint32_t>(0);
    const uint32_t num_steps = get_arg_val<uint32_t>(1);

    ckernel::init_sfpu(cb_base, cb_output);

    constexpr int begin_wavelet_id = ChunkIndex * WaveletChunkSize;

    for (uint32_t step = 0; step < num_steps; ++step) {
        const uint32_t step_arg_base = 2 + step * kArgsPerStep;
        const uint32_t full_step_index = get_arg_val<uint32_t>(step_arg_base + 0);
        const uint32_t output_stick_count = get_arg_val<uint32_t>(step_arg_base + 1);

        const bool executed = dispatch_wavelet_step_range_impl<begin_wavelet_id, MaxWaveletId>(
            wavelet_id,
            full_step_index,
            cb_input0,
            cb_input1,
            cb_base,
            cb_output,
            output_stick_count,
            std::make_index_sequence<WaveletChunkSize>{});
        (void)executed;
    }
}

}  // namespace ttwv::kernels::lwt
