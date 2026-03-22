#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_eltwise_ternary_sfpu.h"
#include "stencil/policy.hpp"

namespace ckernel::detail {

inline constexpr char kErrNullInputIndices[] = "dst_input_indices must not be null";
inline constexpr char kErrFilterLenZero[] = "filter_len must be > 0";
inline constexpr char kErrUnsupportedVectorMode[] = "unsupported vector_mode";
inline constexpr char kErrInputIndexOutOfRange[] = "dst_input_indices contains out-of-range tile index";
inline constexpr char kErrOutputIndexOutOfRange[] = "dst_index_out exceeds max dest tiles";

template <DstSync DstSyncMode>
void stencil_sfpu_start(const std::uint32_t param) {
    _llk_math_eltwise_ternary_sfpu_start_<DstSyncMode>(param);
}

inline void stencil_sfpu_done() { _llk_math_eltwise_ternary_sfpu_done_(); }

inline void advance_face_by_blocks(const int blocks) {
#pragma GCC unroll 0
    for (size_t i = 0; i < blocks; ++i) {
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }
}

inline bool is_supported_vector_mode(const int vector_mode) {
    return vector_mode == static_cast<int>(ckernel::VectorMode::R) ||
           vector_mode == static_cast<int>(ckernel::VectorMode::C) ||
           vector_mode == static_cast<int>(ckernel::VectorMode::RC);
}

template <typename Callable, typename... Args>
inline void run_faces(
    Callable& sfpu_func,
    const uint32_t* dst_input_indices,
    const uint32_t filter_len,
    const uint32_t dst_index_out,
    const std::size_t face_count,
    const int blocks_per_face,
    Args&... args) {
#pragma GCC unroll 0
    for (std::size_t face = 0; face < face_count; ++face) {
        sfpu_func(dst_input_indices, filter_len, dst_index_out, args...);
        advance_face_by_blocks(blocks_per_face);
    }
}

}  // namespace ckernel::detail

template <bool APPROXIMATE, typename PolicyT, typename Callable, typename... Args>
void _llk_math_eltwise_stencil_sfpu_params_(
    Callable&& sfpu_func,
    const uint32_t* dst_input_indices,
    uint32_t filter_len,
    uint32_t dst_index_out,
    int vector_mode = static_cast<int>(ckernel::VectorMode::RC),
    Args&&... args) {
    [[maybe_unused]] constexpr bool k_approximate = APPROXIMATE;

    LLK_ASSERT((dst_input_indices != nullptr), ckernel::detail::kErrNullInputIndices);
    LLK_ASSERT((filter_len > 0), ckernel::detail::kErrFilterLenZero);
    LLK_ASSERT(ckernel::detail::is_supported_vector_mode(vector_mode), ckernel::detail::kErrUnsupportedVectorMode);
    stencil::assert_taps_within_policy<PolicyT>(filter_len);

    constexpr uint32_t sync_mode_tiles =
        (DST_SYNC_MODE == DstSync::SyncHalf) ? DEST_NUM_TILES_FP16_HALF : DEST_NUM_TILES_FP16;
    constexpr uint32_t dst_max = DST_ACCUM_MODE ? (sync_mode_tiles >> 1U) : sync_mode_tiles;

#pragma GCC unroll 0
    for (size_t i{0}; i < filter_len; ++i) {
        LLK_ASSERT((dst_input_indices[i] < dst_max), ckernel::detail::kErrInputIndexOutOfRange);
    }
    LLK_ASSERT((dst_index_out < dst_max), ckernel::detail::kErrOutputIndexOutOfRange);
    ckernel::detail::stencil_sfpu_start<DST_SYNC_MODE>(0);

    const ckernel::VectorMode mode = static_cast<ckernel::VectorMode>(vector_mode);
    auto invoke_face =
        [&](const uint32_t* input_indices, const uint32_t len, const uint32_t output_index, auto&... call_args) {
            std::forward<Callable>(sfpu_func)(input_indices, len, output_index, call_args...);
        };

    if (mode == ckernel::VectorMode::R) {
        ckernel::detail::run_faces(invoke_face, dst_input_indices, filter_len, dst_index_out, 2, 2, args...);
        ckernel::detail::advance_face_by_blocks(4);
    } else if (mode == ckernel::VectorMode::C) {
        ckernel::detail::run_faces(invoke_face, dst_input_indices, filter_len, dst_index_out, 2, 4, args...);
    } else if (mode == ckernel::VectorMode::RC) {
        ckernel::detail::run_faces(invoke_face, dst_input_indices, filter_len, dst_index_out, 4, 2, args...);
    }

    ckernel::detail::stencil_sfpu_done();
}

template <bool APPROXIMATE, std::size_t FILTER_LEN, typename PolicyT, typename Callable, typename... Args>
inline void _llk_math_eltwise_stencil_sfpu_params_(
    Callable&& sfpu_func,
    const std::array<uint32_t, FILTER_LEN>& dst_input_indices,
    uint32_t dst_index_out,
    int vector_mode = static_cast<int>(ckernel::VectorMode::RC),
    Args&&... args) {
    static_assert(FILTER_LEN > 0, "FILTER_LEN must be > 0");
    static_assert(FILTER_LEN <= PolicyT::max_taps_per_pass, "FILTER_LEN exceeds per-pass stencil policy capacity");

    _llk_math_eltwise_stencil_sfpu_params_<APPROXIMATE, PolicyT>(
        std::forward<Callable>(sfpu_func),
        dst_input_indices.data(),
        static_cast<std::uint32_t>(FILTER_LEN),
        dst_index_out,
        vector_mode,
        std::forward<Args>(args)...);
}
