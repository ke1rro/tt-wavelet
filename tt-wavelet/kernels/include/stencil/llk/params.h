#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "../policy.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_eltwise_ternary_sfpu.h"

namespace ckernel::stencil {

inline constexpr char kErrNullInputIndices[] = "dst_input_indices must not be null";
inline constexpr char kErrNullCoefficients[] = "coefficients must not be null";
inline constexpr char kErrFilterLenZero[] = "filter_len must be > 0";
inline constexpr char kErrUnsupportedVectorMode[] = "unsupported vector_mode";
inline constexpr char kErrInputIndexOutOfRange[] = "dst_input_indices contains out-of-range tile index";
inline constexpr char kErrBaseIndexOutOfRange[] = "dst_index_base exceeds max dest tiles";
inline constexpr char kErrOutputIndexOutOfRange[] = "dst_index_out exceeds max dest tiles";

/** @brief Starts SFPU execution for stencil-style face traversal. */
template <DstSync DstSyncMode>
inline void stencil_sfpu_start(const uint32_t param) {
    _llk_math_eltwise_ternary_sfpu_start_<DstSyncMode>(param);
}

/** @brief Finalizes SFPU execution for stencil-style face traversal. */
inline void stencil_sfpu_done() { _llk_math_eltwise_ternary_sfpu_done_(); }

/** @brief Advances destination register cursor by 8 rows per block. */
inline void advance_face_by_blocks(const int blocks) {
#pragma GCC unroll 0
    for (size_t i = 0; i < blocks; ++i) {
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }
}

/** @brief Returns true when vector_mode is one of R, C, or RC. */
inline bool is_supported_vector_mode(const int vector_mode) {
    return vector_mode == static_cast<int>(ckernel::VectorMode::R) ||
           vector_mode == static_cast<int>(ckernel::VectorMode::C) ||
           vector_mode == static_cast<int>(ckernel::VectorMode::RC);
}

/** @brief Executes a face callback across the selected vector mode layout. */
template <typename FaceCallable>
inline void run_faces(FaceCallable&& run_face, const int vector_mode) {
    const ckernel::VectorMode mode = static_cast<ckernel::VectorMode>(vector_mode);

    if (mode == ckernel::VectorMode::R) {
#pragma GCC unroll 0
        for (size_t face = 0; face < 2; ++face) {
            run_face();
            advance_face_by_blocks(2);
        }
        advance_face_by_blocks(4);
        return;
    }

    if (mode == ckernel::VectorMode::C) {
#pragma GCC unroll 0
        for (size_t face = 0; face < 2; ++face) {
            run_face();
            advance_face_by_blocks(4);
        }
        return;
    }

#pragma GCC unroll 0
    for (size_t face = 0; face < 4; ++face) {
        run_face();
        advance_face_by_blocks(2);
    }
}

inline uint32_t get_dst_index_limit() {
    constexpr uint32_t dest_num_tiles_full = (DEST_REGISTER_FULL_SIZE * DEST_FACE_WIDTH) / (TILE_HEIGHT * TILE_HEIGHT);
    constexpr uint32_t sync_mode_tiles =
        (DST_SYNC_MODE == DstSync::SyncHalf) ? (dest_num_tiles_full >> 1U) : dest_num_tiles_full;
    constexpr uint32_t dst_max = DST_ACCUM_MODE ? (sync_mode_tiles >> 1U) : sync_mode_tiles;
    return dst_max;
}

inline void validate_vector_mode(const int vector_mode) {
    LLK_ASSERT(is_supported_vector_mode(vector_mode), kErrUnsupportedVectorMode);
}

inline void validate_filter_inputs(const uint32_t* dst_input_indices, const uint32_t filter_len) {
    LLK_ASSERT((dst_input_indices != nullptr), kErrNullInputIndices);
    LLK_ASSERT((filter_len > 0), kErrFilterLenZero);
}

inline void validate_index_range(const uint32_t index, const uint32_t dst_max, const char* err_msg) {
    LLK_ASSERT((index < dst_max), err_msg);
}

inline void validate_input_index_range(
    const uint32_t* dst_input_indices, const uint32_t filter_len, const uint32_t dst_max) {
#pragma GCC unroll 0
    for (size_t i = 0; i < filter_len; ++i) {
        LLK_ASSERT((dst_input_indices[i] < dst_max), kErrInputIndexOutOfRange);
    }
}

/**
 * @brief Shared stencil SFPU wrapper for generic stencil:
 * out = f(input_0 ... input_{L-1}).
 */
template <bool APPROXIMATE, typename PolicyT, typename Callable, typename... Args>
inline void _llk_math_eltwise_stencil_sfpu_params_(
    Callable&& sfpu_func,
    const uint32_t* dst_input_indices,
    const uint32_t filter_len,
    const uint32_t dst_index_out,
    const int vector_mode,
    Args&&... args) {
    [[maybe_unused]] constexpr bool k_approximate = APPROXIMATE;

    validate_filter_inputs(dst_input_indices, filter_len);
    validate_vector_mode(vector_mode);
    assert_taps_within_policy<PolicyT>(filter_len);

    const uint32_t dst_max = get_dst_index_limit();
    validate_input_index_range(dst_input_indices, filter_len, dst_max);
    validate_index_range(dst_index_out, dst_max, kErrOutputIndexOutOfRange);

    stencil_sfpu_start<DST_SYNC_MODE>(0);
    auto run_face = [&]() { sfpu_func(dst_input_indices, filter_len, dst_index_out, std::forward<Args>(args)...); };
    run_faces(run_face, vector_mode);
    stencil_sfpu_done();
}

/**
 * @brief Shared stencil SFPU wrapper for stencil_acc and stencil_mac/affine:
 * out = base + f(input_0 ... input_{L-1}).
 */
template <bool APPROXIMATE, typename PolicyT, typename Callable, typename... Args>
inline void _llk_math_eltwise_stencil_sfpu_params_(
    Callable&& sfpu_func,
    const uint32_t dst_index_base,
    const uint32_t* dst_input_indices,
    const uint32_t filter_len,
    const uint32_t dst_index_out,
    const int vector_mode,
    Args&&... args) {
    [[maybe_unused]] constexpr bool k_approximate = APPROXIMATE;

    validate_filter_inputs(dst_input_indices, filter_len);
    validate_vector_mode(vector_mode);
    assert_taps_within_policy<PolicyT>(filter_len);

    const uint32_t dst_max = get_dst_index_limit();
    validate_index_range(dst_index_base, dst_max, kErrBaseIndexOutOfRange);
    validate_input_index_range(dst_input_indices, filter_len, dst_max);
    validate_index_range(dst_index_out, dst_max, kErrOutputIndexOutOfRange);

    stencil_sfpu_start<DST_SYNC_MODE>(0);
    auto run_face = [&]() {
        sfpu_func(dst_index_base, dst_input_indices, filter_len, dst_index_out, std::forward<Args>(args)...);
    };
    run_faces(run_face, vector_mode);
    stencil_sfpu_done();
}

template <bool APPROXIMATE, size_t FILTER_LEN, typename PolicyT, typename Callable, typename... Args>
inline void _llk_math_eltwise_stencil_sfpu_params_(
    Callable&& sfpu_func,
    const std::array<uint32_t, FILTER_LEN>& dst_input_indices,
    const uint32_t dst_index_out,
    const int vector_mode,
    Args&&... args) {
    static_assert(FILTER_LEN > 0, "FILTER_LEN must be > 0");
    static_assert(FILTER_LEN <= PolicyT::max_taps_per_pass, "FILTER_LEN exceeds per-pass stencil policy capacity");

    _llk_math_eltwise_stencil_sfpu_params_<APPROXIMATE, PolicyT>(
        std::forward<Callable>(sfpu_func),
        dst_input_indices.data(),
        static_cast<uint32_t>(FILTER_LEN),
        dst_index_out,
        vector_mode,
        std::forward<Args>(args)...);
}

template <bool APPROXIMATE, size_t FILTER_LEN, typename PolicyT, typename Callable, typename... Args>
inline void _llk_math_eltwise_stencil_sfpu_params_(
    Callable&& sfpu_func,
    const uint32_t dst_index_base,
    const std::array<uint32_t, FILTER_LEN>& dst_input_indices,
    const uint32_t dst_index_out,
    const int vector_mode,
    Args&&... args) {
    static_assert(FILTER_LEN > 0, "FILTER_LEN must be > 0");
    static_assert(FILTER_LEN <= PolicyT::max_taps_per_pass, "FILTER_LEN exceeds per-pass stencil policy capacity");

    _llk_math_eltwise_stencil_sfpu_params_<APPROXIMATE, PolicyT>(
        std::forward<Callable>(sfpu_func),
        dst_index_base,
        dst_input_indices.data(),
        static_cast<uint32_t>(FILTER_LEN),
        dst_index_out,
        vector_mode,
        std::forward<Args>(args)...);
}

}  // namespace ckernel::stencil
