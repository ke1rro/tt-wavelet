#include <array>
#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "llk_math_eltwise_ternary_sfpu_params.h"

#ifdef TRISC_MATH
inline void custom_stencil_mac_face(
    uint32_t dst_index_base,
    uint32_t dst_index_in0,
    uint32_t dst_index_in1,
    uint32_t dst_index_out,
    float c0,
    float c1) {
    constexpr uint32_t n_vector_in_tile = 32;

    const uint32_t base_idx = dst_index_base * n_vector_in_tile;
    const uint32_t in0_idx = dst_index_in0 * n_vector_in_tile;
    const uint32_t in1_idx = dst_index_in1 * n_vector_in_tile;
    const uint32_t out_idx = dst_index_out * n_vector_in_tile;

    for (size_t i = 0; i < 8; ++i) {
        sfpi::vFloat base = sfpi::dst_reg[base_idx + i];
        sfpi::vFloat in0 = sfpi::dst_reg[in0_idx + i];
        sfpi::vFloat in1 = sfpi::dst_reg[in1_idx + i];
        sfpi::dst_reg[out_idx + i] = base + sfpi::vFloat(c0) * in0 + sfpi::vFloat(c1) * in1;
    }
}
#endif

inline void custom_stencil_mac_tile(
    uint32_t dst_index_base,
    uint32_t dst_index_in0,
    uint32_t dst_index_in1,
    uint32_t dst_index_out,
    float c0,
    float c1) {
    MATH(_llk_math_eltwise_ternary_sfpu_params_<false>(
        custom_stencil_mac_face,
        dst_index_base,
        dst_index_in0,
        dst_index_in1,
        dst_index_out,
        static_cast<int>(VectorMode::RC),
        c0,
        c1));
}

void kernel_main() {
    constexpr uint32_t n_tile_pairs = 1;

    constexpr auto cb_base = tt::CBIndex::c_0;
    constexpr auto cb_in0 = tt::CBIndex::c_1;
    constexpr auto cb_in1 = tt::CBIndex::c_2;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    constexpr uint32_t dst_base = 0;
    constexpr uint32_t dst_in0 = 1;
    constexpr uint32_t dst_in1 = 2;
    constexpr uint32_t dst_out = 3;

    constexpr float c0 = 0.5f;
    constexpr float c1 = -0.25f;

    init_sfpu(cb_base, cb_out0);
    copy_tile_init(cb_base);

    for (uint32_t i = 0; i < n_tile_pairs; ++i) {
        cb_wait_front(cb_base, 1);
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        tile_regs_acquire();

        copy_tile(cb_base, 0, dst_base);

        copy_tile_to_dst_init_short_with_dt(cb_base, cb_in0);
        copy_tile(cb_in0, 0, dst_in0);

        copy_tile_to_dst_init_short_with_dt(cb_in0, cb_in1);
        copy_tile(cb_in1, 0, dst_in1);

        custom_stencil_mac_tile(dst_base, dst_in0, dst_in1, dst_out, c0, c1);

        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_out0, 1);
        pack_tile(dst_out, cb_out0);
        tile_regs_release();

        cb_push_back(cb_out0, 1);
        cb_pop_front(cb_base, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
