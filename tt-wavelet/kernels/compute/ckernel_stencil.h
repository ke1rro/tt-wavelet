// ckernel_sfpu_stencil.h
// SFPU kernel for horizontal 1D stencil convolution.
// Adapted from docs/ckernel.h prototype.
//
// Computes g[i] = sum_{j=0}^{k-1} h[j] * f[i-j]
// on even/odd column faces using the SFPU shift-and-accumulate approach.

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// ============================================================================
// ROTATE: 1-element right shift across two consecutive face registers.
//
// After rotation, column 0 of b gets the value that was at column 0 of a
// (the element that shifted off the right end of a wraps into b's column 0).
//
// a_reg, b_reg: LReg indices (p_sfpu::LREG0..7)
// ============================================================================
#define STENCIL_ROTATE(a_reg, b_reg)                                       \
    do {                                                                   \
        /* Rotate a right by 1 within each subvector */                    \
        TTI_SFPSHFT2(0, a_reg, a_reg, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);      \
        /* Rotate b right by 1 within each subvector */                    \
        TTI_SFPSHFT2(0, b_reg, b_reg, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);      \
        /* Set LaneEnable = true for lanes where mask != 0 (col 0 only) */ \
        TTI_SFPSETCC(0, p_sfpu::LREG14, 0, SFPSETCC_MOD1_LREG_NE0);        \
        /* Copy a -> b for enabled lanes (col 0 only) */                   \
        TTI_SFPMOV(0, a_reg, b_reg, 0);                                    \
        /* Restore all lanes enabled */                                    \
        TTI_SFPENCC(0, 0, 0, 0);                                           \
    } while (0)

// ============================================================================
// BROADCAST: load a 32-bit float constant (bit-cast as uint32_t) into an LReg.
// Replicates the scalar value across all lanes.
// ============================================================================
#define STENCIL_BROADCAST(val_u32, lreg)                             \
    do {                                                             \
        TTI_SFPLOADI(lreg, SFPLOADI_MOD0_LOWER, (val_u32) & 0xFFFF); \
        TTI_SFPLOADI(lreg, SFPLOADI_MOD0_UPPER, (val_u32) >> 16);    \
    } while (0)

// ============================================================================
// calculate_stencil_init: pre-load the column-0 mask into vConstFloatPrgm2
// (mapped to LREG14). Must be called once before any stencil body calls.
// ============================================================================
template <uint8_t K>
inline void calculate_stencil_init() {
    vFloat mask = 0.0f;
    vInt tid = vConstTileId;
    vInt check = tid & 0xE;  // bits [3:1] of tile ID → col index within subvec
    v_if(check == 0) { mask = 1.0f; }
    v_endif;
    vConstFloatPrgm2 = mask;
}

// ============================================================================
// calculate_stencil_body: one 4-row block of the stencil computation.
//
// Parameters (all are DST tile indices):
//   dst_f_e_0: halo even columns
//   dst_f_o_0: halo odd columns
//   dst_f_e_1: current even columns
//   dst_f_o_1: current odd columns
//   dst_g_e:   output even columns (zeroed by this function)
//   dst_g_o:   output odd columns  (zeroed by this function)
//
// h_packed: K uint32_t values, each being bit-cast of h[j] as float32
// ============================================================================
template <uint8_t K>
inline void calculate_stencil_body(
    const uint32_t h_packed[K],
    const uint dst_f_e_0,
    const uint dst_f_o_0,
    const uint dst_f_e_1,
    const uint dst_f_o_1,
    const uint dst_g_e,
    const uint dst_g_o) {
    constexpr uint dst_tile_size = 64;  // rows per tile in Dst

    // Load input faces into LRegs
    TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, dst_f_e_0 * dst_tile_size);  // f_e_0
    TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, dst_f_o_0 * dst_tile_size);  // f_o_0
    TT_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, dst_f_e_1 * dst_tile_size);  // f_e_1
    TT_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, dst_f_o_1 * dst_tile_size);  // f_o_1

    // Zero the output accumulators
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);  // g_e = 0
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);  // g_o = 0

    // Main stencil loop — unrolled
#pragma GCC unroll 17
    for (uint8_t j = 0; j < K; j++) {
        // Broadcast h[j] into LREG6
        STENCIL_BROADCAST(h_packed[j], p_sfpu::LREG6);

        if ((j & 1) == 0) {
            // Even coefficient index: accumulate h[2m] terms
            // g_e += h[j] * f_e_1
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LREG4, 0);
            // g_o += h[j] * f_o_1
            TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LREG5, 0);
        } else {
            // Odd coefficient index: accumulate h[2m+1] terms with cross-terms
            // g_o += h[j] * f_e_1
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LREG5, 0);
            // Rotate odd: ROTATE(f_o_0, f_o_1)
            STENCIL_ROTATE(p_sfpu::LREG1, p_sfpu::LREG3);
            // g_e += h[j] * f_o_1 (now shifted)
            TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LREG4, 0);
            // Rotate even: ROTATE(f_e_0, f_e_1)
            STENCIL_ROTATE(p_sfpu::LREG0, p_sfpu::LREG2);
        }
    }

    // Store results back to Dst
    TT_SFPSTORE(p_sfpu::LREG4, 0, ADDR_MOD_7, dst_g_e * dst_tile_size);  // g_e
    TT_SFPSTORE(p_sfpu::LREG5, 0, ADDR_MOD_7, dst_g_o * dst_tile_size);  // g_o

    dst_reg++;
}

// ============================================================================
// calculate_stencil: full stencil over ITERATIONS row-groups.
// ITERATIONS=8 processes all 32 rows of a tile (8 × 4 rows per group).
// For 1D signals use ITERATIONS=1 (only row 0 matters).
// ============================================================================
template <uint8_t K, int ITERATIONS = 8>
inline void calculate_stencil(
    const uint32_t h_packed[K],
    const uint dst_f_e_0,
    const uint dst_f_o_0,
    const uint dst_f_e_1,
    const uint dst_f_o_1,
    const uint dst_g_e,
    const uint dst_g_o) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_stencil_body<K>(h_packed, dst_f_e_0, dst_f_o_0, dst_f_e_1, dst_f_o_1, dst_g_e, dst_g_o);
    }
}

}  // namespace sfpu
}  // namespace ckernel
