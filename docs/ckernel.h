// Created using DeepWiki for fast prototyping
// ckernel_sfpu_stencil.h  
  
#pragma once  
  
#include "ckernel.h"  
#include "ckernel_defs.h"  
#include "sfpi.h"  
  
using namespace sfpi;  
  
namespace ckernel {  
namespace sfpu {  
  
// ============================================================================  
// Horizontal Stencil SFPU Kernel  
//  
// Computes g'[i:i+16] = g[i+16:i+32] = T(f[i:i+16], f[i+16:i+32], h)  
// where g[i] = sum_{j=0}^{k-1} h[j] * f[i-j]  
//  
// The SFPU processes even and odd columns separately. Each dst_reg "row"  
// (32 elements) contains 4 subvectors of 8 columns each.  
//  
// Tile layout (WH/BH):  
//   Face 0 (dst_reg[0..3]):  even columns of rows 0-15  
//   Face 1 (dst_reg[4..7]):  odd  columns of rows 0-15  
//   Face 2 (dst_reg[8..11]): even columns of rows 16-31  
//   Face 3 (dst_reg[12..15]): odd  columns of rows 16-31  
//  
// Template parameter K: stencil filter length (1 < K < 18)  
// h_coeffs: array of K float coefficients (bit-cast to uint32_t)  
// ============================================================================  

template <uint8_t K>  
inline void calculate_stencil_init() {  
    vFloat mask = 0.0f;  
    vInt tid = reinterpret<vInt>(vConstTileId);  
    vInt check = tid & 0xE;  // bits [3:1] of tile ID  
    v_if (check == 0) {  
        mask = 1.0f;  
    }
    v_endif;  
    vConstFloatPrgm2 = mask;  
}  
  
// ROTATE(a, b): 1-element right shift within subvectors.  
// After rotation, column 0 of b gets column 0 of a (the element that  
// "fell off" the right end of a wraps around, and we overwrite b's  
// column 0 with a's column 0 via masking).  
//  
// Uses TTI intrinsics because we need precise control over:  
//   1. subvec_shflror1 on both a and b  
//   2. Conditional copy of a[col0] -> b[col0] using the preloaded mask  
//  
// a_reg, b_reg: LReg indices (p_sfpu::LREG0..7)  
#define STENCIL_ROTATE(a_reg, b_reg)                                                    \  
    do {                                                                                 \  
        /* Rotate a right by 1 within each subvector */                                  \  
        TTI_SFPSHFT2(0, a_reg, a_reg, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);            \  
        /* Rotate b right by 1 within each subvector */                                  \  
        TTI_SFPSHFT2(0, b_reg, b_reg, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);            \  
        /* Load mask from vConstFloatPrgm2 into a temp LReg for SETCC */                 \  
        /* Set LaneEnable = true for lanes where mask != 0 (i.e., col 0) */              \  
        TTI_SFPSETCC(0, p_sfpu::LREG14, 0, SFPSETCC_MOD1_LREG_NE0);                    \  
        /* Copy a -> b for enabled lanes (col 0 only) */                                 \  
        TTI_SFPMOV(0, a_reg, b_reg, 0);                                                 \  
        /* Restore all lanes enabled */                                                  \  
        TTI_SFPENCC(0, 0, 0, 0);                                                        \  
    } while (0)  
  
// BROADCAST: load a 32-bit float constant (bit-cast as uint32_t) into an LReg.  
// This replicates the value across all 32 lanes.  
#define STENCIL_BROADCAST(val_u32, lreg)                                                 \  
    do {                                                                                 \  
        TTI_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_LOWER, (val_u32) & 0xFFFF);              \  
        TTI_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_UPPER, (val_u32) >> 16);                 \  
    } while (0)  
  
// Main stencil computation.  
//  
// Parameters (all are dst_reg indices, i.e., offsets into the Dst register):  
//   f_e_0, f_o_0: halo face (even/odd columns of the previous 16 columns)  
//   f_e_1, f_o_1: current face (even/odd columns of the current 16 columns)  
//   g_e,   g_o:   output (even/odd columns), should be pre-zeroed  
//  
// h_packed: array of K uint32_t values, each being the bit-cast of h[j] as float32  
//  
// This function processes one set of 4 rows (one dst_reg "row") at a time.  
// The caller must iterate over all rows and advance dst_reg accordingly.  
//  
// Register allocation:  
//   LREG0: f_e_0 (halo even)  
//   LREG1: f_o_0 (halo odd)  
//   LREG2: f_e_1 (current even)  
//   LREG3: f_o_1 (current odd)  
//   LREG4: g_e (accumulator for even output)  
//   LREG5: g_o (accumulator for odd output)  
//   LREG6: tmp (broadcast coefficient)  
//   LREG7: (scratch, used by ROTATE via SFPMOV)  
//   LREG14: mask (preloaded in vConstFloatPrgm2 -> LREG14 is the const reg)  
  
template <uint8_t K>  
inline void calculate_stencil_body(  
    const uint32_t h_packed[K],  
    const uint dst_f_e_0,   // dst index for halo even columns  
    const uint dst_f_o_0,   // dst index for halo odd columns  
    const uint dst_f_e_1,   // dst index for current even columns  
    const uint dst_f_o_1,   // dst index for current odd columns  
    const uint dst_g_e,     // dst index for output even columns  
    const uint dst_g_o      // dst index for output odd columns  
) {  
    constexpr uint dst_tile_size = 64;  // rows per tile in Dst  
  
    // Load input faces into LRegs  
    TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, dst_f_e_0 * dst_tile_size);  // f_e_0  
    TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, dst_f_o_0 * dst_tile_size);  // f_o_0  
    TT_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, dst_f_e_1 * dst_tile_size);  // f_e_1  
    TT_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, dst_f_o_1 * dst_tile_size);  // f_o_1  
  
    // Zero the output accumulators  
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);  // g_e = 0  
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);  // g_o = 0  
  
    // Main stencil loop  
#pragma unroll 17
    for (uint8_t j = 0; j < K; j++) {  
        // Broadcast h[j] into LREG6  
        STENCIL_BROADCAST(h_packed[j], p_sfpu::LREG6);  
  
        if ((j & 1) == 0) {  
            // Even index: accumulate h[2j] terms  
            // g_e += h[j] * f_e_1  
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LREG4, 0);  
            // g_o += h[j] * f_o_1  
            TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LREG5, 0);  
        } else {  
            // Odd index: accumulate h[2j+1] terms with cross-terms  
            // g_o += h[j] * f_e_1  
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LREG5, 0);  
            // Rotate odd columns: ROTATE(f_o_0, f_o_1)  
            STENCIL_ROTATE(p_sfpu::LREG1, p_sfpu::LREG3);  
            // g_e += h[j] * f_o_1 (now shifted)  
            TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LREG4, 0);  
            // Rotate even columns: ROTATE(f_e_0, f_e_1)  
            STENCIL_ROTATE(p_sfpu::LREG0, p_sfpu::LREG2);  
        }  
    }  
  
    // Store results back to Dst  
    TT_SFPSTORE(p_sfpu::LREG4, 0, ADDR_MOD_7, dst_g_e * dst_tile_size);  // g_e  
    TT_SFPSTORE(p_sfpu::LREG5, 0, ADDR_MOD_7, dst_g_o * dst_tile_size);  // g_o  
  
    dst_reg++;  
}

// Full stencil over ITERATIONS rows (standard SFPU iteration pattern).  
// ITERATIONS=8 processes all 8 "rows" of a face (8 × 4 subvec rows = 32 rows).  
template <uint8_t K, int ITERATIONS = 8>  
inline void calculate_stencil(  
    const uint32_t h_packed[K],  
    const uint dst_f_e_0,  
    const uint dst_f_o_0,  
    const uint dst_f_e_1,  
    const uint dst_f_o_1,  
    const uint dst_g_e,  
    const uint dst_g_o  
) {  
    for (int d = 0; d < ITERATIONS; d++) {  
        calculate_stencil_body<K>(  
            h_packed,  
            dst_f_e_0, dst_f_o_0,  
            dst_f_e_1, dst_f_o_1,  
            dst_g_e, dst_g_o);  
    }  
}  
  
}  // namespace sfpu  
}  // namespace ckernel