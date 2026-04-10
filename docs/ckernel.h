#pragma once

#include <algorithm>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

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
#define STENCIL_ROTATE(a_reg, b_reg)                                                    \
    do {                                                                                \
        /* Shift a to the right */                                                      \
        TTI_SFPSHFT2(0, a_reg, a_reg, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);             \
        /* Shift b to the right */                                                      \
        TTI_SFPSHFT2(0, b_reg, b_reg, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);             \
        /* Set LaneEnable=true for lanes with col=0 and LaneEnable=false for others*/   \
        TTI_SFPSETCC(0, p_sfpu::LREG14, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);               \
        /* Copy first column of a to the first column of b */                           \
        TTI_SFPMOV(0, a_reg, b_reg, 0);                                                 \
        /* Set all LaneEnable=true */                                                   \
        TTI_SFPSETCC(1, 0, 0, sfpi::SFPSETCC_MOD1_IMM_BIT0);                            \
    } while (0)

// BROADCAST: load a 32-bit float constant (bit-cast as uint32_t) into an LReg.
#define STENCIL_BROADCAST(val_u32, lreg)                                                \
    do {                                                                                \
        TTI_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_UPPER, (val_u32) >> 16);                 \
        TTI_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_LOWER, (val_u32) & 0xFFFF);              \
    } while (0)

// Arguments:
//   h_packed: array of K uint32_t values, each being the bit-cast of h[j] as float32
//   f_e_0, f_o_0: input1 offset (even/odd columns of the previous 16 columns)
//   f_e_1, f_o_1: input2 offset (even/odd columns of the current 16 columns)
//   g_e,   g_o:   output (even/odd columns), should be pre-zeroed
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
    // Register allocations
    const auto& f_e_0 = p_sfpu::LREG0;
    const auto& f_o_0 = p_sfpu::LREG1;
    const auto& f_e_1 = p_sfpu::LREG2;
    const auto& f_o_1 = p_sfpu::LREG3;
    const auto& g_e = p_sfpu::LREG4;
    const auto& g_o = p_sfpu::LREG5;
    const auto& tmp = p_sfpu::LREG6;

    // Load inputs into LRegs
    TT_SFPLOAD(f_e_0, 0, ADDR_MOD_7, dst_f_e_0);  // f_e_0
    TT_SFPLOAD(f_o_0, 0, ADDR_MOD_7, dst_f_o_0);  // f_o_0
    TT_SFPLOAD(f_e_1, 0, ADDR_MOD_7, dst_f_e_1);  // f_e_1
    TT_SFPLOAD(f_o_1, 0, ADDR_MOD_7, dst_f_o_1);  // f_o_1

    // Zero the output accumulators
    TTI_SFPMOV(0, p_sfpu::LCONST_0, g_e, 0);  // g_e = 0
    TTI_SFPMOV(0, p_sfpu::LCONST_0, g_o, 0);  // g_o = 0

#pragma unroll 17
    for (uint8_t j = 0; j < K; j++) {
        STENCIL_BROADCAST(h_packed[j], tmp);

        if ((j & 1) == 0) {
            // g_e += h[j] * f_e_1
            TTI_SFPMAD(f_e_1, tmp, g_e, g_e, 0);
            // g_o += h[j] * f_o_1
            TTI_SFPMAD(f_o_1, tmp, g_o, g_o, 0);
        } else {
            // g_o += h[j] * f_e_1
            TTI_SFPMAD(f_e_1, tmp, g_o, g_o, 0);
            // Rotate odd columns: ROTATE(f_o_0, f_o_1)
            STENCIL_ROTATE(f_o_0, f_o_1);
            // g_e += h[j] * f_o_1 (now shifted)
            TTI_SFPMAD(f_o_1, tmp, g_e, g_e, 0);
            // Rotate even columns: ROTATE(f_e_0, f_e_1)
            STENCIL_ROTATE(f_e_0, f_e_1);
        }
    }

    // Store results back to Dst
    TT_SFPSTORE(g_e, 0, ADDR_MOD_7, dst_g_e);  // g_e
    TT_SFPSTORE(g_o, 0, ADDR_MOD_7, dst_g_o);  // g_o
}

template <uint8_t K>
inline void calculate_stencil_faces(
    const uint32_t h_packed[K],
    const uint input1,
    const uint input2,
    const uint output,
    const uint rows = 16
) {
    // Vertical rows
#pragma unroll 4
    for (uint offset = 0; offset < std::min(rows, 16) * 2; offset += 8) {
        calculate_stencil_body<K>(
            h_packed,
            input1 + offset, input1 + offset + 1,
            input2 + offset, input2 + offset + 1,
            output + offset, output + offset + 1
        );
    }
}


// Arguments:
//   h_packed: array of K uint32_t values, each being the bit-cast of h[j] as float32
//   input1: first tile index in dst register
//   input2: second tile index in dst register (=8 to work only on first tile)
//   output: tile index in dst register for output (can be same as either input)
//   rows: number of rows to process (in multiples of 4, max 8)
template <uint8_t K>
inline void calculate_stencil(
    const uint32_t h_packed[K],
    const uint input1,
    const uint input2,
    const uint output,
    const uint rows = 32
) {
    constexpr uint FACE_SIZE = 256;

    // First face-row
    // col0/1
    calculate_stencil_faces<K>(
        h_packed,
        input1,
        input1 + FACE_SIZE,
        output,
        rows,
    );
    // col1/2
    calculate_stencil_faces<K>(
        h_packed,
        input1 + FACE_SIZE,
        input2,
        output + FACE_SIZE,
        rows,
    );

    if (rows < 16) {
        return;
    }

    // Second face-row
    // col0/1
    calculate_stencil_faces<K>(
        h_packed,
        input1 + FACE_SIZE * 2,
        input1 + FACE_SIZE * 3,
        output + FACE_SIZE * 2,
        rows - 16,
    );
    // col1/2
    calculate_stencil_faces<K>(
        h_packed,
        input1 + FACE_SIZE * 3,
        input2 + FACE_SIZE * 2,
        output + FACE_SIZE * 3,
        rows - 16,
    );
}

}  // namespace sfpu
}  // namespace ckernel
