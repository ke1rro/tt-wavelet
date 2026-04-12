// Register allocation:
//   LREG0: f_e_0 (input1 even)
//   LREG1: f_o_0 (input1 odd)
//   LREG2: f_e_1 (input2 even)
//   LREG3: f_o_1 (input2 odd)
//   LREG4: g_e (accumulator for even output)
//   LREG5: g_o (accumulator for odd output)
//   LREG6: tmp (broadcast coefficient)
//   LREG14: lane mask for column 0, preloaded by _horizontal_stencil_init()
#pragma once

#include <array>
#include <algorithm>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "cmath_common.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline uint32_t _get_dst_base(
    const std::uint32_t tile_index,
    const std::uint32_t face_index
) {
    // addr uint10: XTTTFFRRCX
    // T: tile index (0-7) << 6
    // F: face index (0-3) << 4
    // R: row index (0,4,8,12)
    // C: column index (0 for even cols, 1 for odd cols)*2
    return (tile_index << 6) + (face_index << 4);
}

inline void _horizontal_stencil_init() {
    _init_sfpu_config_reg();

    addr_mod_t addr_mod{.dest = {.incr = 0}};
    addr_mod.set(ADDR_MOD_3);

    // Enable all lanes
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);

    // Compute mask
    // LREG7 = 0
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG7, 0);
    v_if((vConstTileId & 0xF) == 0) {
        // LREG = 1 if id%16 == 0 (first column only)
        TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, 0x0001);
    }
    v_endif;
}

// _horizontal_stencil_rotate_(a, b): 1-element right shift within subvectors.
// After rotation, column 0 of b gets column 0 of a (the element that
// "fell off" the right end of a wraps around, and we overwrite b's
// column 0 with a's column 0 via masking).
inline void _horizontal_stencil_rotate_(std::uint32_t a_reg, std::uint32_t b_reg) {
    // Shift a to the right
    TTI_SFPSHFT2(0, a_reg, a_reg, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    // Shift b to the right
    TTI_SFPSHFT2(0, b_reg, b_reg, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    // Set LaneEnable=true for lanes with col=0 and LaneEnable=false for others
    TTI_SFPSETCC(0, p_sfpu::LREG7, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    // Copy first column of a to the first column of b
    TTI_SFPMOV(0, a_reg, b_reg, 0);
    // Set all LaneEnable=true
    TTI_SFPENCC(0, 0, 0, sfpi::SFPENCC_MOD1_EU_R1);
}

// Arguments:
//   h_packed: array of K uint32_t values, each being the bit-cast of h[j] as float32
//   f0: input1 offset of 4x16 block
//   f1: input2 offset of 4x16 block
//   g: output offset of 4x16 block
//
// This function processes one set of 4 rows (one block) at a time.
// The caller must iterate over all rows and advance dst_reg accordingly.
template <uint8_t K>
inline void _horizontal_stencil_block(
    const uint32_t h_packed[K],
    const uint32_t dst_f0,  // dst index for input1
    const uint32_t dst_f1,  // dst index for input2
    const uint32_t dst_g    // dst index for output
) {
    // Register allocations
    const auto& f_e_0 = p_sfpu::LREG0;
    const auto& f_o_0 = p_sfpu::LREG1;
    const auto& f_e_1 = p_sfpu::LREG2;
    const auto& f_o_1 = p_sfpu::LREG3;
    const auto& g_e = p_sfpu::LREG4;
    const auto& g_o = p_sfpu::LREG5;
    const auto& tmp = p_sfpu::LREG6;

    // Load inputs into LRegs, for odd columns offset of 2 is used
    TT_SFPLOAD(f_e_0, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_f0);    // f_e_0
    TT_SFPLOAD(f_o_0, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_f0+2);  // f_o_0
    TT_SFPLOAD(f_e_1, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_f1);    // f_e_1
    TT_SFPLOAD(f_o_1, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_f1+2);  // f_o_1

    // Zero the output accumulators
    TTI_SFPMOV(0, p_sfpu::LCONST_0, g_e, 0);  // g_e = 0
    TTI_SFPMOV(0, p_sfpu::LCONST_0, g_o, 0);  // g_o = 0

    #pragma unroll 17
    for (uint8_t j = 0; j < K; j++) {
        TT_SFPLOADI(tmp, sfpi::SFPLOADI_MOD0_UPPER, (h_packed[j]) >> 16);
        TT_SFPLOADI(tmp, sfpi::SFPLOADI_MOD0_LOWER, (h_packed[j]) & 0xFFFF);

        if ((j & 1) == 0) {
            // g_e += h[j] * f_e_1
            TTI_SFPMAD(f_e_1, tmp, g_e, g_e, 0);
            // g_o += h[j] * f_o_1
            TTI_SFPMAD(f_o_1, tmp, g_o, g_o, 0);
        } else {
            // g_o += h[j] * f_e_1
            TTI_SFPMAD(f_e_1, tmp, g_o, g_o, 0);
            // Rotate odd columns: ROTATE(f_o_0, f_o_1)
            _horizontal_stencil_rotate_(f_o_0, f_o_1);
            // g_e += h[j] * f_o_1 (now shifted)
            TTI_SFPMAD(f_o_1, tmp, g_e, g_e, 0);
            // Rotate even columns: ROTATE(f_e_0, f_e_1)
            _horizontal_stencil_rotate_(f_e_0, f_e_1);
        }
    }

    // Store results back to Dst
    TT_SFPSTORE(g_e, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, dst_g);   // g_e
    TT_SFPSTORE(g_o, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, dst_g+2); // g_o
}

template <uint8_t K, uint32_t Rows>
inline void _horizontal_stencil_face(
    const uint32_t h_packed[K],
    const uint32_t input1,
    const uint32_t input2,
    const uint32_t output
) {
    constexpr uint32_t ROWS = std::min(Rows, static_cast<uint32_t>(16));
    constexpr uint32_t ROW_STRIDE = 4;

    #pragma unroll 4
    for (uint32_t row = 0; row < ROWS; row += ROW_STRIDE) {
        _horizontal_stencil_block<K>(
            h_packed,
            input1 + row,
            input2 + row,
            output + row
        );
    }
}

// Arguments:
//   h_packed: array of K uint32_t values, each being the bit-cast of h[j] as float32
//   input1: first tile index in dst register
//   input2: second tile index in dst register
//   output: tile index in dst register for output (can be same as either input)
// 
// Template arguments:
//   Rows: minimum number of Rows of the tile to be processed
template <uint8_t K, uint32_t Rows>
inline void _horizontal_stencil(
    const uint32_t h_packed[K],
    const uint32_t input1,
    const uint32_t input2,
    const uint32_t output
) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    // We use addr mod 3, so base=0
    ckernel::math::clear_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    _horizontal_stencil_face<K, Rows>(
        h_packed,
        _get_dst_base(input1, 0),
        _get_dst_base(input1, 1),
        _get_dst_base(output, 0)
    );
    _horizontal_stencil_face<K, Rows>(
        h_packed,
        _get_dst_base(input1, 1),
        _get_dst_base(input2, 0),
        _get_dst_base(output, 1)
    );

    if constexpr (Rows > 16) {
        _horizontal_stencil_face<K, Rows-16>(
            h_packed,
            _get_dst_base(input1, 2),
            _get_dst_base(input1, 3),
            _get_dst_base(output, 2)
        );
        _horizontal_stencil_face<K, Rows-16>(
            h_packed,
            _get_dst_base(input1, 3),
            _get_dst_base(input2, 2),
            _get_dst_base(output, 3)
        );
    }

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
}

}  // namespace sfpu
}  // namespace ckernel

inline void hstencil_init() {
    MATH((ckernel::sfpu::_horizontal_stencil_init()));
}

template<uint8_t K, uint32_t Rows = 32>
inline void hstencil_tile(
    std::array<uint32_t, K> h_packed,
    const uint32_t input1,
    const uint32_t input2,
    const uint32_t output
) {
    MATH((ckernel::sfpu::_horizontal_stencil<K, Rows>(
        h_packed.data(),
        input1,
        input2,
        output
    )));
}

template<uint8_t K>
inline void hstencil_row(
    std::array<uint32_t, K> h_packed,
    const uint32_t input1,
    const uint32_t input2,
    const uint32_t output
) {
    hstencil_tile<K, 1>(h_packed, input1, input2, output);
}
