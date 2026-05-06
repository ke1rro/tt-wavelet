// Register allocation:
//   LREG0: f_e_0 (input1 even)
//   LREG1: f_o_0 (input1 odd)
//   LREG2: f_e_1 (input2 even)
//   LREG3: f_o_1 (input2 odd)
//   LREG4: g_e (accumulator for even output)
//   LREG5: g_o (accumulator for odd output)
//   LREG6: tmp (broadcast coefficient)
//   LREG7: tmp_acc (temporary MAD destination for fused plus-base)
//   LREG14: programmable column-zero mask preloaded by _horizontal_stencil_init()
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline uint32_t _get_dst_base(const std::uint32_t tile_index, const std::uint32_t face_index) {
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
    math::reset_counters(p_setrwc::SET_ABD_F);

    // Enable all lanes
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);

    // Compute mask in programmable LREG14. SFPCONFIG copies the first eight
    // lanes of LREG0 and broadcasts them vertically, so seed only lane zero.
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    v_if(vConstTileId == 0) { TT_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0001); }
    v_endif;
    TTI_SFPENCC(0, 0, 0, sfpi::SFPENCC_MOD1_EU_R1);
    TTI_SFPCONFIG(0, p_sfpu::LREG14, 0);
    TTI_SFPNOP;
}

// _horizontal_stencil_rotate_(a, b): 1-element right shift within subvectors.
// After rotation, column 0 of b gets the element shifted out of a.
inline void _horizontal_stencil_rotate_(std::uint32_t a_reg, std::uint32_t b_reg) {
    // Shift a to the right. The following SHFLSHR1 uses this instruction's
    // source register as its lane-zero halo, giving b a non-wrapping shift.
    TTI_SFPSHFT2(0, a_reg, a_reg, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, b_reg, b_reg, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    TTI_SFPNOP;
}

inline void _horizontal_stencil_mad_accumulate_(
    const std::uint32_t source_reg,
    const std::uint32_t coeff_reg,
    const std::uint32_t accumulator_reg,
    const std::uint32_t tmp_acc_reg) {
    TTI_SFPMAD(source_reg, coeff_reg, accumulator_reg, tmp_acc_reg, 0);
    TTI_NOP;
    TTI_SFPMOV(0, tmp_acc_reg, accumulator_reg, 0);
}

inline void _splice_shift_first_dreg_right_one(const std::uint32_t dreg) {
    TT_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dreg);
    TT_SFPLOAD(p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dreg + 2);

    TT_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, dreg + 2);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TT_SFPSTORE(p_sfpu::LREG1, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, dreg);
}

inline void _splice_shift_chained_dreg_right_one(const std::uint32_t prev_dreg, const std::uint32_t dreg) {
    TT_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, prev_dreg + 2);
    TT_SFPLOAD(p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dreg);
    TT_SFPLOAD(p_sfpu::LREG2, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dreg + 2);

    TT_SFPSTORE(p_sfpu::LREG1, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, dreg + 2);
    _horizontal_stencil_rotate_(p_sfpu::LREG0, p_sfpu::LREG2);
    TT_SFPSTORE(p_sfpu::LREG2, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, dreg);
}

inline void _splice_shift_row_band_right_one(
    const std::uint32_t dreg0, const std::uint32_t dreg1, const std::uint32_t dreg2, const std::uint32_t dreg3) {
    _splice_shift_chained_dreg_right_one(dreg2, dreg3);
    _splice_shift_chained_dreg_right_one(dreg1, dreg2);
    _splice_shift_chained_dreg_right_one(dreg0, dreg1);
    _splice_shift_first_dreg_right_one(dreg0);
}

template <uint8_t K>
inline void _splice_shift(const std::uint32_t tile0, const std::uint32_t tile1) {
    static_assert(K < 16, "splice_shift only supports K < 16");

    if constexpr (K == 0) {
        return;
    }

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    ckernel::math::clear_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    const std::uint32_t tile0_face0 = _get_dst_base(tile0, 0);
    const std::uint32_t tile0_face1 = _get_dst_base(tile0, 1);
    const std::uint32_t tile0_face2 = _get_dst_base(tile0, 2);
    const std::uint32_t tile0_face3 = _get_dst_base(tile0, 3);
    const std::uint32_t tile1_face0 = _get_dst_base(tile1, 0);
    const std::uint32_t tile1_face1 = _get_dst_base(tile1, 1);
    const std::uint32_t tile1_face2 = _get_dst_base(tile1, 2);
    const std::uint32_t tile1_face3 = _get_dst_base(tile1, 3);

#pragma unroll 15
    for (std::uint8_t step = 0; step < K; ++step) {
#pragma unroll 4
        for (std::uint32_t row = 0; row < 16; row += 4) {
            _splice_shift_row_band_right_one(
                tile0_face0 + row, tile0_face1 + row, tile1_face0 + row, tile1_face1 + row);
            _splice_shift_row_band_right_one(
                tile0_face2 + row, tile0_face3 + row, tile1_face2 + row, tile1_face3 + row);
        }
    }

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
}

inline std::uint32_t _splice_first_dreg(const std::uint32_t tile, const std::uint32_t row) {
    return _get_dst_base(tile, row < 16 ? 0 : 2) + (row & 0xC);
}

inline std::uint32_t _splice_last_dreg(const std::uint32_t tile, const std::uint32_t row) {
    return _get_dst_base(tile, row < 16 ? 1 : 3) + (row & 0xC);
}

template <std::uint32_t SrcRow, std::uint32_t DstRow>
inline void _splice_copy_dreg_row(const std::uint32_t src_dreg, const std::uint32_t dst_dreg) {
    static_assert(SrcRow < 4, "splice row copy source row must be in one DRegister");
    static_assert(DstRow < 4, "splice row copy destination row must be in one DRegister");

    TT_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, src_dreg);
    TT_SFPLOAD(p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, src_dreg + 2);
    TT_SFPLOAD(p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_dreg);
    TT_SFPLOAD(p_sfpu::LREG5, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_dreg + 2);

    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0 + SrcRow, p_sfpu::LREG4 + DstRow, 0);
    TTI_SFPTRANSP(0, 0, 0, 0);

    TT_SFPSTORE(p_sfpu::LREG4, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, dst_dreg);
    TT_SFPSTORE(p_sfpu::LREG5, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, dst_dreg + 2);
}

template <std::uint32_t Row>
inline void _splice_recover1_rows(const std::uint32_t tile0, const std::uint32_t tile1) {
    if constexpr (Row < 32) {
        _splice_copy_dreg_row<(Row - 1) & 0x3, Row & 0x3>(
            _splice_last_dreg(tile1, Row - 1), _splice_first_dreg(tile0, Row));
        _splice_recover1_rows<Row + 1>(tile0, tile1);
    }
}

template <std::uint32_t Row>
inline void _splice_recover2_rows(const std::uint32_t tile0, const std::uint32_t tile1) {
    if constexpr (Row < 31) {
        _splice_copy_dreg_row<(Row + 1) & 0x3, Row & 0x3>(
            _splice_first_dreg(tile0, Row + 1), _splice_last_dreg(tile1, Row));
        _splice_recover2_rows<Row + 1>(tile0, tile1);
    }
}

inline void _splice_recover1(const std::uint32_t tile0, const std::uint32_t tile1, const std::uint32_t tmp_tile) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    ckernel::math::clear_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    const std::uint32_t tmp_dreg = _get_dst_base(tmp_tile, 0);

    _splice_copy_dreg_row<0, 0>(tmp_dreg, _splice_first_dreg(tile0, 0));
    _splice_recover1_rows<1>(tile0, tile1);
    _splice_copy_dreg_row<3, 0>(_splice_last_dreg(tile1, 31), tmp_dreg);

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
}

inline void _splice_recover2(const std::uint32_t tile0, const std::uint32_t tile1, const std::uint32_t tmp_tile) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    ckernel::math::clear_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    const std::uint32_t tmp_dreg = _get_dst_base(tmp_tile, 0);

    _splice_copy_dreg_row<3, 3>(tmp_dreg, _splice_last_dreg(tile1, 31));
    _splice_recover2_rows<0>(tile0, tile1);
    _splice_copy_dreg_row<0, 3>(_splice_first_dreg(tile0, 0), tmp_dreg);

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
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
    TT_SFPLOAD(f_e_0, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_f0);      // f_e_0
    TT_SFPLOAD(f_o_0, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_f0 + 2);  // f_o_0
    TT_SFPLOAD(f_e_1, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_f1);      // f_e_1
    TT_SFPLOAD(f_o_1, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_f1 + 2);  // f_o_1

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
    TT_SFPSTORE(g_e, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, dst_g);      // g_e
    TT_SFPSTORE(g_o, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, dst_g + 2);  // g_o
}

// Same stencil as _horizontal_stencil_block, but initializes the output
// accumulators from a base tile so the lifting step is base + stencil(source)
// in one SFPU accumulation chain.
template <uint8_t K>
inline void _horizontal_stencil_plus_base_block(
    const uint32_t h_packed[K],
    const uint32_t dst_f0,
    const uint32_t dst_f1,
    const uint32_t dst_base,
    const uint32_t dst_g) {
    const auto& f_e_0 = p_sfpu::LREG0;
    const auto& f_o_0 = p_sfpu::LREG1;
    const auto& f_e_1 = p_sfpu::LREG2;
    const auto& f_o_1 = p_sfpu::LREG3;
    const auto& g_e = p_sfpu::LREG4;
    const auto& g_o = p_sfpu::LREG5;
    const auto& tmp = p_sfpu::LREG6;
    const auto& tmp_acc = p_sfpu::LREG7;

    TT_SFPLOAD(f_e_0, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_f0);
    TT_SFPLOAD(f_o_0, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_f0 + 2);
    TT_SFPLOAD(f_e_1, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_f1);
    TT_SFPLOAD(f_o_1, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_f1 + 2);

    TTI_SFPMOV(0, p_sfpu::LCONST_0, g_e, 0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, g_o, 0);
    TT_SFPLOAD(g_e, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_base);
    TT_SFPLOAD(g_o, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_base + 2);
    TTI_SFPNOP;
    TTI_SFPNOP;

#pragma unroll 17
    for (uint8_t j = 0; j < K; j++) {
        TT_SFPLOADI(tmp, sfpi::SFPLOADI_MOD0_UPPER, (h_packed[j]) >> 16);
        TT_SFPLOADI(tmp, sfpi::SFPLOADI_MOD0_LOWER, (h_packed[j]) & 0xFFFF);

        if ((j & 1) == 0) {
            _horizontal_stencil_mad_accumulate_(f_e_1, tmp, g_e, tmp_acc);
            _horizontal_stencil_mad_accumulate_(f_o_1, tmp, g_o, tmp_acc);
        } else {
            _horizontal_stencil_mad_accumulate_(f_e_1, tmp, g_o, tmp_acc);
            _horizontal_stencil_rotate_(f_o_0, f_o_1);
            _horizontal_stencil_mad_accumulate_(f_o_1, tmp, g_e, tmp_acc);
            _horizontal_stencil_rotate_(f_e_0, f_e_1);
        }
    }

    TTI_SFPNOP;
    TTI_SFPNOP;
    TT_SFPSTORE(g_e, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, dst_g);
    TT_SFPSTORE(g_o, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, dst_g + 2);
}

template <uint8_t K, uint32_t Rows>
inline void _horizontal_stencil_face(
    const uint32_t h_packed[K], const uint32_t input1, const uint32_t input2, const uint32_t output) {
    constexpr uint32_t ROWS = std::min(Rows, static_cast<uint32_t>(16));
    constexpr uint32_t ROW_STRIDE = 4;

#pragma unroll 4
    for (uint32_t row = 0; row < ROWS; row += ROW_STRIDE) {
        _horizontal_stencil_block<K>(h_packed, input1 + row, input2 + row, output + row);
    }
}

template <uint8_t K, uint32_t Rows>
inline void _horizontal_stencil_plus_base_face(
    const uint32_t h_packed[K],
    const uint32_t input1,
    const uint32_t input2,
    const uint32_t base,
    const uint32_t output) {
    constexpr uint32_t ROWS = std::min(Rows, static_cast<uint32_t>(16));
    constexpr uint32_t ROW_STRIDE = 4;

#pragma unroll 4
    for (uint32_t row = 0; row < ROWS; row += ROW_STRIDE) {
        _horizontal_stencil_plus_base_block<K>(h_packed, input1 + row, input2 + row, base + row, output + row);
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
    const uint32_t output1,
    const uint32_t output2) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    // We use addr mod 3, so base=0
    ckernel::math::clear_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    _horizontal_stencil_face<K, Rows>(
        h_packed, _get_dst_base(input1, 0), _get_dst_base(input1, 1), _get_dst_base(output1, 0));
    _horizontal_stencil_face<K, Rows>(
        h_packed, _get_dst_base(input1, 1), _get_dst_base(input2, 0), _get_dst_base(output1, 1));
    _horizontal_stencil_face<K, Rows>(
        h_packed, _get_dst_base(input2, 0), _get_dst_base(input2, 1), _get_dst_base(output2, 0));

    if constexpr (Rows > 16) {
        _horizontal_stencil_face<K, Rows - 16>(
            h_packed, _get_dst_base(input1, 2), _get_dst_base(input1, 3), _get_dst_base(output1, 2));
        _horizontal_stencil_face<K, Rows - 16>(
            h_packed, _get_dst_base(input1, 3), _get_dst_base(input2, 2), _get_dst_base(output1, 3));
        _horizontal_stencil_face<K, Rows>(
            h_packed, _get_dst_base(input2, 2), _get_dst_base(input2, 3), _get_dst_base(output2, 2));
    }

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
}

template <uint8_t K, uint32_t Rows>
inline void _horizontal_stencil_plus_base(
    const uint32_t h_packed[K],
    const uint32_t input1,
    const uint32_t input2,
    const uint32_t base1,
    const uint32_t base2,
    const uint32_t output1,
    const uint32_t output2) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    ckernel::math::clear_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    _horizontal_stencil_plus_base_face<K, Rows>(
        h_packed,
        _get_dst_base(input1, 0),
        _get_dst_base(input1, 1),
        _get_dst_base(base1, 0),
        _get_dst_base(output1, 0));
    _horizontal_stencil_plus_base_face<K, Rows>(
        h_packed,
        _get_dst_base(input1, 1),
        _get_dst_base(input2, 0),
        _get_dst_base(base1, 1),
        _get_dst_base(output1, 1));
    _horizontal_stencil_plus_base_face<K, Rows>(
        h_packed,
        _get_dst_base(input2, 0),
        _get_dst_base(input2, 1),
        _get_dst_base(base2, 0),
        _get_dst_base(output2, 0));

    if constexpr (Rows > 16) {
        _horizontal_stencil_plus_base_face<K, Rows - 16>(
            h_packed,
            _get_dst_base(input1, 2),
            _get_dst_base(input1, 3),
            _get_dst_base(base1, 2),
            _get_dst_base(output1, 2));
        _horizontal_stencil_plus_base_face<K, Rows - 16>(
            h_packed,
            _get_dst_base(input1, 3),
            _get_dst_base(input2, 2),
            _get_dst_base(base1, 3),
            _get_dst_base(output1, 3));
        _horizontal_stencil_plus_base_face<K, Rows>(
            h_packed,
            _get_dst_base(input2, 2),
            _get_dst_base(input2, 3),
            _get_dst_base(base2, 2),
            _get_dst_base(output2, 2));
    }

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
}

inline void _scale_tile(const uint32_t tile, const uint32_t scalar_packed) {
    _init_sfpu_config_reg();

    addr_mod_t addr_mod{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    };
    addr_mod.set(ADDR_MOD_3);

    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    math::reset_counters(p_setrwc::SET_ABD_F);
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    ckernel::math::clear_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    const auto& value = p_sfpu::LREG0;
    const auto& scalar = p_sfpu::LREG1;
    const auto& product = p_sfpu::LREG2;

    TT_SFPLOADI(scalar, sfpi::SFPLOADI_MOD0_UPPER, scalar_packed >> 16);
    TT_SFPLOADI(scalar, sfpi::SFPLOADI_MOD0_LOWER, scalar_packed & 0xFFFF);

#pragma unroll 4
    for (uint32_t face = 0; face < 4; ++face) {
        const uint32_t face_base = _get_dst_base(tile, face);
#pragma unroll 8
        for (uint32_t row = 0; row < 16; row += 4) {
            TT_SFPLOAD(value, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, face_base + row);
            TTI_SFPMUL(value, scalar, p_sfpu::LCONST_0, product, 0);
            TTI_NOP;
            TT_SFPSTORE(product, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, face_base + row);

            TT_SFPLOAD(value, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, face_base + row + 2);
            TTI_SFPMUL(value, scalar, p_sfpu::LCONST_0, product, 0);
            TTI_NOP;
            TT_SFPSTORE(product, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, face_base + row + 2);
        }
    }

    math::clear_dst_reg_addr();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
}

}  // namespace sfpu
}  // namespace ckernel

inline void hstencil_init() { MATH((ckernel::sfpu::_horizontal_stencil_init())); }

inline void splice_ops_init() { MATH((ckernel::sfpu::_horizontal_stencil_init())); }

template <uint8_t K, uint32_t Rows = 32>
inline void hstencil_tile(
    std::array<uint32_t, K> h_packed,
    const uint32_t input1,
    const uint32_t input2,
    const uint32_t output1,
    const uint32_t output2) {
    MATH((ckernel::sfpu::_horizontal_stencil<K, Rows>(h_packed.data(), input1, input2, output1, output2)));
}

template <uint8_t K, uint32_t Rows = 32>
inline void hstencil_plus_base_tile(
    std::array<uint32_t, K> h_packed,
    const uint32_t input1,
    const uint32_t input2,
    const uint32_t base1,
    const uint32_t base2,
    const uint32_t output1,
    const uint32_t output2) {
    MATH((ckernel::sfpu::_horizontal_stencil_plus_base<K, Rows>(
        h_packed.data(), input1, input2, base1, base2, output1, output2)));
}

inline void scale_tile(const uint32_t tile, const uint32_t scalar_packed) {
    MATH((ckernel::sfpu::_scale_tile(tile, scalar_packed)));
}

template <uint8_t K>
inline void splice_shift(const uint32_t tile0, const uint32_t tile1) {
    MATH((ckernel::sfpu::_splice_shift<K>(tile0, tile1)));
}

inline void splice_recover1(const uint32_t tile0, const uint32_t tile1, const uint32_t tmp_tile) {
    MATH((ckernel::sfpu::_splice_recover1(tile0, tile1, tmp_tile)));
}

inline void splice_recover2(const uint32_t tile0, const uint32_t tile1, const uint32_t tmp_tile) {
    MATH((ckernel::sfpu::_splice_recover2(tile0, tile1, tmp_tile)));
}
