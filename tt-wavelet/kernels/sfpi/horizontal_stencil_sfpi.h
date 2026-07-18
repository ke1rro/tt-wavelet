// Register allocation:
//   LREG0: f_e_0 (input1 even)
//   LREG1: f_o_0 (input1 odd)
//   LREG2: f_e_1 (input2 even)
//   LREG3: f_o_1 (input2 odd)
//   LREG4: g_e (accumulator for even output)
//   LREG5: g_o (accumulator for odd output)
//   LREG6: tmp (broadcast coefficient)
//   LREG7: temporary MAD result / Blackhole lane-position scratch
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "../ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

#ifndef TT_WAVELET_TENSIX_ARCH
#error "tt-wavelet architecture wrapper was not included"
#endif

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

inline uint32_t _get_narrow_dst_base(const std::uint32_t tile_index, const std::uint32_t face_index) {
    // A 32x16 FP32 tile occupies 32 destination rows rather than the 64 rows
    // used by a 32x32 tile.  Its two faces are the top and bottom 16x16 faces.
    return (tile_index << 5) + (face_index << 4);
}

inline void _horizontal_stencil_clear_addr_mod_base_() {
#if TT_WAVELET_TENSIX_ARCH == TT_WAVELET_TENSIX_ARCH_WORMHOLE
    // Wormhole can select address modifiers 4..7 through a programmable
    // base.  This kernel programs and uses ADDR_MOD_3, so restore that bank.
    ckernel::math::clear_addr_mod_base();
#elif TT_WAVELET_TENSIX_ARCH == TT_WAVELET_TENSIX_ARCH_BLACKHOLE
    // Blackhole has no alternate address-modifier bank or base selector.
#else
#error "Unsupported Tensix architecture for horizontal SFPI stencil"
#endif
}

inline void _horizontal_stencil_init() {
    _init_sfpu_config_reg();

    addr_mod_t addr_mod{.dest = {.incr = 0}};
    addr_mod.set(ADDR_MOD_3);
    math::reset_counters(p_setrwc::SET_ABD_F);

    // Enable all lanes
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
}

// _horizontal_stencil_rotate_(a, b): 1-element right shift within subvectors.
// After rotation, column 0 of b gets the element shifted out of a.
inline void _horizontal_stencil_rotate_(std::uint32_t a_reg, std::uint32_t b_reg) {
#if TT_WAVELET_TENSIX_ARCH == TT_WAVELET_TENSIX_ARCH_BLACKHOLE
    // Blackhole fixes the Wormhole SHFLSHR1 erratum: lane zero is now zero,
    // so it can no longer be used as an implicit cross-register halo.  Rotate
    // both registers contractually, then replace only lane zero of b with the
    // value rotated into lane zero of a.
    TTI_SFPSHFT2(0, a_reg, a_reg, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, b_reg, b_reg, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    // LREG15 contains lane*2. Masking its low four bits yields zero exactly
    // for lanes 0, 8, 16, and 24: column zero of every 8-lane subvector.
    // Blackhole's SFPAND USE_VB mode permits LREG15 as the first operand.
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_USHORT, 0x000f);
    TTI_SFPAND(p_sfpu::LTILEID, p_sfpu::LREG7, p_sfpu::LREG7, 1);
    TTI_SFPSETCC(0, p_sfpu::LREG7, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPMOV(0, a_reg, b_reg, 0);
    TTI_SFPENCC(0, 0, 0, sfpi::SFPENCC_MOD1_EU_R1);
#elif TT_WAVELET_TENSIX_ARCH == TT_WAVELET_TENSIX_ARCH_WORMHOLE
    // Shift a to the right. The following SHFLSHR1 uses this instruction's
    // source register as its lane-zero halo, giving b a non-wrapping shift.
    // This is a Wormhole-only fast path that relies on the documented
    // SHFLSHR1 hardware erratum and must not be compiled for Blackhole.
    TTI_SFPSHFT2(0, a_reg, a_reg, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, b_reg, b_reg, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    TTI_SFPNOP;
#else
#error "Unsupported Tensix architecture for horizontal SFPI stencil"
#endif
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

inline void _horizontal_stencil_scale_register_(
    const std::uint32_t value_reg, const std::uint32_t scalar_reg, const std::uint32_t product_reg) {
    TTI_SFPMUL(value_reg, scalar_reg, p_sfpu::LCONST_0, product_reg, 0);
    TTI_NOP;
    TTI_SFPMOV(0, product_reg, value_reg, 0);
}

template <
    uint8_t K,
    bool ScaleSource = false,
    bool ScaleBase = false,
    uint32_t SourceScalePacked = 0,
    uint32_t BaseScalePacked = 0>
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

    if constexpr (ScaleSource) {
        TT_SFPLOADI(tmp, sfpi::SFPLOADI_MOD0_UPPER, SourceScalePacked >> 16);
        TT_SFPLOADI(tmp, sfpi::SFPLOADI_MOD0_LOWER, SourceScalePacked & 0xFFFF);
        _horizontal_stencil_scale_register_(f_e_0, tmp, tmp_acc);
        _horizontal_stencil_scale_register_(f_o_0, tmp, tmp_acc);
        _horizontal_stencil_scale_register_(f_e_1, tmp, tmp_acc);
        _horizontal_stencil_scale_register_(f_o_1, tmp, tmp_acc);
    }

    TTI_SFPMOV(0, p_sfpu::LCONST_0, g_e, 0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, g_o, 0);
    TT_SFPLOAD(g_e, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_base);
    TT_SFPLOAD(g_o, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_base + 2);
    TTI_SFPNOP;
    TTI_SFPNOP;

    if constexpr (ScaleBase) {
        TT_SFPLOADI(tmp, sfpi::SFPLOADI_MOD0_UPPER, BaseScalePacked >> 16);
        TT_SFPLOADI(tmp, sfpi::SFPLOADI_MOD0_LOWER, BaseScalePacked & 0xFFFF);
        _horizontal_stencil_scale_register_(g_e, tmp, tmp_acc);
        _horizontal_stencil_scale_register_(g_o, tmp, tmp_acc);
    }

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
            // g_e += h[j] * f_o_1 (now shifted)
            _horizontal_stencil_mad_accumulate_(f_o_1, tmp, g_e, tmp_acc);
            // Rotate even columns: ROTATE(f_e_0, f_e_1)
            if (j != K - 1) {  // No need to rotate on the last iteration
                _horizontal_stencil_rotate_(f_e_0, f_e_1);
            }
        }
    }
    // TODO Check if NOPS needed
    TTI_SFPNOP;
    TTI_SFPNOP;
    TT_SFPSTORE(g_e, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, dst_g);
    TT_SFPSTORE(g_o, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, dst_g + 2);
}

template <
    uint8_t K,
    uint32_t Rows,
    bool ScaleSource = false,
    bool ScaleBase = false,
    uint32_t SourceScalePacked = 0,
    uint32_t BaseScalePacked = 0>
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
        _horizontal_stencil_plus_base_block<K, ScaleSource, ScaleBase, SourceScalePacked, BaseScalePacked>(
            h_packed, input1 + row, input2 + row, base + row, output + row);
    }
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
    _horizontal_stencil_clear_addr_mod_base_();
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

template <
    uint8_t K,
    bool ScaleSource = false,
    bool ScaleBase = false,
    uint32_t SourceScalePacked = 0,
    uint32_t BaseScalePacked = 0>
inline void _horizontal_stencil_plus_base_narrow(
    const uint32_t h_packed[K],
    const uint32_t source0,
    const uint32_t source1,
    const uint32_t source2,
    const uint32_t source3,
    const uint32_t base0,
    const uint32_t base1,
    const uint32_t base2) {
    math::set_dst_write_addr<DstTileShape::Tile32x16, UnpackDestination::SrcRegs>(0);
    _horizontal_stencil_clear_addr_mod_base_();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    const uint32_t sources[4] = {source0, source1, source2, source3};
    const uint32_t bases[3] = {base0, base1, base2};
#pragma unroll 3
    for (uint32_t block = 0; block < 3; ++block) {
#pragma unroll 2
        for (uint32_t face = 0; face < 2; ++face) {
            _horizontal_stencil_plus_base_face<K, 16, ScaleSource, ScaleBase, SourceScalePacked, BaseScalePacked>(
                h_packed,
                _get_narrow_dst_base(sources[block], face),
                _get_narrow_dst_base(sources[block + 1], face),
                _get_narrow_dst_base(bases[block], face),
                _get_narrow_dst_base(bases[block], face));
        }
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
    _horizontal_stencil_clear_addr_mod_base_();
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

inline void _scale_narrow_tile(const uint32_t tile, const uint32_t scalar_packed) {
    _init_sfpu_config_reg();

    addr_mod_t addr_mod{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    };
    addr_mod.set(ADDR_MOD_3);

    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    math::reset_counters(p_setrwc::SET_ABD_F);
    math::set_dst_write_addr<DstTileShape::Tile32x16, UnpackDestination::SrcRegs>(0);
    _horizontal_stencil_clear_addr_mod_base_();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    const auto& value = p_sfpu::LREG0;
    const auto& scalar = p_sfpu::LREG1;
    const auto& product = p_sfpu::LREG2;

    TT_SFPLOADI(scalar, sfpi::SFPLOADI_MOD0_UPPER, scalar_packed >> 16);
    TT_SFPLOADI(scalar, sfpi::SFPLOADI_MOD0_LOWER, scalar_packed & 0xFFFF);

#pragma unroll 2
    for (uint32_t face = 0; face < 2; ++face) {
        const uint32_t face_base = _get_narrow_dst_base(tile, face);
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
inline void hstencil_plus_base_narrow_tiles(
    std::array<uint32_t, K> h_packed,
    const uint32_t source0,
    const uint32_t source1,
    const uint32_t source2,
    const uint32_t source3,
    const uint32_t base0,
    const uint32_t base1,
    const uint32_t base2) {
    MATH((ckernel::sfpu::_horizontal_stencil_plus_base_narrow<K>(
        h_packed.data(), source0, source1, source2, source3, base0, base1, base2)));
}

template <uint8_t K, bool ScaleSource, bool ScaleBase, uint32_t SourceScalePacked, uint32_t BaseScalePacked>
inline void hstencil_scaled_inputs_plus_base_narrow_tiles(
    std::array<uint32_t, K> h_packed,
    const uint32_t source0,
    const uint32_t source1,
    const uint32_t source2,
    const uint32_t source3,
    const uint32_t base0,
    const uint32_t base1,
    const uint32_t base2) {
    MATH((ckernel::sfpu::
              _horizontal_stencil_plus_base_narrow<K, ScaleSource, ScaleBase, SourceScalePacked, BaseScalePacked>(
                  h_packed.data(), source0, source1, source2, source3, base0, base1, base2)));
}

inline void scale_narrow_tile(const uint32_t tile, const uint32_t scalar_packed) {
    MATH((ckernel::sfpu::_scale_narrow_tile(tile, scalar_packed)));
}
