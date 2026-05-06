#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel::sfpu {

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
    _horizontal_rotate(p_sfpu::LREG0, p_sfpu::LREG2);
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

} // namespace ckernel::sfpu

template <uint8_t K>
inline void shift_splice(const uint32_t tile0, const uint32_t tile1) {
    MATH((ckernel::sfpu::_splice_shift<K>(tile0, tile1)));
}
