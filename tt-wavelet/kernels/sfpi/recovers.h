#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

#include "../sfpi/common.h"

using namespace sfpi;

namespace ckernel::sfpu {

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

} // namespace ckernel::sfpu

inline void recover1_splice(const uint32_t tile0, const uint32_t tile1, const uint32_t tmp_tile) {
    MATH((ckernel::sfpu::_splice_recover1(tile0, tile1, tmp_tile)));
}

inline void recover2_splice(const uint32_t tile0, const uint32_t tile1, const uint32_t tmp_tile) {
    MATH((ckernel::sfpu::_splice_recover2(tile0, tile1, tmp_tile)));
}
