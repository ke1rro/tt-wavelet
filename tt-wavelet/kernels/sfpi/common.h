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

inline uint32_t _get_dst_base(const std::uint32_t tile_index, const std::uint32_t face_index) {
    // addr uint10: XTTTFFRRCX
    // T: tile index (0-7) << 6
    // F: face index (0-3) << 4
    // R: row index (0,4,8,12)
    // C: column index (0 for even cols, 1 for odd cols)*2
    return (tile_index << 6) + (face_index << 4);
}

// horizontal_rotate(a, b): 1-element right shift within DoubleRegs
// After rotation, column 0 of b gets the element shifted out of a.
inline void _horizontal_rotate(std::uint32_t a_reg, std::uint32_t b_reg) {
    // Shift a to the right. The following SHFLSHR1 uses this instruction's
    // source register as its lane-zero halo, giving b a non-wrapping shift.
    TTI_SFPSHFT2(0, a_reg, a_reg, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, b_reg, b_reg, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    TTI_SFPNOP;
    // Non-contractual behavior of SHFLSHR1 was abused here, thus if
    // porting to future architectures, use masked SFPMOV to achieve the same effect
}

} // namespace ckernel::sfpu
