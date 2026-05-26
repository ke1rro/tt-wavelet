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

inline void _splice_ops_init() {
    addr_mod_t addr_mod{.dest = {.incr = 0}};
    addr_mod.set(ADDR_MOD_3);

    // Enable all lanes
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
}

inline void _splice_pipeline_init() {
    _init_sfpu_config_reg();
    _splice_ops_init();
    // Build [1, 0, ..., 0] in each 8-lane group. Seeding SHFLROR1 with
    // LCONST_0 also makes the Wormhole SHFLSHR1 halo deterministic zero.
    TTI_SFPSHFT2(0, p_sfpu::LCONST_0, p_sfpu::LCONST_0, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPMOV(0, p_sfpu::LCONST_1, p_sfpu::LREG0, 0);
#pragma unroll 7
    for (uint32_t lane = 0; lane < 7; ++lane) {
        TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
        TTI_SFPNOP;
    }
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPCONFIG(0, p_sfpu::LREG14, 0);
    TTI_SFPNOP;
}

}  // namespace ckernel::sfpu

inline void splice_pipeline_init() { MATH((ckernel::sfpu::_splice_pipeline_init())); }

inline void splice_ops_init() { MATH((ckernel::sfpu::_splice_ops_init())); }
