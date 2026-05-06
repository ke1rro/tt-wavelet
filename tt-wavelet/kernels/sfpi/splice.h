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
    _init_sfpu_config_reg();

    addr_mod_t addr_mod{.dest = {.incr = 0}};
    addr_mod.set(ADDR_MOD_3);

    // Enable all lanes
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);

    // Compute mask in programmable LREG14. SFPCONFIG copies the first eight
    // lanes of LREG0 and broadcasts them vertically, so seed only lane zero.
    // TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    // v_if(vConstTileId == 0) { TT_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0001); }
    // v_endif;
    // TTI_SFPENCC(0, 0, 0, sfpi::SFPENCC_MOD1_EU_R1);
    // TTI_SFPCONFIG(0, p_sfpu::LREG14, 0);
    // TTI_SFPNOP;
    // This is not used right now
}


} // namespace ckernel::sfpu

inline void splice_ops_init() { MATH((ckernel::sfpu::_splice_ops_init())); }
