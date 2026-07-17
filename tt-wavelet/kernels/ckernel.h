#pragma once

// Project-local architecture contract.  TT-Metal's hardware-selected JIT
// defines exactly one ARCH_* macro; SFPI code should branch on this single
// value instead of repeating architecture detection throughout the kernels.
#define TT_WAVELET_TENSIX_ARCH_WORMHOLE 1
#define TT_WAVELET_TENSIX_ARCH_BLACKHOLE 2

// Forward to the architecture-specific TT LLK kernel header selected by the
// TT-Metal JIT.  Do not include both LLK trees: they expose the same header
// names with architecture-specific register and instruction definitions.
#if defined(ARCH_BLACKHOLE) && defined(ARCH_WORMHOLE)
#error "TT-Metal JIT defined both ARCH_BLACKHOLE and ARCH_WORMHOLE"
#elif defined(ARCH_BLACKHOLE)
#define TT_WAVELET_TENSIX_ARCH TT_WAVELET_TENSIX_ARCH_BLACKHOLE
#include "../../tt-metal/tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel.h"
#elif defined(ARCH_WORMHOLE)
#define TT_WAVELET_TENSIX_ARCH TT_WAVELET_TENSIX_ARCH_WORMHOLE
#include "../../tt-metal/tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel.h"
#else
#error "tt-wavelet SFPI kernels support only Wormhole and Blackhole"
#endif
