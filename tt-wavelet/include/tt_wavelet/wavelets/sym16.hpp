#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym16_tag {};

inline constexpr auto sym16_scheme = make_lifting_scheme(
    32,
    8,
    8,
    PredictStep<1>{{-2.0009294470554999f}, {0xc0000f3au}, -1},
    UpdateStep<2>{{0.058472500207035427f, 0.399888480184173f}, {0x3d6f80dcu, 0x3eccbe2fu}, 0},
    PredictStep<2>{{-4.3572196681229185f, -1.3485479258700559f}, {0xc08b6e58u, 0xbfac9d38u}, -1},
    UpdateStep<2>{{-0.04005402474673804f, 0.10830846401803426f}, {0xbd240fb0u, 0x3dddd0d4u}, 0},
    PredictStep<2>{{4.6335478667165146f, 2.7190584512670743f}, {0x40944606u, 0x402e050eu}, -1},
    UpdateStep<2>{{-0.099714707515253465f, -0.056533388630127732f}, {0xbdcc3739u, 0xbd678f8eu}, 0},
    PredictStep<2>{{-2.7743658971170668f, 5.2625772735711092f}, {0xc0318f36u, 0x40a86708u}, -1},
    UpdateStep<2>{{-0.028412737447851014f, 0.084639222325947477f}, {0xbce8c1d4u, 0x3dad5754u}, 0},
    PredictStep<2>{{2.5491048339795443f, 1.1764583769156958f}, {0x40232489u, 0x3f969630u}, -1},
    UpdateStep<2>{{0.17657864980656574f, -0.054796007033209831f}, {0x3e34d109u, 0xbd6071c7u}, 0},
    PredictStep<2>{{1.79100979927172f, -3.5547790545018678f}, {0x3fe53fcfu, 0xc0638180u}, -1},
    UpdateStep<2>{{-0.088294643826915181f, -0.16734214058369287f}, {0xbdb4d3d2u, 0xbe2b5bbdu}, 0},
    PredictStep<2>{{7.6009815074568658f, 1.2057650469795707f}, {0x40f33b3eu, 0x3f9a5682u}, -1},
    UpdateStep<2>{{0.0098322600920704526f, -0.10862082606411104f}, {0x3c21177du, 0xbdde7498u}, 0},
    PredictStep<2>{{-19.868289538849499f, -3.7983543245783435f}, {0xc19ef242u, 0xc073183du}, -1},
    UpdateStep<2>{{0.014513169155310342f, 0.025916165867064363f}, {0x3c6dc8a5u, 0x3cd44e24u}, 0},
    PredictStep<1>{{-17.208280790287219f}, {0xc189aa8fu}, 0},
    ScaleEvenStep<1>{{5.4612968924947136f}, {0x40aec2f2u}, 0},
    ScaleOddStep<1>{{0.18310669053247555f}, {0x3e3b8052u}, 0});

template <>
struct scheme_traits<sym16_tag> {
    using SchemeType = decltype(sym16_scheme);
    static constexpr const char* name = "sym16";
    static constexpr int id = 93;
    static constexpr int tap_size = 32;
    static constexpr int delay_even = 8;
    static constexpr int delay_odd = 8;
    static constexpr int num_steps = 19;
    static constexpr const auto& scheme = sym16_scheme;
};

}  // namespace ttwv
