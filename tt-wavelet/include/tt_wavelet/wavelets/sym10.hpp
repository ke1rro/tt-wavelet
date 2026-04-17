#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym10_tag {};

inline constexpr auto sym10_scheme = make_lifting_scheme(
    20,
    5,
    5,
    PredictStep<1>{{8.0533127778739217f}, {0x4100da5eu}, -1},
    UpdateStep<2>{{0.030264604896245172f, -0.12228698257837521f}, {0x3cf7ed7au, 0xbdfa7199u}, 0},
    PredictStep<2>{{8.7593620786409812f, -26.396907896909436f}, {0x410c2659u, 0xc1d32cdeu}, -1},
    UpdateStep<2>{{0.017810178317534842f, -0.034740951493743698f}, {0x3c91e6a7u, 0xbd0e4c87u}, 0},
    PredictStep<2>{{1.4814231376821672f, -5.7892552363227479f}, {0x3fbd9f46u, 0xc0b94194u}, -1},
    UpdateStep<2>{{-0.010952958767575448f, -0.0050434460428580054f}, {0xbc33740au, 0xbba5437eu}, 0},
    PredictStep<2>{{-9.3742758447654531f, 3.1303191173128559f}, {0xc115fd09u, 0x40485726u}, -1},
    UpdateStep<2>{{-0.066296990860538349f, 0.025762364949882807f}, {0xbd87c6b7u, 0x3cd30b98u}, 0},
    PredictStep<2>{{-13.123765639102318f, 10.231349589412808f}, {0xc151faf2u, 0x4123b39cu}, -1},
    UpdateStep<2>{{0.053245246606484074f, 0.05914837648354393f}, {0x3d5a17b0u, 0x3d724591u}, 0},
    PredictStep<1>{{-13.853349293041154f}, {0xc15da752u}, 0},
    ScaleEvenStep<1>{{2.8744125379022876f}, {0x4037f660u}, 0},
    ScaleOddStep<1>{{0.34789717440134332f}, {0x3eb21f94u}, 0});

template <>
struct scheme_traits<sym10_tag> {
    using SchemeType = decltype(sym10_scheme);
    static constexpr const char* name = "sym10";
    static constexpr int id = 87;
    static constexpr int tap_size = 20;
    static constexpr int delay_even = 5;
    static constexpr int delay_odd = 5;
    static constexpr int num_steps = 13;
    static constexpr const auto& scheme = sym10_scheme;
};

}  // namespace ttwv
