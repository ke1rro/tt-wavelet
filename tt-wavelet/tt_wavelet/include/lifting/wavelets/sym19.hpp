#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct sym19_tag {};

inline constexpr auto sym19_scheme = make_lifting_scheme(
    38,
    9,
    10,
    PredictStep<1>{ { 1.1778363807084933f }, 0 },
    UpdateStep<2>{ { -0.49337624153367698f, 0.10718239644640744f }, 0 },
    PredictStep<2>{ { -0.57331670044723571f, 17.757542226642059f }, -1 },
    UpdateStep<2>{ { -0.055431587950508322f, -0.001248308377124182f }, 0 },
    PredictStep<2>{ { 24.730098517753493f, -653.22564945235763f }, -1 },
    UpdateStep<2>{ { 0.0014649499135974122f, -0.00020422129308200461f }, 0 },
    PredictStep<2>{ { 1476.9960705811795f, 1806.1683842259984f }, -1 },
    UpdateStep<2>{ { -0.00021726882356400698f, -0.00026517118927846662f }, 0 },
    PredictStep<2>{ { 1849.1367685612267f, 283.20656873397149f }, -1 },
    UpdateStep<2>{ { -7.7926694336290638e-05f, 0.00052928445612608881f }, 0 },
    PredictStep<2>{ { -963.68810434168074f, -210.54836117513807f }, -1 },
    UpdateStep<2>{ { 0.00022485511915643793f, 0.0029444257535002722f }, 0 },
    PredictStep<2>{ { -303.95536993538076f, -37.871315970072395f }, -1 },
    UpdateStep<2>{ { 0.003084950424882933f, 0.04511574939929059f }, 0 },
    PredictStep<2>{ { -21.408529081459903f, -2.4242704809051596f }, -1 },
    UpdateStep<2>{ { 0.1098123428743676f, 0.095290722493559687f }, 0 },
    PredictStep<2>{ { -2.2517435869776903f, 1.3460695949601784f }, -1 },
    UpdateStep<2>{ { -0.066075129588057058f, -0.13300237155989392f }, 0 },
    PredictStep<2>{ { 2.1310670509213989f, -10.811191962466632f }, -1 },
    SwapStep{ {}, 0 },
    PredictStep<1>{ { 0.08422345335496273f }, 0 },
    ScaleEvenStep<1>{ { -0.16247171486763512f }, 0 },
    ScaleOddStep<1>{ { 6.1549174932676429f }, 0 }
);

template <>
struct scheme_traits<sym19_tag> {
    using SchemeType = decltype(sym19_scheme);
    static constexpr const char* name = "sym19";
    static constexpr int id         = 96;
    static constexpr int tap_size   = 38;
    static constexpr int delay_even = 9;
    static constexpr int delay_odd  = 10;
    static constexpr int num_steps  = 23;
    static constexpr const auto& scheme = sym19_scheme;
};

}
