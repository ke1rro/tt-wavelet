#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif3_tag {};

inline constexpr auto coif3_scheme = make_lifting_scheme(
    18,
    4,
    5,
    PredictStep<1>{{-2.0515539827866274f}, {0xc0034ca9u}, 0},
    UpdateStep<2>{{0.39385749847296075f, 0.21483935705595031f}, {0x3ec9a7b1u, 0x3e5bfed9u}, 0},
    PredictStep<2>{{-2.5880424409152916f, 0.59452010930386912f}, {0xc025a27du, 0x3f183278u}, -1},
    UpdateStep<2>{{-0.10426705485141172f, -0.05233608022579983f}, {0xbdd589f7u, 0xbd565e5cu}, 0},
    PredictStep<2>{{0.31292566283120704f, -1.0697411069980074f}, {0x3ea037cbu, 0xbf88ed47u}, -1},
    UpdateStep<2>{{0.15226372139055047f, -0.76473064671697244f}, {0x3e1beb05u, 0xbf43c563u}, 0},
    PredictStep<2>{{1.086318044410939f, -4.0885443565261168f}, {0x3f8b0c78u, 0xc082d55bu}, -1},
    UpdateStep<2>{{0.24111771738878687f, -0.43481425474025059f}, {0x3e76e790u, 0xbede9ff9u}, 0},
    PredictStep<2>{{2.2897052993756963f, -16.695716192526508f}, {0x40128a88u, 0xc18590d4u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{0.059890621454768894f}, {0x3d754fdeu}, 0},
    ScaleEvenStep<1>{{-0.023371966502539712f}, {0xbcbf7691u}, 0},
    ScaleOddStep<1>{{42.786301267860154f}, {0x422b252cu}, 0});

template <>
struct scheme_traits<coif3_tag> {
    using SchemeType = decltype(coif3_scheme);
    static constexpr const char* name = "coif3";
    static constexpr int id = 25;
    static constexpr int tap_size = 18;
    static constexpr int delay_even = 4;
    static constexpr int delay_odd = 5;
    static constexpr int num_steps = 13;
    static constexpr const auto& scheme = coif3_scheme;
};

}  // namespace ttwv
