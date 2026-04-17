#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif1_tag {};

inline constexpr auto coif1_scheme = make_lifting_scheme(
    6,
    1,
    2,
    PredictStep<1>{{-4.6457513110645898f}, {0xc094a9ffu}, 0},
    UpdateStep<2>{{0.20571891388307387f, -0.01673667737846022f}, {0x3e52a7fbu, 0xbc891b5bu}, 0},
    PredictStep<2>{{7.4686269665968821f, -91.80064794363031f}, {0x40eefefeu, 0xc2b799efu}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{0.010410807933185046f}, {0x3c2a9218u}, 0},
    ScaleEvenStep<1>{{-0.047338472275362822f}, {0xbd41e5fcu}, 0},
    ScaleOddStep<1>{{21.124467096933486f}, {0x41a8fee9u}, 0});

template <>
struct scheme_traits<coif1_tag> {
    using SchemeType = decltype(coif1_scheme);
    static constexpr const char* name = "coif1";
    static constexpr int id = 15;
    static constexpr int tap_size = 6;
    static constexpr int delay_even = 1;
    static constexpr int delay_odd = 2;
    static constexpr int num_steps = 7;
    static constexpr const auto& scheme = coif1_scheme;
};

}  // namespace ttwv
