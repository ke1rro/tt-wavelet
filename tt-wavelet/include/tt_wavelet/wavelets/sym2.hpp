#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym2_tag {};

inline constexpr auto sym2_scheme = make_lifting_scheme(
    4,
    1,
    1,
    PredictStep<1>{{-0.57735026918961241f}, -1},
    UpdateStep<2>{{0.20096189432284639f, 0.433012701892587f}, 0},
    PredictStep<1>{{-0.33333333333314979f}, 0},
    ScaleEvenStep<1>{{1.1153550716502658f}, 0},
    ScaleOddStep<1>{{0.8965754721681698f}, 0});

template <>
struct scheme_traits<sym2_tag> {
    using SchemeType = decltype(sym2_scheme);
    static constexpr const char* name = "sym2";
    static constexpr int id = 97;
    static constexpr int tap_size = 4;
    static constexpr int delay_even = 1;
    static constexpr int delay_odd = 1;
    static constexpr int num_steps = 5;
    static constexpr const auto& scheme = sym2_scheme;
};

}  // namespace ttwv
