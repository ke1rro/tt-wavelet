#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym4_tag {};

inline constexpr auto sym4_scheme = make_lifting_scheme(
    8,
    2,
    2,
    PredictStep<1>{{2.5565839655041254f}, -1},
    UpdateStep<2>{{0.019031207046960385f, -0.33924399186570864f}, 0},
    PredictStep<2>{{-5.8769279791734448f, -1.0590572691142952f}, -1},
    UpdateStep<2>{{0.19494086178921297f, 0.065984607412632776f}, 0},
    PredictStep<1>{{-4.3440174091278969f}, 0},
    ScaleEvenStep<1>{{1.3592304016052734f}, 0},
    ScaleOddStep<1>{{0.73571044233485616f}, 0});

template <>
struct scheme_traits<sym4_tag> {
    using SchemeType = decltype(sym4_scheme);
    static constexpr const char* name = "sym4";
    static constexpr int id = 100;
    static constexpr int tap_size = 8;
    static constexpr int delay_even = 2;
    static constexpr int delay_odd = 2;
    static constexpr int num_steps = 7;
    static constexpr const auto& scheme = sym4_scheme;
};

}  // namespace ttwv
