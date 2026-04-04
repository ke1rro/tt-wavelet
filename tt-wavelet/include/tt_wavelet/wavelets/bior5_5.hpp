#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct bior5_5_tag {};

inline constexpr auto bior5_5_scheme = make_lifting_scheme(
    12,
    3,
    3,
    PredictStep<2>{{4.993274520953169f, 4.993274520953169f}, -1},
    UpdateStep<2>{{-0.0043674455917754361f, -0.0043674455917754361f}, 0},
    PredictStep<2>{{-5.5857862004412357f, -5.5857862004412357f}, -1},
    UpdateStep<2>{{0.35223144284791624f, 0.35223144284791624f}, 0},
    PredictStep<2>{{-0.29009307325126349f, -0.29009307325126349f}, -1},
    ScaleEvenStep<1>{{0.92496193505949531f}, 0},
    ScaleOddStep<1>{{1.0811255707897625f}, 0});

template <>
struct scheme_traits<bior5_5_tag> {
    using SchemeType = decltype(bior5_5_scheme);
    static constexpr const char* name = "bior5.5";
    static constexpr int id = 13;
    static constexpr int tap_size = 12;
    static constexpr int delay_even = 3;
    static constexpr int delay_odd = 3;
    static constexpr int num_steps = 7;
    static constexpr const auto& scheme = bior5_5_scheme;
};

}  // namespace ttwv
