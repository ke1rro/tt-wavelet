#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db7_tag {};

inline constexpr auto db7_scheme = make_lifting_scheme(
    14,
    3,
    4,
    PredictStep<1>{{5.0934984843036126f}, 0},
    UpdateStep<2>{{-0.1890420920719923f, -0.023998406989475848f}, 0},
    PredictStep<2>{{12.285444996734372f, -25.327550146946159f}, -1},
    UpdateStep<2>{{0.025264889214113569f, -0.052400135475944611f}, 0},
    PredictStep<2>{{16.876128620184218f, -42.943152427916097f}, -1},
    UpdateStep<2>{{0.022825428696966656f, -0.087609166135817623f}, 0},
    PredictStep<2>{{11.398697950329087f, -92.910379385533346f}, -1},
    SwapStep{{}, 0},
    PredictStep<1>{{0.01076283795702062f}, 0},
    ScaleEvenStep<1>{{0.0069928524547264326f}, 0},
    ScaleOddStep<1>{{-143.00316022313686f}, 0});

template <>
struct scheme_traits<db7_tag> {
    using SchemeType = decltype(db7_scheme);
    static constexpr const char* name = "db7";
    static constexpr int id = 67;
    static constexpr int tap_size = 14;
    static constexpr int delay_even = 3;
    static constexpr int delay_odd = 4;
    static constexpr int num_steps = 11;
    static constexpr const auto& scheme = db7_scheme;
};

}  // namespace ttwv
