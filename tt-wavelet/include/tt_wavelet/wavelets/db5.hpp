#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db5_tag {};

inline constexpr auto db5_scheme = make_lifting_scheme(
    10,
    2,
    3,
    PredictStep<1>{{3.771519211689268f}, 0},
    UpdateStep<2>{{-0.24772929136032976f, -0.061736570470349854f}, 0},
    PredictStep<2>{{7.5975797354057519f, -19.027396275110604f}, -1},
    UpdateStep<2>{{0.044520705161802412f, -0.14201121405642173f}, 0},
    PredictStep<2>{{6.9189679172596978f, -44.228550858569371f}, -1},
    SwapStep{{}, 0},
    PredictStep<1>{{0.022600019079444381f}, 0},
    ScaleEvenStep<1>{{0.02169954477294184f}, 0},
    ScaleOddStep<1>{{-46.083916066614727f}, 0});

template <>
struct scheme_traits<db5_tag> {
    using SchemeType = decltype(db5_scheme);
    static constexpr const char* name = "db5";
    static constexpr int id = 65;
    static constexpr int tap_size = 10;
    static constexpr int delay_even = 2;
    static constexpr int delay_odd = 3;
    static constexpr int num_steps = 9;
    static constexpr const auto& scheme = db5_scheme;
};

}  // namespace ttwv
