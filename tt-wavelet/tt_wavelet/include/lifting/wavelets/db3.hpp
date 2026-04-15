#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct db3_tag {};

inline constexpr auto db3_scheme = make_lifting_scheme(
    6,
    1,
    2,
    PredictStep<1>{ { 2.4254972439119578f }, 0 },
    UpdateStep<2>{ { -0.35238765767485553f, -0.266042234902853f }, 0 },
    PredictStep<2>{ { 2.8953474541450976f, -14.93198122020725f }, -1 },
    SwapStep{ {}, 0 },
    PredictStep<1>{ { 0.066227766016837969f }, 0 },
    ScaleEvenStep<1>{ { 0.083742580136891953f }, 0 },
    ScaleOddStep<1>{ { -11.941356456480376f }, 0 }
);

template <>
struct scheme_traits<db3_tag> {
    using SchemeType = decltype(db3_scheme);
    static constexpr const char* name = "db3";
    static constexpr int id         = 54;
    static constexpr int tap_size   = 6;
    static constexpr int delay_even = 1;
    static constexpr int delay_odd  = 2;
    static constexpr int num_steps  = 7;
    static constexpr const auto& scheme = db3_scheme;
};

}
