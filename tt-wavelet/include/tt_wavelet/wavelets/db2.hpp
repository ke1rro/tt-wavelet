#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db2_tag {};

inline constexpr auto db2_scheme = make_lifting_scheme(
    4,
    1,
    1,
    PredictStep<1>{{-0.57735026918962584f}, -1},
    UpdateStep<2>{{0.20096189432334197f, 0.43301270189221924f}, 0},
    PredictStep<1>{{-0.33333333333333331f}, 0},
    ScaleEvenStep<1>{{1.1153550716504106f}, 0},
    ScaleOddStep<1>{{0.89657547216805344f}, 0});

template <>
struct scheme_traits<db2_tag> {
    using SchemeType = decltype(db2_scheme);
    static constexpr const char* name = "db2";
    static constexpr int id = 43;
    static constexpr int tap_size = 4;
    static constexpr int delay_even = 1;
    static constexpr int delay_odd = 1;
    static constexpr int num_steps = 5;
    static constexpr const auto& scheme = db2_scheme;
};

}  // namespace ttwv
