#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct db4_tag {};

inline constexpr auto db4_scheme = make_lifting_scheme(
    8,
    2,
    2,
    PredictStep<1>{ { -0.32227588800028117f }, -1 },
    UpdateStep<2>{ { -1.117123605116217f, 0.29195312600347528f }, 0 },
    PredictStep<2>{ { -1.6889170665560465f, 0.54002828341971398f }, -1 },
    UpdateStep<2>{ { 0.0066173380106253708f, 0.55479469680433824f }, 0 },
    PredictStep<1>{ { -0.31909219261386168f }, 0 },
    ScaleEvenStep<1>{ { 2.6337752658977194f }, 0 },
    ScaleOddStep<1>{ { 0.37968311607602218f }, 0 }
);

template <>
struct scheme_traits<db4_tag> {
    using SchemeType = decltype(db4_scheme);
    static constexpr const char* name = "db4";
    static constexpr int id         = 64;
    static constexpr int tap_size   = 8;
    static constexpr int delay_even = 2;
    static constexpr int delay_odd  = 2;
    static constexpr int num_steps  = 7;
    static constexpr const auto& scheme = db4_scheme;
};

}
