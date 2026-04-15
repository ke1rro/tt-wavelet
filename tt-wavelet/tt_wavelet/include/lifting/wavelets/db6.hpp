#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct db6_tag {};

inline constexpr auto db6_scheme = make_lifting_scheme(
    12,
    3,
    3,
    PredictStep<1>{ { -0.22550617856378838f }, -1 },
    UpdateStep<2>{ { -0.7273420740972355f, 0.21459345000300822f }, 0 },
    PredictStep<2>{ { -1.1250225054190004f, 0.50700556856554491f }, -1 },
    UpdateStep<2>{ { -1.6002958831693423f, 0.65957141363468041f }, 0 },
    PredictStep<2>{ { -2.048404990496036f, 0.59003904064599322f }, -1 },
    UpdateStep<2>{ { 0.00032844464336151017f, 0.48580427914618152f }, 0 },
    PredictStep<1>{ { -0.2839909049744847f }, 0 },
    ScaleEvenStep<1>{ { 5.4225109087386487f }, 0 },
    ScaleOddStep<1>{ { 0.18441641092661515f }, 0 }
);

template <>
struct scheme_traits<db6_tag> {
    using SchemeType = decltype(db6_scheme);
    static constexpr const char* name = "db6";
    static constexpr int id         = 66;
    static constexpr int tap_size   = 12;
    static constexpr int delay_even = 3;
    static constexpr int delay_odd  = 3;
    static constexpr int num_steps  = 9;
    static constexpr const auto& scheme = db6_scheme;
};

}
