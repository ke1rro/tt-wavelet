#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct db9_tag {};

inline constexpr auto db9_scheme = make_lifting_scheme(
    18,
    4,
    5,
    PredictStep<1>{ { 6.4035666702953069f }, 0 },
    UpdateStep<2>{ { -0.15244530713808124f, -0.011841125677809242f }, 0 },
    PredictStep<2>{ { 16.749528977304408f, -31.830582754431546f }, -1 },
    UpdateStep<2>{ { 0.014824497741866111f, -0.026099693868582081f }, 0 },
    PredictStep<2>{ { 28.150164304959713f, -52.674885304825466f }, -1 },
    UpdateStep<2>{ { 0.017209657282905352f, -0.038217096723663892f }, 0 },
    PredictStep<2>{ { 25.630328472293062f, -76.352637463934698f }, -1 },
    UpdateStep<2>{ { 0.013066335116885161f, -0.061206082615797415f }, 0 },
    PredictStep<2>{ { 16.336491052927066f, -163.83250334830785f }, -1 },
    SwapStep{ {}, 0 },
    PredictStep<1>{ { 0.0061037883922917362f }, 0 },
    ScaleEvenStep<1>{ { 0.0025114269497479055f }, 0 },
    ScaleOddStep<1>{ { -398.18000682853989f }, 0 }
);

template <>
struct scheme_traits<db9_tag> {
    using SchemeType = decltype(db9_scheme);
    static constexpr const char* name = "db9";
    static constexpr int id         = 69;
    static constexpr int tap_size   = 18;
    static constexpr int delay_even = 4;
    static constexpr int delay_odd  = 5;
    static constexpr int num_steps  = 13;
    static constexpr const auto& scheme = db9_scheme;
};

}
