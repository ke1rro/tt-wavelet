#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db9_tag {};

inline constexpr auto db9_scheme = make_lifting_scheme(
    18,
    4,
    5,
    PredictStep<1>{{6.4035666702953069f}, {0x40ccea05u}, 0},
    UpdateStep<2>{{-0.15244530713808124f, -0.011841125677809242f}, {0xbe1c1a9fu, 0xbc420148u}, 0},
    PredictStep<2>{{16.749528977304408f, -31.830582754431546f}, {0x4185ff09u, 0xc1fea509u}, -1},
    UpdateStep<2>{{0.014824497741866111f, -0.026099693868582081f}, {0x3c72e273u, 0xbcd5cf06u}, 0},
    PredictStep<2>{{28.150164304959713f, -52.674885304825466f}, {0x41e13389u, 0xc252b315u}, -1},
    UpdateStep<2>{{0.017209657282905352f, -0.038217096723663892f}, {0x3c8cfb44u, 0xbd1c8988u}, 0},
    PredictStep<2>{{25.630328472293062f, -76.352637463934698f}, {0x41cd0aeau, 0xc298b48du}, -1},
    UpdateStep<2>{{0.013066335116885161f, -0.061206082615797415f}, {0x3c56142fu, 0xbd7ab33bu}, 0},
    PredictStep<2>{{16.336491052927066f, -163.83250334830785f}, {0x4182b122u, 0xc323d51fu}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{0.0061037883922917362f}, {0x3bc8024au}, 0},
    ScaleEvenStep<1>{{0.0025114269497479055f}, {0x3b2496c1u}, 0},
    ScaleOddStep<1>{{-398.18000682853989f}, {0xc3c7170au}, 0});

template <>
struct scheme_traits<db9_tag> {
    using SchemeType = decltype(db9_scheme);
    static constexpr const char* name = "db9";
    static constexpr int id = 69;
    static constexpr int tap_size = 18;
    static constexpr int delay_even = 4;
    static constexpr int delay_odd = 5;
    static constexpr int num_steps = 13;
    static constexpr const auto& scheme = db9_scheme;
};

}  // namespace ttwv
