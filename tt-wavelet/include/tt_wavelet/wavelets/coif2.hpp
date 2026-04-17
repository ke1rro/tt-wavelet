#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif2_tag {};

inline constexpr auto coif2_scheme = make_lifting_scheme(
    12,
    3,
    3,
    PredictStep<1>{{0.39520948862008259f}, {0x3eca58e6u}, -1},
    UpdateStep<2>{{0.48655312628154684f, -0.34182037906645996f}, {0x3ef91d7eu, 0xbeaf0315u}, 0},
    PredictStep<2>{{-0.10235638480685388f, -0.49406182054950643f}, {0xbdd1a039u, 0xbefcf5acu}, -1},
    UpdateStep<2>{{-1.4797286989698755f, 0.13092196383207658f}, {0xbfbd67c0u, 0x3e061068u}, 0},
    PredictStep<2>{{-2.0172532392545133f, 0.42871598963852731f}, {0xc0011aadu, 0x3edb80aau}, -1},
    UpdateStep<2>{{0.0034275264729955995f, 0.48314673498579797f}, {0x3b60a05au, 0x3ef75f02u}, 0},
    PredictStep<1>{{-0.56297768922952474f}, {0xbf101f4eu}, 0},
    ScaleEvenStep<1>{{3.5782269574681429f}, {0x406501acu}, 0},
    ScaleOddStep<1>{{0.27946801918556141f}, {0x3e8f166fu}, 0});

template <>
struct scheme_traits<coif2_tag> {
    using SchemeType = decltype(coif2_scheme);
    static constexpr const char* name = "coif2";
    static constexpr int id = 24;
    static constexpr int tap_size = 12;
    static constexpr int delay_even = 3;
    static constexpr int delay_odd = 3;
    static constexpr int num_steps = 9;
    static constexpr const auto& scheme = coif2_scheme;
};

}  // namespace ttwv
