#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct bior6_8_tag {};

inline constexpr auto bior6_8_scheme = make_lifting_scheme(
    18,
    4,
    5,
    UpdateStep<2>{{-0.99715069105011589f, -0.99715069105011589f}, {0xbf7f4545u, 0xbf7f4545u}, -1},
    PredictStep<2>{{0.27351197468674704f, 0.27351197468674704f}, {0x3e8c09c3u, 0x3e8c09c3u}, 0},
    UpdateStep<2>{{-0.38746044045031081f, -0.38746044045031081f}, {0xbec66137u, 0xbec66137u}, -1},
    PredictStep<2>{{-0.28650325796999498f, -0.28650325796999498f}, {0xbe92b08eu, 0xbe92b08eu}, 0},
    UpdateStep<2>{{0.54859416825250251f, 0.54859416825250251f}, {0x3f0c70abu, 0x3f0c70abu}, -1},
    PredictStep<4>{
        {-0.099823217010263671f, 0.34381326275317581f, 0.34381326275317581f, -0.099823217010263671f},
        {0xbdcc701du, 0x3eb0084bu, 0x3eb0084bu, 0xbdcc701du},
        -1},
    SwapStep{{}, {}, 0},
    ScaleEvenStep<1>{{1.1513061546402357f}, {0x3f935e00u}, 0},
    ScaleOddStep<1>{{-0.86857869730791426f}, {0xbf5e5b2cu}, 0});

template <>
struct scheme_traits<bior6_8_tag> {
    using SchemeType = decltype(bior6_8_scheme);
    static constexpr const char* name = "bior6.8";
    static constexpr int id = 14;
    static constexpr int tap_size = 18;
    static constexpr int delay_even = 4;
    static constexpr int delay_odd = 5;
    static constexpr int num_steps = 9;
    static constexpr const auto& scheme = bior6_8_scheme;
};

}  // namespace ttwv
