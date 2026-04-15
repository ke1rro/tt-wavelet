#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct bior6_8_tag {};

inline constexpr auto bior6_8_scheme = make_lifting_scheme(
    18,
    4,
    5,
    UpdateStep<2>{ { -0.99715069105011589f, -0.99715069105011589f }, -1 },
    PredictStep<2>{ { 0.27351197468674704f, 0.27351197468674704f }, 0 },
    UpdateStep<2>{ { -0.38746044045031081f, -0.38746044045031081f }, -1 },
    PredictStep<2>{ { -0.28650325796999498f, -0.28650325796999498f }, 0 },
    UpdateStep<2>{ { 0.54859416825250251f, 0.54859416825250251f }, -1 },
    PredictStep<4>{ { -0.099823217010263671f, 0.34381326275317581f, 0.34381326275317581f, -0.099823217010263671f }, -1 },
    SwapStep{ {}, 0 },
    ScaleEvenStep<1>{ { 1.1513061546402357f }, 0 },
    ScaleOddStep<1>{ { -0.86857869730791426f }, 0 }
);

template <>
struct scheme_traits<bior6_8_tag> {
    using SchemeType = decltype(bior6_8_scheme);
    static constexpr const char* name = "bior6.8";
    static constexpr int id         = 14;
    static constexpr int tap_size   = 18;
    static constexpr int delay_even = 4;
    static constexpr int delay_odd  = 5;
    static constexpr int num_steps  = 9;
    static constexpr const auto& scheme = bior6_8_scheme;
};

}
