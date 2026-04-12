#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym7_tag {};

inline constexpr auto sym7_scheme = make_lifting_scheme(
    14,
    3,
    4,
    PredictStep<1>{{0.39055082667931068f}, {0x3ec7f647u}, 0},
    UpdateStep<2>{{-0.33886392687924444f, -0.10483867067978561f}, {0xbead7f93u, 0xbdd6b5a8u}, 0},
    PredictStep<2>{{0.13725594548684125f, 1.3542211315828436f}, {0x3e0c8cd3u, 0x3fad571eu}, -1},
    UpdateStep<2>{{-0.43344969711109066f, -96.112960786057371f}, {0xbedded1eu, 0xc2c039d6u}, 0},
    PredictStep<2>{{0.010404275154054418f, -8.6562567078544854e-06f}, {0x3c2a76b1u, 0xb7113a57u}, -1},
    UpdateStep<2>{{5330.183535373304f, 17127.221303866958f}, {0x45a69178u, 0x4685ce71u}, 0},
    PredictStep<2>{{-1.9447466418965284e-05f, -0.00010536325382433175f}, {0xb7a3231eu, 0xb8dcf677u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{8884.9038519905553f}, {0x460ad39eu}, 0},
    ScaleEvenStep<1>{{48.171932823512954f}, {0x4240b00fu}, 0},
    ScaleOddStep<1>{{-0.020758976054867685f}, {0xbcaa0ebau}, 0});

template <>
struct scheme_traits<sym7_tag> {
    using SchemeType = decltype(sym7_scheme);
    static constexpr const char* name = "sym7";
    static constexpr int id = 103;
    static constexpr int tap_size = 14;
    static constexpr int delay_even = 3;
    static constexpr int delay_odd = 4;
    static constexpr int num_steps = 11;
    static constexpr const auto& scheme = sym7_scheme;
};

}  // namespace ttwv
