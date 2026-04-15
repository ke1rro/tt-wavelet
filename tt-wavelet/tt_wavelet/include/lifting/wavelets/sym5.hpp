#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct sym5_tag {};

inline constexpr auto sym5_scheme = make_lifting_scheme(
    10,
    2,
    3,
    PredictStep<1>{ { -1.0799918455175574f }, 0 },
    UpdateStep<2>{ { 0.49852318422944608f, -0.1131044403344823f }, 0 },
    PredictStep<2>{ { 0.50075842492756506f, -2.4659476305429417f }, -1 },
    UpdateStep<2>{ { 0.24040347972276505f, 0.055842424766306444f }, 0 },
    PredictStep<2>{ { -1.3043080355335845f, -3.3265774192975304f }, -1 },
    SwapStep{ {}, 0 },
    PredictStep<1>{ { 0.10166231087600881f }, 0 },
    ScaleEvenStep<1>{ { -0.37711544460644503f }, 0 },
    ScaleOddStep<1>{ { 2.6517078902551785f }, 0 }
);

template <>
struct scheme_traits<sym5_tag> {
    using SchemeType = decltype(sym5_scheme);
    static constexpr const char* name = "sym5";
    static constexpr int id         = 101;
    static constexpr int tap_size   = 10;
    static constexpr int delay_even = 2;
    static constexpr int delay_odd  = 3;
    static constexpr int num_steps  = 9;
    static constexpr const auto& scheme = sym5_scheme;
};

}
