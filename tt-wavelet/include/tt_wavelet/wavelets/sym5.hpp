#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym5_tag {};

inline constexpr auto sym5_scheme = make_lifting_scheme(
    10,
    2,
    3,
    PredictStep<1>{{-1.0799918455175574f}, {0xbf8a3d2cu}, 0},
    UpdateStep<2>{{0.49852318422944608f, -0.1131044403344823f}, {0x3eff3e6eu, 0xbde7a34du}, 0},
    PredictStep<2>{{0.50075842492756506f, -2.4659476305429417f}, {0x3f0031b4u, 0xc01dd216u}, -1},
    UpdateStep<2>{{0.24040347972276505f, 0.055842424766306444f}, {0x3e762c54u, 0x3d64bb07u}, 0},
    PredictStep<2>{{-1.3043080355335845f, -3.3265774192975304f}, {0xbfa6f391u, 0xc054e6a5u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{0.10166231087600881f}, {0x3dd03454u}, 0},
    ScaleEvenStep<1>{{-0.37711544460644503f}, {0xbec11547u}, 0},
    ScaleOddStep<1>{{2.6517078902551785f}, {0x4029b595u}, 0});

template <>
struct scheme_traits<sym5_tag> {
    using SchemeType = decltype(sym5_scheme);
    static constexpr const char* name = "sym5";
    static constexpr int id = 101;
    static constexpr int tap_size = 10;
    static constexpr int delay_even = 2;
    static constexpr int delay_odd = 3;
    static constexpr int num_steps = 9;
    static constexpr const auto& scheme = sym5_scheme;
};

}  // namespace ttwv
