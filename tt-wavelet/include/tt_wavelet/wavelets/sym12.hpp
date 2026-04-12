#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym12_tag {};

inline constexpr auto sym12_scheme = make_lifting_scheme(
    24,
    6,
    6,
    PredictStep<1>{{-9.8615381246938369f}, {0xc11dc8dcu}, -1},
    UpdateStep<2>{{-0.02656309860199179f, 0.10037195654646704f}, {0xbcd99adbu, 0x3dcd8fd0u}, 0},
    PredictStep<2>{{-12.285473170833361f, 32.826677787300028f}, {0xc144914cu, 0x42034e85u}, -1},
    UpdateStep<2>{{-0.019893913275891198f, 0.039740537778558412f}, {0xbca2f88fu, 0x3d22c6f9u}, 0},
    PredictStep<2>{{-7.531485335885681f, 9.698595669232887f}, {0xc0f101eeu, 0x411b2d73u}, -1},
    UpdateStep<2>{{-0.0023808369652606703f, 0.016730081682320392f}, {0xbb1c07d1u, 0x3c890d86u}, 0},
    PredictStep<2>{{4.8709180750523107f, 1.2227448148096733f}, {0x409bde90u, 0x3f9c82e7u}, -1},
    UpdateStep<2>{{0.01547974202768369f, -0.0090907823919844123f}, {0x3c7d9ebeu, 0xbc14f181u}, 0},
    PredictStep<2>{{19.272267925324595f, -7.6506692509285816f}, {0x419a2d9bu, 0xc0f4d248u}, -1},
    UpdateStep<2>{{0.046425277091769378f, -0.023877671789650931f}, {0x3d3e286eu, 0xbcc39b1bu}, 0},
    PredictStep<2>{{19.032204047075222f, -16.438797857260905f}, {0x419841f4u, 0xc18382a8u}, -1},
    UpdateStep<2>{{0.012836072335339682f, -0.042665234491412019f}, {0x3c524e64u, 0xbd2ec1beu}, 0},
    PredictStep<1>{{-21.897776434122395f}, {0xc1af2ea5u}, 0},
    ScaleEvenStep<1>{{5.9178218770562241f}, {0x40bd5eccu}, 0},
    ScaleOddStep<1>{{0.16898109148520746f}, {0x3e2d0961u}, 0});

template <>
struct scheme_traits<sym12_tag> {
    using SchemeType = decltype(sym12_scheme);
    static constexpr const char* name = "sym12";
    static constexpr int id = 89;
    static constexpr int tap_size = 24;
    static constexpr int delay_even = 6;
    static constexpr int delay_odd = 6;
    static constexpr int num_steps = 15;
    static constexpr const auto& scheme = sym12_scheme;
};

}  // namespace ttwv
