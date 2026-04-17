#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif6_tag {};

inline constexpr auto coif6_scheme = make_lifting_scheme(
    36,
    9,
    9,
    PredictStep<1>{{0.62554479985985834f}, {0x3f2023b4u}, -1},
    UpdateStep<2>{{1.3935457870029002f, -0.44960969512914806f}, {0x3fb25fb5u, 0xbee6333eu}, 0},
    PredictStep<2>{{0.44637548016421819f, -0.56681152081890085f}, {0x3ee48b54u, 0xbf111a8fu}, -1},
    UpdateStep<2>{{0.73583763681121017f, -1.5677905218135482f}, {0x3f3c5fdbu, 0xbfc8ad5cu}, 0},
    PredictStep<2>{{0.18972576093264271f, -0.46112096308014477f}, {0x3e424778u, 0xbeec180cu}, -1},
    UpdateStep<2>{{0.20864262775597064f, -0.42158974367403979f}, {0x3e55a66au, 0xbed7da9cu}, 0},
    PredictStep<2>{{-0.055167758969699346f, -0.099929678277480366f}, {0xbd61f797u, 0xbdcca7eeu}, -1},
    UpdateStep<2>{{-0.55104550371069849f, 0.11687863171742839f}, {0xbf0d1151u, 0x3def5e10u}, 0},
    PredictStep<2>{{-0.2804334830702781f, 0.22878301396667969f}, {0xbe8f94fau, 0x3e6a4618u}, -1},
    UpdateStep<2>{{-0.90707089328894863f, 0.63521052576483172f}, {0xbf6835ccu, 0x3f229d28u}, 0},
    PredictStep<2>{{-0.76306700619218559f, 0.33790609500514579f}, {0xbf43585cu, 0x3ead0207u}, -1},
    UpdateStep<2>{{-3.1312829843149426f, 0.90774829884691077f}, {0xc04866f1u, 0x3f686231u}, 0},
    PredictStep<2>{{-1.3551565997225969f, 0.30787802012291765f}, {0xbfad75c5u, 0x3e9da230u}, -1},
    UpdateStep<2>{{-0.35697919228812319f, 0.7365046298328487f}, {0xbeb6c5fau, 0x3f3c8b91u}, 0},
    PredictStep<2>{{-7.5633541307469603f, 2.7785214036430297f}, {0xc0f206ffu, 0x4031d34bu}, -1},
    UpdateStep<2>{{-0.51628874322109053f, 0.13207045383743324f}, {0xbf042b80u, 0x3e073d7au}, 0},
    PredictStep<2>{{-11.996884471400939f, 1.936760524459453f}, {0xc13ff33du, 0x3ff7e7c5u}, -1},
    UpdateStep<2>{{1.196789285540364e-08f, 0.083354817396059877f}, {0x324d9b59u, 0x3daab5eeu}, 0},
    PredictStep<1>{{-0.91351148048637654f}, {0xbf69dbe3u}, 0},
    ScaleEvenStep<1>{{93.470395027784733f}, {0x42baf0d8u}, 0},
    ScaleOddStep<1>{{0.010698574663161988f}, {0x3c2f4913u}, 0});

template <>
struct scheme_traits<coif6_tag> {
    using SchemeType = decltype(coif6_scheme);
    static constexpr const char* name = "coif6";
    static constexpr int id = 28;
    static constexpr int tap_size = 36;
    static constexpr int delay_even = 9;
    static constexpr int delay_odd = 9;
    static constexpr int num_steps = 21;
    static constexpr const auto& scheme = coif6_scheme;
};

}  // namespace ttwv
