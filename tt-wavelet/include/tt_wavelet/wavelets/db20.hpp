#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db20_tag {};

inline constexpr auto db20_scheme = make_lifting_scheme(
    40,
    10,
    10,
    PredictStep<1>{{-0.40242426897619299f}, {0xbece0a8eu}, -1},
    UpdateStep<2>{{-0.58134232363791838f, 0.071661818235120481f}, {0xbf14d2dau, 0x3d92c36eu}, 0},
    PredictStep<2>{{-0.71649707466243673f, 0.20232057817794602f}, {0xbf376c5au, 0x3e4f2d20u}, -1},
    UpdateStep<2>{{-0.90415882817976656f, 0.32524068476166923f}, {0xbf6776f4u, 0x3ea685f2u}, 0},
    PredictStep<2>{{2.1054704163132851f, 0.41452599271045348f}, {0x4006c007u, 0x3ed43cc0u}, -1},
    UpdateStep<2>{{0.0020945204956796877f, -0.8639213768280305f}, {0x3b094439u, 0xbf5d29f4u}, 0},
    PredictStep<2>{{1.8334448613375993f, 0.54293219658437519f}, {0x3feaae52u, 0x3f0afd9bu}, -1},
    UpdateStep<2>{{-1.2215909361925203f, 0.89163491037932607f}, {0xbf9c5d18u, 0x3f64422fu}, 0},
    PredictStep<2>{{-0.88953978621649876f, 0.64372984573370573f}, {0xbf63b8e1u, 0x3f24cb7bu}, -1},
    UpdateStep<2>{{-1.4116147844823703f, 0.97206717710239066f}, {0xbfb4afcbu, 0x3f78d965u}, 0},
    PredictStep<2>{{-1.0304782254758516f, 0.65938297443568061f}, {0xbf83e6b6u, 0x3f28cd53u}, -1},
    UpdateStep<2>{{-1.6271262084766629f, 0.94175253283068894f}, {0xbfd045acu, 0x3f7116b2u}, 0},
    PredictStep<2>{{-1.1963176788148302f, 0.60837600269598702f}, {0xbf9920f0u, 0x3f1bbe88u}, -1},
    UpdateStep<2>{{-1.9219488205317936f, 0.83369953238444205f}, {0xbff6026bu, 0x3f556d55u}, 0},
    PredictStep<2>{{-1.4488225990913204f, 0.52004713333128327f}, {0xbfb97305u, 0x3f0521cfu}, -1},
    UpdateStep<2>{{-2.4038249349983154f, 0.69017149818641477f}, {0xc019d845u, 0x3f30af14u}, 0},
    PredictStep<2>{{-1.8922446000841526f, 0.41600147938905774f}, {0xbff23512u, 0x3ed4fe25u}, -1},
    UpdateStep<2>{{-3.3520816949156145f, 0.5284727699597348f}, {0xc0568882u, 0x3f0749feu}, 0},
    PredictStep<2>{{-2.9684058156985804f, 0.29832208301631918f}, {0xc03dfa5cu, 0x3e98bdacu}, -1},
    UpdateStep<2>{{1.0358333632182811e-12f, 0.3368811618167204f}, {0x2b91c7d4u, 0x3eac7bb0u}, 0},
    PredictStep<1>{{-0.14271779560940895f}, {0xbe12249du}, 0},
    ScaleEvenStep<1>{{-609.25213321490037f}, {0xc4185023u}, 0},
    ScaleOddStep<1>{{-0.0016413565837237237f}, {0xbad722cau}, 0});

template <>
struct scheme_traits<db20_tag> {
    using SchemeType = decltype(db20_scheme);
    static constexpr const char* name = "db20";
    static constexpr int id = 44;
    static constexpr int tap_size = 40;
    static constexpr int delay_even = 10;
    static constexpr int delay_odd = 10;
    static constexpr int num_steps = 23;
    static constexpr const auto& scheme = db20_scheme;
};

}  // namespace ttwv
