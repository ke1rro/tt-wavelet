#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db14_tag {};

inline constexpr auto db14_scheme = make_lifting_scheme(
    28,
    7,
    7,
    PredictStep<1>{{-0.10364495383590744f}, {0xbdd443ceu}, -1},
    UpdateStep<2>{{-0.31579168979365757f, 0.10250198174277796f}, {0xbea1af73u, 0x3dd1ec8fu}, 0},
    PredictStep<2>{{-0.51635713848961062f, 0.292859556609951f}, {0xbf042ffbu, 0x3e95f1b0u}, -1},
    UpdateStep<2>{{-0.7149501373561552f, 0.46606944071422735f}, {0xbf3706f9u, 0x3eeea0a7u}, 0},
    PredictStep<2>{{-0.88042359434529782f, 0.5975490254358149f}, {0xbf616371u, 0x3f18f8f9u}, -1},
    UpdateStep<2>{{-1.0449386631580402f, 0.70216685569879966f}, {0xbf85c08du, 0x3f33c135u}, 0},
    PredictStep<2>{{-1.1719453156027093f, 0.74831366208914185f}, {0xbf96024eu, 0x3f3f917cu}, -1},
    UpdateStep<2>{{-1.3193137837739022f, 0.7661705545830847f}, {0xbfa8df46u, 0x3f4423c1u}, 0},
    PredictStep<2>{{-1.452967163309008f, 0.72997898401351935f}, {0xbfb9fad4u, 0x3f3adfe7u}, -1},
    UpdateStep<2>{{-1.6577003671559671f, 0.68164936192426673f}, {0xbfd42f87u, 0x3f2e8093u}, 0},
    PredictStep<2>{{-1.9024256302593117f, 0.60225971023257407f}, {0xbff382afu, 0x3f1a2db1u}, -1},
    UpdateStep<2>{{-2.327782537242578f, 0.52555852836543304f}, {0xc014fa64u, 0x3f068b01u}, 0},
    PredictStep<2>{{-3.0178542689793333f, 0.42958980211413805f}, {0xc0412486u, 0x3edbf332u}, -1},
    UpdateStep<2>{{3.7724440449202002e-09f, 0.3313612061611611f}, {0x31819ec5u, 0x3ea9a82du}, 0},
    PredictStep<1>{{-0.20280292564362215f}, {0xbe4fab92u}, 0},
    ScaleEvenStep<1>{{85.627418716583264f}, {0x42ab413du}, 0},
    ScaleOddStep<1>{{0.011678502224969353f}, {0x3c3f5730u}, 0});

template <>
struct scheme_traits<db14_tag> {
    using SchemeType = decltype(db14_scheme);
    static constexpr const char* name = "db14";
    static constexpr int id = 37;
    static constexpr int tap_size = 28;
    static constexpr int delay_even = 7;
    static constexpr int delay_odd = 7;
    static constexpr int num_steps = 17;
    static constexpr const auto& scheme = db14_scheme;
};

}  // namespace ttwv
