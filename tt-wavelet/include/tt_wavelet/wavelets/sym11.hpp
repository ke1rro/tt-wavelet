#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym11_tag {};

inline constexpr auto sym11_scheme = make_lifting_scheme(
    22,
    5,
    6,
    PredictStep<1>{{0.22592135469188568f}, {0x3e6757edu}, 0},
    UpdateStep<2>{{-0.21495019804878834f, -0.87380362928334288f}, {0xbe5c1be8u, 0xbf5fb198u}, 0},
    PredictStep<2>{{0.52362219797589771f, -0.035398892130974344f}, {0x3f060c1bu, 0xbd10fe6eu}, -1},
    UpdateStep<2>{{0.1084798803030794f, 1.1574012046641873f}, {0x3dde2ab3u, 0x3f9425b9u}, 0},
    PredictStep<2>{{-0.2635063269237144f, 0.78945064263178377f}, {0xbe86ea4du, 0x3f4a1970u}, -1},
    UpdateStep<2>{{-1.0102156832656921f, 9.8051274942310513f}, {0xbf814ebfu, 0x411ce1cdu}, 0},
    PredictStep<2>{{-0.10171332769006111f, -0.0051047191673875307f}, {0xbdd04f14u, 0xbba7457du}, -1},
    UpdateStep<2>{{94.635991832273191f, -11.417827964355032f}, {0x42bd45a1u, 0xc136af6cu}, 0},
    PredictStep<2>{{0.0011754817286835392f, 0.004876946882548105f}, {0x3a9a129fu, 0x3b9fceccu}, -1},
    UpdateStep<2>{{-38.905306629334888f, 86.926537139868799f}, {0xc21b9f09u, 0x42adda63u}, 0},
    PredictStep<2>{{-0.0062001854591211781f, -0.016338502515647838f}, {0xbbcb2aedu, 0xbc85d853u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{54.492355593458441f}, {0x4259f82cu}, 0},
    ScaleEvenStep<1>{{-4.3732996945296678f}, {0xc08bf212u}, 0},
    ScaleOddStep<1>{{0.22866029539453878f}, {0x3e6a25edu}, 0});

template <>
struct scheme_traits<sym11_tag> {
    using SchemeType = decltype(sym11_scheme);
    static constexpr const char* name = "sym11";
    static constexpr int id = 88;
    static constexpr int tap_size = 22;
    static constexpr int delay_even = 5;
    static constexpr int delay_odd = 6;
    static constexpr int num_steps = 15;
    static constexpr const auto& scheme = sym11_scheme;
};

}  // namespace ttwv
