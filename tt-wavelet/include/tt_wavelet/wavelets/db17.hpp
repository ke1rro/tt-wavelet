#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db17_tag {};

inline constexpr auto db17_scheme = make_lifting_scheme(
    34,
    8,
    9,
    PredictStep<1>{{11.591272575258786f}, {0x413975dau}, 0},
    UpdateStep<2>{{-0.083014146423134469f, -0.0018626732144502735f}, {0xbdaa0352u, 0xbaf424f1u}, 0},
    PredictStep<2>{{55.458829906695961f, -54.112699150478278f}, {0x425dd5d8u, 0xc2587367u}, -1},
    UpdateStep<2>{{0.00091402075218396754f, -0.0044381721507382342f}, {0x3a6f9ae5u, 0xbb916e16u}, 0},
    PredictStep<2>{{32.578752983630658f, -119.38901272028775f}, {0x420250a5u, 0xc2eec72du}, -1},
    UpdateStep<2>{{0.0037757984646984996f, -0.0058073511516599785f}, {0x3b777363u, 0xbbbe4b98u}, 0},
    PredictStep<2>{{109.43605175989144f, -154.03474081555987f}, {0x42dadf42u, 0xc31a08e5u}, -1},
    UpdateStep<2>{{0.0050293032202743939f, -0.007321862419054451f}, {0x3ba4ccdau, 0xbbefec3cu}, 0},
    PredictStep<2>{{120.09467300221351f, -186.91615046295286f}, {0x42f03079u, 0xc33aea89u}, -1},
    UpdateStep<2>{{0.0050631279277342945f, -0.0087115169027566844f}, {0x3ba5e898u, 0xbc0ebac0u}, 0},
    PredictStep<2>{{112.63490594215979f, -222.40798444204492f}, {0x42e14512u, 0xc35e6872u}, -1},
    UpdateStep<2>{{0.0044742792171064113f, -0.010549751294634037f}, {0x3b929cf9u, 0xbc2cd8ddu}, 0},
    PredictStep<2>{{94.705348335314184f, -278.06476383904567f}, {0x42bd6923u, 0xc38b084au}, -1},
    UpdateStep<2>{{0.0035959162160983217f, -0.013806427913989731f}, {0x3b6ba977u, 0xbc62345bu}, 0},
    PredictStep<2>{{72.429527664470555f, -389.40354084819148f}, {0x4290dbebu, 0xc3c2b3a7u}, -1},
    UpdateStep<2>{{0.0025680294469405712f, -0.021777840915631241f}, {0x3b284c62u, 0xbcb26771u}, 0},
    PredictStep<2>{{45.91823406791692f, -818.86843324284507f}, {0x4237ac46u, 0xc44cb794u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{0.0012211973980109805f}, {0x3aa01096u}, 0},
    ScaleEvenStep<1>{{6.2919626324730671e-05f}, {0x3883f3b8u}, 0},
    ScaleOddStep<1>{{-15893.292099971486f}, {0xc678552bu}, 0});

template <>
struct scheme_traits<db17_tag> {
    using SchemeType = decltype(db17_scheme);
    static constexpr const char* name = "db17";
    static constexpr int id = 40;
    static constexpr int tap_size = 34;
    static constexpr int delay_even = 8;
    static constexpr int delay_odd = 9;
    static constexpr int num_steps = 21;
    static constexpr const auto& scheme = db17_scheme;
};

}  // namespace ttwv
