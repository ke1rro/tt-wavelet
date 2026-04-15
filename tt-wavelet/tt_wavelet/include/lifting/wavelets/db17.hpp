#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct db17_tag {};

inline constexpr auto db17_scheme = make_lifting_scheme(
    34,
    8,
    9,
    PredictStep<1>{ { 11.591272575258786f }, 0 },
    UpdateStep<2>{ { -0.083014146423134469f, -0.0018626732144502735f }, 0 },
    PredictStep<2>{ { 55.458829906695961f, -54.112699150478278f }, -1 },
    UpdateStep<2>{ { 0.00091402075218396754f, -0.0044381721507382342f }, 0 },
    PredictStep<2>{ { 32.578752983630658f, -119.38901272028775f }, -1 },
    UpdateStep<2>{ { 0.0037757984646984996f, -0.0058073511516599785f }, 0 },
    PredictStep<2>{ { 109.43605175989144f, -154.03474081555987f }, -1 },
    UpdateStep<2>{ { 0.0050293032202743939f, -0.007321862419054451f }, 0 },
    PredictStep<2>{ { 120.09467300221351f, -186.91615046295286f }, -1 },
    UpdateStep<2>{ { 0.0050631279277342945f, -0.0087115169027566844f }, 0 },
    PredictStep<2>{ { 112.63490594215979f, -222.40798444204492f }, -1 },
    UpdateStep<2>{ { 0.0044742792171064113f, -0.010549751294634037f }, 0 },
    PredictStep<2>{ { 94.705348335314184f, -278.06476383904567f }, -1 },
    UpdateStep<2>{ { 0.0035959162160983217f, -0.013806427913989731f }, 0 },
    PredictStep<2>{ { 72.429527664470555f, -389.40354084819148f }, -1 },
    UpdateStep<2>{ { 0.0025680294469405712f, -0.021777840915631241f }, 0 },
    PredictStep<2>{ { 45.91823406791692f, -818.86843324284507f }, -1 },
    SwapStep{ {}, 0 },
    PredictStep<1>{ { 0.0012211973980109805f }, 0 },
    ScaleEvenStep<1>{ { 6.2919626324730671e-05f }, 0 },
    ScaleOddStep<1>{ { -15893.292099971486f }, 0 }
);

template <>
struct scheme_traits<db17_tag> {
    using SchemeType = decltype(db17_scheme);
    static constexpr const char* name = "db17";
    static constexpr int id         = 40;
    static constexpr int tap_size   = 34;
    static constexpr int delay_even = 8;
    static constexpr int delay_odd  = 9;
    static constexpr int num_steps  = 21;
    static constexpr const auto& scheme = db17_scheme;
};

}
