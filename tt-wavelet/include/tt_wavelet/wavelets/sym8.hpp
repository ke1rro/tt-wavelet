#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym8_tag {};

inline constexpr auto sym8_scheme = make_lifting_scheme(
    16,
    4,
    4,
    PredictStep<1>{{6.2390965318689222f}, {0x40c7a6aeu}, -1},
    UpdateStep<2>{{0.034380149908114513f, -0.1562652322345541f}, {0x3d0cd233u, 0xbe2003feu}, 0},
    PredictStep<2>{{5.1308125676808904f, -19.001870937128704f}, {0x40a42f9eu, 0xc19803d5u}, -1},
    UpdateStep<2>{{0.0065800143696067084f, -0.023540908951942469f}, {0x3bd79d29u, 0xbcc0d8ddu}, 0},
    PredictStep<2>{{-3.576656251962949f, -1.6138298088457212f}, {0xc064e7f0u, 0xbfce91fau}, -1},
    UpdateStep<2>{{-0.062041284814645728f, 0.01400137396605277f}, {0xbd7e1f01u, 0x3c656605u}, 0},
    PredictStep<2>{{-12.969466315244516f, 8.1981288892981787f}, {0xc14f82efu, 0x41032b89u}, -1},
    UpdateStep<2>{{0.061857002592835567f, 0.055630084491937883f}, {0x3d7d5dc5u, 0x3d63dc5fu}, 0},
    PredictStep<1>{{-12.319917462680916f}, {0xc1451e62u}, 0},
    ScaleEvenStep<1>{{2.6237085339634949f}, {0x4027ead7u}, 0},
    ScaleOddStep<1>{{0.38113989685026256f}, {0x3ec324c5u}, 0});

template <>
struct scheme_traits<sym8_tag> {
    using SchemeType = decltype(sym8_scheme);
    static constexpr const char* name = "sym8";
    static constexpr int id = 104;
    static constexpr int tap_size = 16;
    static constexpr int delay_even = 4;
    static constexpr int delay_odd = 4;
    static constexpr int num_steps = 11;
    static constexpr const auto& scheme = sym8_scheme;
};

}  // namespace ttwv
