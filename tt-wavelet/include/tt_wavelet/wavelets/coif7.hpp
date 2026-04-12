#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif7_tag {};

inline constexpr auto coif7_scheme = make_lifting_scheme(
    42,
    10,
    11,
    PredictStep<1>{{-1.5309254880247831f}, {0xbfc3f55eu}, 0},
    UpdateStep<2>{{0.45784922323505595f, 0.63877120670120002f}, {0x3eea6b37u, 0x3f238683u}, 0},
    PredictStep<2>{{-1.2840404341486487f, 1.0774677625456404f}, {0xbfa45b70u, 0x3f89ea77u}, -1},
    UpdateStep<2>{{-0.70776629765619792f, 0.40897909419994144f}, {0xbf35302cu, 0x3ed165b5u}, 0},
    PredictStep<2>{{-1.2653660963876121f, 0.55236511357296625f}, {0xbfa1f784u, 0x3f0d67cdu}, -1},
    UpdateStep<2>{{-0.30722475432479851f, 0.229227219171107f}, {0xbe9d4c90u, 0x3e6aba8au}, 0},
    PredictStep<2>{{-0.44565802633790158f, 0.1371502385756096f}, {0xbee42d4au, 0x3e0c711du}, -1},
    UpdateStep<2>{{-0.07773304465723993f, -0.065020790173030307f}, {0xbd9f3281u, 0xbd85299fu}, 0},
    PredictStep<2>{{0.1150895546302337f, -0.34000513778577762f}, {0x3debb413u, 0xbeae1527u}, -1},
    UpdateStep<2>{{0.18158784867245462f, -0.3975691574510079f}, {0x3e39f22au, 0xbecb8e2fu}, 0},
    PredictStep<2>{{0.60314430366138494f, -0.59261672816383948f}, {0x3f1a67aau, 0xbf17b5bbu}, -1},
    UpdateStep<2>{{0.39389828970041252f, -0.62828549984170823f}, {0x3ec9ad09u, 0xbf20d752u}, 0},
    PredictStep<2>{{0.69480386899007784f, -1.427977101661996f}, {0x3f31deabu, 0xbfb6c7f4u}, -1},
    UpdateStep<2>{{0.53638200339739983f, -1.9000536009322881f}, {0x3f095055u, 0xbff334f5u}, 0},
    PredictStep<2>{{0.51378882368954537f, -2.9810533528907248f}, {0x3f0387aau, 0xc03ec994u}, -1},
    UpdateStep<2>{{0.33520940714813935f, -0.096896153862078102f}, {0x3eaba091u, 0xbdc6717eu}, 0},
    PredictStep<2>{{10.231745165278197f, -24.621197730990311f}, {0x4123b53au, 0xc1c4f837u}, -1},
    UpdateStep<2>{{0.040554774052037355f, -0.13124784606127632f}, {0x3d261cc3u, 0xbe0665d6u}, 0},
    PredictStep<2>{{7.6180851756419514f, -34.48995780062608f}, {0x40f3c75bu, 0xc209f5b7u}, -1},
    UpdateStep<2>{{0.028993744830056505f, -0.20725857848535839f}, {0x3ced844au, 0xbe543b98u}, 0},
    PredictStep<2>{{4.824890101135475f, -72.797781064550819f}, {0x409a6580u, 0xc2919877u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{0.013736682420506643f}, {0x3c610fd2u}, 0},
    ScaleEvenStep<1>{{-0.00057975172435946281f}, {0xba17fa7bu}, 0},
    ScaleOddStep<1>{{1724.8762840763386f}, {0x44d79c0bu}, 0});

template <>
struct scheme_traits<coif7_tag> {
    using SchemeType = decltype(coif7_scheme);
    static constexpr const char* name = "coif7";
    static constexpr int id = 29;
    static constexpr int tap_size = 42;
    static constexpr int delay_even = 10;
    static constexpr int delay_odd = 11;
    static constexpr int num_steps = 25;
    static constexpr const auto& scheme = coif7_scheme;
};

}  // namespace ttwv
