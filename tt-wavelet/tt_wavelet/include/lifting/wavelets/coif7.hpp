#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct coif7_tag {};

inline constexpr auto coif7_scheme = make_lifting_scheme(
    42,
    10,
    11,
    PredictStep<1>{ { -1.5309254880247831f }, 0 },
    UpdateStep<2>{ { 0.45784922323505595f, 0.63877120670120002f }, 0 },
    PredictStep<2>{ { -1.2840404341486487f, 1.0774677625456404f }, -1 },
    UpdateStep<2>{ { -0.70776629765619792f, 0.40897909419994144f }, 0 },
    PredictStep<2>{ { -1.2653660963876121f, 0.55236511357296625f }, -1 },
    UpdateStep<2>{ { -0.30722475432479851f, 0.229227219171107f }, 0 },
    PredictStep<2>{ { -0.44565802633790158f, 0.1371502385756096f }, -1 },
    UpdateStep<2>{ { -0.07773304465723993f, -0.065020790173030307f }, 0 },
    PredictStep<2>{ { 0.1150895546302337f, -0.34000513778577762f }, -1 },
    UpdateStep<2>{ { 0.18158784867245462f, -0.3975691574510079f }, 0 },
    PredictStep<2>{ { 0.60314430366138494f, -0.59261672816383948f }, -1 },
    UpdateStep<2>{ { 0.39389828970041252f, -0.62828549984170823f }, 0 },
    PredictStep<2>{ { 0.69480386899007784f, -1.427977101661996f }, -1 },
    UpdateStep<2>{ { 0.53638200339739983f, -1.9000536009322881f }, 0 },
    PredictStep<2>{ { 0.51378882368954537f, -2.9810533528907248f }, -1 },
    UpdateStep<2>{ { 0.33520940714813935f, -0.096896153862078102f }, 0 },
    PredictStep<2>{ { 10.231745165278197f, -24.621197730990311f }, -1 },
    UpdateStep<2>{ { 0.040554774052037355f, -0.13124784606127632f }, 0 },
    PredictStep<2>{ { 7.6180851756419514f, -34.48995780062608f }, -1 },
    UpdateStep<2>{ { 0.028993744830056505f, -0.20725857848535839f }, 0 },
    PredictStep<2>{ { 4.824890101135475f, -72.797781064550819f }, -1 },
    SwapStep{ {}, 0 },
    PredictStep<1>{ { 0.013736682420506643f }, 0 },
    ScaleEvenStep<1>{ { -0.00057975172435946281f }, 0 },
    ScaleOddStep<1>{ { 1724.8762840763386f }, 0 }
);

template <>
struct scheme_traits<coif7_tag> {
    using SchemeType = decltype(coif7_scheme);
    static constexpr const char* name = "coif7";
    static constexpr int id         = 29;
    static constexpr int tap_size   = 42;
    static constexpr int delay_even = 10;
    static constexpr int delay_odd  = 11;
    static constexpr int num_steps  = 25;
    static constexpr const auto& scheme = coif7_scheme;
};

}
