#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct dmey_tag {};

inline constexpr auto dmey_scheme = make_lifting_scheme(
    62,
    15,
    16,
    PredictStep<1>{ { 1.5960022407278813f }, 0 },
    UpdateStep<2>{ { -1.7779840920355983f, -1.1514185556714798f }, 0 },
    PredictStep<2>{ { 28.028017206836175f, 28.028017206836175f }, -1 },
    UpdateStep<2>{ { 0.0025587955585354505f, 0.0025587955585354505f }, 0 },
    PredictStep<2>{ { -13.323849401441363f, -13.323849401441363f }, -1 },
    UpdateStep<2>{ { -0.013720172337578674f, -0.013720172337578674f }, 0 },
    PredictStep<2>{ { -20.154757399396647f, -20.154757399396647f }, -1 },
    UpdateStep<2>{ { -0.012249423432055009f, -0.012249423432055009f }, 0 },
    PredictStep<2>{ { 9.3388423267371028f, 9.3388423267371028f }, -1 },
    UpdateStep<2>{ { -0.49717338132123129f, -0.49717338132123129f }, 0 },
    PredictStep<2>{ { 0.021631934844595687f, 0.021631934844595687f }, -1 },
    UpdateStep<2>{ { 0.52332393369008834f, 0.52332393369008834f }, 0 },
    PredictStep<2>{ { -5.0428506182368817f, -5.0428506182368817f }, -1 },
    UpdateStep<2>{ { 0.5985236097779153f, 0.5985236097779153f }, 0 },
    PredictStep<2>{ { 4.4846260350699954f, 4.4846260350699954f }, -1 },
    UpdateStep<2>{ { -0.4181979148920521f, -0.4181979148920521f }, 0 },
    PredictStep<2>{ { -0.79681914689423172f, -0.79681914689423172f }, -1 },
    UpdateStep<2>{ { 0.17301790911473672f, 0.17301790911473672f }, 0 },
    PredictStep<2>{ { -9.7678222892320417f, -9.7678222892320417f }, -1 },
    UpdateStep<2>{ { -0.036743885636883633f, -0.036743885636883633f }, 0 },
    PredictStep<2>{ { 5.5614712260772352f, 5.5614712260772352f }, -1 },
    UpdateStep<2>{ { 0.011537507298938281f, 0.011537507298938281f }, 0 },
    PredictStep<2>{ { -0.77790957473555711f, -0.77790957473555711f }, -1 },
    UpdateStep<2>{ { -0.015822062244821234f, -0.015822062244821234f }, 0 },
    PredictStep<2>{ { -4.223219973787316f, -4.223219973787316f }, -1 },
    UpdateStep<2>{ { 0.043542905481573906f, 0.043542905481573906f }, 0 },
    PredictStep<2>{ { 9.0461651504283278f, 9.0461651504283278f }, -1 },
    UpdateStep<2>{ { -0.14779238801168995f, -0.14779238801168995f }, 0 },
    PredictStep<2>{ { 1.0444549601312074f, 1.0444549601312074f }, -1 },
    UpdateStep<2>{ { 0.30039790834292662f, 0.30039790834292662f }, 0 },
    PredictStep<2>{ { 0.00030236436693688139f, 0.00030236436693688139f }, -1 },
    ScaleEvenStep<1>{ { -1.5970210912462088f }, 0 },
    ScaleOddStep<1>{ { -0.62616580675191125f }, 0 }
);

template <>
struct scheme_traits<dmey_tag> {
    using SchemeType = decltype(dmey_scheme);
    static constexpr const char* name = "dmey";
    static constexpr int id         = 70;
    static constexpr int tap_size   = 62;
    static constexpr int delay_even = 15;
    static constexpr int delay_odd  = 16;
    static constexpr int num_steps  = 33;
    static constexpr const auto& scheme = dmey_scheme;
};

}
