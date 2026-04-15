#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct db8_tag {};

inline constexpr auto db8_scheme = make_lifting_scheme(
    16,
    4,
    4,
    PredictStep<1>{ { -0.17392388386582808f }, -1 },
    UpdateStep<2>{ { -0.54524004214714239f, 0.16881724371813212f }, 0 },
    PredictStep<2>{ { -0.86429185090970628f, 0.43991331638521275f }, -1 },
    UpdateStep<2>{ { -1.1943611335045856f, 0.63536775889383368f }, 0 },
    PredictStep<2>{ { -1.4345897381272736f, 0.67941051920710782f }, -1 },
    UpdateStep<2>{ { -1.8158146468211536f, 0.66253699892061424f }, 0 },
    PredictStep<2>{ { -2.3235676940119872f, 0.54692259264909004f }, -1 },
    UpdateStep<2>{ { 1.8208920668345837e-05f, 0.43020729347257214f }, 0 },
    PredictStep<1>{ { -0.25595708753357671f }, 0 },
    ScaleEvenStep<1>{ { 10.888554594040391f }, 0 },
    ScaleOddStep<1>{ { 0.091839554218456854f }, 0 }
);

template <>
struct scheme_traits<db8_tag> {
    using SchemeType = decltype(db8_scheme);
    static constexpr const char* name = "db8";
    static constexpr int id         = 68;
    static constexpr int tap_size   = 16;
    static constexpr int delay_even = 4;
    static constexpr int delay_odd  = 4;
    static constexpr int num_steps  = 11;
    static constexpr const auto& scheme = db8_scheme;
};

}
