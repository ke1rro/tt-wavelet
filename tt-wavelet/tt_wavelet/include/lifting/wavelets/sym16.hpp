#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct sym16_tag {};

inline constexpr auto sym16_scheme = make_lifting_scheme(
    32,
    8,
    8,
    PredictStep<1>{ { -2.0009294470554999f }, -1 },
    UpdateStep<2>{ { 0.058472500207035427f, 0.399888480184173f }, 0 },
    PredictStep<2>{ { -4.3572196681229185f, -1.3485479258700559f }, -1 },
    UpdateStep<2>{ { -0.04005402474673804f, 0.10830846401803426f }, 0 },
    PredictStep<2>{ { 4.6335478667165146f, 2.7190584512670743f }, -1 },
    UpdateStep<2>{ { -0.099714707515253465f, -0.056533388630127732f }, 0 },
    PredictStep<2>{ { -2.7743658971170668f, 5.2625772735711092f }, -1 },
    UpdateStep<2>{ { -0.028412737447851014f, 0.084639222325947477f }, 0 },
    PredictStep<2>{ { 2.5491048339795443f, 1.1764583769156958f }, -1 },
    UpdateStep<2>{ { 0.17657864980656574f, -0.054796007033209831f }, 0 },
    PredictStep<2>{ { 1.79100979927172f, -3.5547790545018678f }, -1 },
    UpdateStep<2>{ { -0.088294643826915181f, -0.16734214058369287f }, 0 },
    PredictStep<2>{ { 7.6009815074568658f, 1.2057650469795707f }, -1 },
    UpdateStep<2>{ { 0.0098322600920704526f, -0.10862082606411104f }, 0 },
    PredictStep<2>{ { -19.868289538849499f, -3.7983543245783435f }, -1 },
    UpdateStep<2>{ { 0.014513169155310342f, 0.025916165867064363f }, 0 },
    PredictStep<1>{ { -17.208280790287219f }, 0 },
    ScaleEvenStep<1>{ { 5.4612968924947136f }, 0 },
    ScaleOddStep<1>{ { 0.18310669053247555f }, 0 }
);

template <>
struct scheme_traits<sym16_tag> {
    using SchemeType = decltype(sym16_scheme);
    static constexpr const char* name = "sym16";
    static constexpr int id         = 93;
    static constexpr int tap_size   = 32;
    static constexpr int delay_even = 8;
    static constexpr int delay_odd  = 8;
    static constexpr int num_steps  = 19;
    static constexpr const auto& scheme = sym16_scheme;
};

}
