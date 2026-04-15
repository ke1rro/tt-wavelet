#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct sym10_tag {};

inline constexpr auto sym10_scheme = make_lifting_scheme(
    20,
    5,
    5,
    PredictStep<1>{ { 8.0533127778739217f }, -1 },
    UpdateStep<2>{ { 0.030264604896245172f, -0.12228698257837521f }, 0 },
    PredictStep<2>{ { 8.7593620786409812f, -26.396907896909436f }, -1 },
    UpdateStep<2>{ { 0.017810178317534842f, -0.034740951493743698f }, 0 },
    PredictStep<2>{ { 1.4814231376821672f, -5.7892552363227479f }, -1 },
    UpdateStep<2>{ { -0.010952958767575448f, -0.0050434460428580054f }, 0 },
    PredictStep<2>{ { -9.3742758447654531f, 3.1303191173128559f }, -1 },
    UpdateStep<2>{ { -0.066296990860538349f, 0.025762364949882807f }, 0 },
    PredictStep<2>{ { -13.123765639102318f, 10.231349589412808f }, -1 },
    UpdateStep<2>{ { 0.053245246606484074f, 0.05914837648354393f }, 0 },
    PredictStep<1>{ { -13.853349293041154f }, 0 },
    ScaleEvenStep<1>{ { 2.8744125379022876f }, 0 },
    ScaleOddStep<1>{ { 0.34789717440134332f }, 0 }
);

template <>
struct scheme_traits<sym10_tag> {
    using SchemeType = decltype(sym10_scheme);
    static constexpr const char* name = "sym10";
    static constexpr int id         = 87;
    static constexpr int tap_size   = 20;
    static constexpr int delay_even = 5;
    static constexpr int delay_odd  = 5;
    static constexpr int num_steps  = 13;
    static constexpr const auto& scheme = sym10_scheme;
};

}
