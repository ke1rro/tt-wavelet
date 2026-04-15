#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct db16_tag {};

inline constexpr auto db16_scheme = make_lifting_scheme(
    32,
    8,
    8,
    PredictStep<1>{ { -0.10916035891478494f }, -1 },
    UpdateStep<2>{ { -0.24203260243188657f, 0.090459338338146941f }, 0 },
    PredictStep<2>{ { -0.4394349559418852f, 0.26243921884715576f }, -1 },
    UpdateStep<2>{ { -0.6287392843579247f, 0.42024641363041459f }, 0 },
    PredictStep<2>{ { -0.78649028191140846f, 0.55264151769623981f }, -1 },
    UpdateStep<2>{ { -0.93257906100996257f, 0.66133723099125052f }, 0 },
    PredictStep<2>{ { -1.0558790460779603f, 0.73253435523355903f }, -1 },
    UpdateStep<2>{ { -1.1759399460844986f, 0.77423902354460805f }, 0 },
    PredictStep<2>{ { -1.2842988771773365f, 0.77535174775982463f }, -1 },
    UpdateStep<2>{ { -1.4134168734148138f, 0.75210771728314096f }, 0 },
    PredictStep<2>{ { -1.5596111052812498f, 0.70050941899967956f }, -1 },
    UpdateStep<2>{ { -1.7671078953500539f, 0.63989623522757699f }, 0 },
    PredictStep<2>{ { -2.0413686311738064f, 0.56574693616495175f }, -1 },
    UpdateStep<2>{ { -2.4775590159603218f, 0.48985748742420715f }, 0 },
    PredictStep<2>{ { -3.2262169174059352f, 0.40362275945943954f }, -1 },
    UpdateStep<2>{ { 2.2843195682724686e-10f, 0.3099605554074743f }, 0 },
    PredictStep<1>{ { -0.19149919576800786f }, 0 },
    ScaleEvenStep<1>{ { 170.15805926186462f }, 0 },
    ScaleOddStep<1>{ { 0.0058768888428672702f }, 0 }
);

template <>
struct scheme_traits<db16_tag> {
    using SchemeType = decltype(db16_scheme);
    static constexpr const char* name = "db16";
    static constexpr int id         = 39;
    static constexpr int tap_size   = 32;
    static constexpr int delay_even = 8;
    static constexpr int delay_odd  = 8;
    static constexpr int num_steps  = 19;
    static constexpr const auto& scheme = db16_scheme;
};

}
