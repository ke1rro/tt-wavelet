#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct db14_tag {};

inline constexpr auto db14_scheme = make_lifting_scheme(
    28,
    7,
    7,
    PredictStep<1>{ { -0.10364495383590744f }, -1 },
    UpdateStep<2>{ { -0.31579168979365757f, 0.10250198174277796f }, 0 },
    PredictStep<2>{ { -0.51635713848961062f, 0.292859556609951f }, -1 },
    UpdateStep<2>{ { -0.7149501373561552f, 0.46606944071422735f }, 0 },
    PredictStep<2>{ { -0.88042359434529782f, 0.5975490254358149f }, -1 },
    UpdateStep<2>{ { -1.0449386631580402f, 0.70216685569879966f }, 0 },
    PredictStep<2>{ { -1.1719453156027093f, 0.74831366208914185f }, -1 },
    UpdateStep<2>{ { -1.3193137837739022f, 0.7661705545830847f }, 0 },
    PredictStep<2>{ { -1.452967163309008f, 0.72997898401351935f }, -1 },
    UpdateStep<2>{ { -1.6577003671559671f, 0.68164936192426673f }, 0 },
    PredictStep<2>{ { -1.9024256302593117f, 0.60225971023257407f }, -1 },
    UpdateStep<2>{ { -2.327782537242578f, 0.52555852836543304f }, 0 },
    PredictStep<2>{ { -3.0178542689793333f, 0.42958980211413805f }, -1 },
    UpdateStep<2>{ { 3.7724440449202002e-09f, 0.3313612061611611f }, 0 },
    PredictStep<1>{ { -0.20280292564362215f }, 0 },
    ScaleEvenStep<1>{ { 85.627418716583264f }, 0 },
    ScaleOddStep<1>{ { 0.011678502224969353f }, 0 }
);

template <>
struct scheme_traits<db14_tag> {
    using SchemeType = decltype(db14_scheme);
    static constexpr const char* name = "db14";
    static constexpr int id         = 37;
    static constexpr int tap_size   = 28;
    static constexpr int delay_even = 7;
    static constexpr int delay_odd  = 7;
    static constexpr int num_steps  = 17;
    static constexpr const auto& scheme = db14_scheme;
};

}
