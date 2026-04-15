#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct db15_tag {};

inline constexpr auto db15_scheme = make_lifting_scheme(
    30,
    7,
    8,
    PredictStep<1>{ { 10.299220023221395f }, 0 },
    UpdateStep<2>{ { -0.096207914892059751f, -0.0027902560084822905f }, 0 },
    PredictStep<2>{ { 29.799679237629821f, -51.246396534364308f }, -1 },
    UpdateStep<2>{ { 0.0042002525352739387f, -0.0063332439408615932f }, 0 },
    PredictStep<2>{ { 60.856132221360696f, -87.910339011201444f }, -1 },
    UpdateStep<2>{ { 0.0064496026976610579f, -0.0093125210098273618f }, 0 },
    PredictStep<2>{ { 78.593173119408732f, -117.46060349801127f }, -1 },
    UpdateStep<2>{ { 0.0073140343067391086f, -0.011731706803465731f }, 0 },
    PredictStep<2>{ { 80.131421397455341f, -143.7790984837344f }, -1 },
    UpdateStep<2>{ { 0.0068200858666384034f, -0.014343867890226418f }, 0 },
    PredictStep<2>{ { 69.405549796047438f, -179.95987819028798f }, -1 },
    UpdateStep<2>{ { 0.0055530971256945788f, -0.018774844801404077f }, 0 },
    PredictStep<2>{ { 53.259655246988125f, -252.54350191232919f }, -1 },
    UpdateStep<2>{ { 0.003959703584031189f, -0.029728866032233219f }, 0 },
    PredictStep<2>{ { 33.637339310593667f, -533.58492633378762f }, -1 },
    SwapStep{ {}, 0 },
    PredictStep<1>{ { 0.001874115910073172f }, 0 },
    ScaleEvenStep<1>{ { 0.00015914371486505261f }, 0 },
    ScaleOddStep<1>{ { -6283.6286110825004f }, 0 }
);

template <>
struct scheme_traits<db15_tag> {
    using SchemeType = decltype(db15_scheme);
    static constexpr const char* name = "db15";
    static constexpr int id         = 38;
    static constexpr int tap_size   = 30;
    static constexpr int delay_even = 7;
    static constexpr int delay_odd  = 8;
    static constexpr int num_steps  = 19;
    static constexpr const auto& scheme = db15_scheme;
};

}
