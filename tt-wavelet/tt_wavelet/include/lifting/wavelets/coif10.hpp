#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct coif10_tag {};

inline constexpr auto coif10_scheme = make_lifting_scheme(
    60,
    15,
    15,
    PredictStep<1>{ { 0.7132767894922879f }, -1 },
    UpdateStep<2>{ { 1.7540798088726735f, -0.47275577508579603f }, 0 },
    PredictStep<2>{ { 0.44496621786329815f, -0.49887195595502387f }, -1 },
    UpdateStep<2>{ { 1.5134785487872304f, -1.9054055413375817f }, 0 },
    PredictStep<2>{ { 0.2934266713265814f, -0.51440678456355393f }, -1 },
    UpdateStep<2>{ { 1.0623615323309017f, -1.8183624845353186f }, 0 },
    PredictStep<2>{ { 0.25018780197300433f, -0.26432420115490501f }, -1 },
    UpdateStep<2>{ { 0.72837736425403121f, -1.0358250848794248f }, 0 },
    PredictStep<2>{ { 0.093939894807340718f, -0.20245131910996594f }, -1 },
    UpdateStep<2>{ { 0.16804485684217962f, -0.38220110016211339f }, 0 },
    PredictStep<2>{ { -0.028619420738579787f, -0.042535118830928573f }, -1 },
    UpdateStep<2>{ { -0.46082689565462032f, 0.11351170630640733f }, 0 },
    PredictStep<2>{ { -0.14811445887624602f, 0.11062348113415291f }, -1 },
    UpdateStep<2>{ { -0.92350019535611316f, 0.5930398581325812f }, 0 },
    PredictStep<2>{ { -0.33956271160327395f, 0.20499058542599516f }, -1 },
    UpdateStep<2>{ { -1.0918515695887512f, 1.1500735300282554f }, 0 },
    PredictStep<2>{ { -0.39482057852409669f, 0.3352944693644917f }, -1 },
    UpdateStep<2>{ { -1.9721713063898181f, 1.1262925555643799f }, 0 },
    PredictStep<2>{ { -0.59322357113186308f, 0.36030646050964804f }, -1 },
    UpdateStep<2>{ { -6.6445300520970578f, 1.4655138831652108f }, 0 },
    PredictStep<2>{ { 1.0845418748876099f, 0.14940770366000031f }, -1 },
    UpdateStep<2>{ { -0.11398380696752623f, -0.92192042573002386f }, 0 },
    PredictStep<2>{ { -16.660557648355947f, 8.6942810977377594f }, -1 },
    UpdateStep<2>{ { -0.14152526973669308f, 0.05987404127341639f }, 0 },
    PredictStep<2>{ { -20.175502878523176f, 7.0627521267771716f }, -1 },
    UpdateStep<2>{ { -0.17622002603371745f, 0.049562373110663256f }, 0 },
    PredictStep<2>{ { -26.247153128801219f, 5.6746996071417009f }, -1 },
    UpdateStep<2>{ { -0.24500075235991398f, 0.038099362421490646f }, 0 },
    PredictStep<2>{ { -41.08186734584033f, 4.0816200992088216f }, -1 },
    UpdateStep<2>{ { 5.5445185208960055e-14f, 0.024341639379092464f }, 0 },
    PredictStep<1>{ { -1.9560136108873891f }, 0 },
    ScaleEvenStep<1>{ { -2437.1205319567825f }, 0 },
    ScaleOddStep<1>{ { -0.00041032028858953995f }, 0 }
);

template <>
struct scheme_traits<coif10_tag> {
    using SchemeType = decltype(coif10_scheme);
    static constexpr const char* name = "coif10";
    static constexpr int id         = 16;
    static constexpr int tap_size   = 60;
    static constexpr int delay_even = 15;
    static constexpr int delay_odd  = 15;
    static constexpr int num_steps  = 33;
    static constexpr const auto& scheme = coif10_scheme;
};

}
