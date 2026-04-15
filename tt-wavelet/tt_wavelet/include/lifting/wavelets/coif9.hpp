#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct coif9_tag {};

inline constexpr auto coif9_scheme = make_lifting_scheme(
    54,
    13,
    14,
    PredictStep<1>{ { -1.4366106040844768f }, 0 },
    UpdateStep<2>{ { 0.46889064120753954f, 0.81142028360392893f }, 0 },
    PredictStep<2>{ { -1.0607755431465575f, 0.93774887822095332f }, -1 },
    UpdateStep<2>{ { -0.88347243010766474f, 0.65175183104612011f }, 0 },
    PredictStep<2>{ { -1.1114886573150888f, 0.57965200680285367f }, -1 },
    UpdateStep<2>{ { -0.71919287650226271f, 0.44627082153449682f }, 0 },
    PredictStep<2>{ { -0.4836830286124284f, 0.45845033104482513f }, -1 },
    UpdateStep<2>{ { -0.43247566942454774f, 0.20120710349135787f }, 0 },
    PredictStep<2>{ { -0.25252162716038695f, 0.08903848688946743f }, -1 },
    UpdateStep<2>{ { -0.074248606992623714f, -0.065268286100430978f }, 0 },
    PredictStep<2>{ { 0.078387114523126952f, -0.25428686081518975f }, -1 },
    UpdateStep<2>{ { 0.20189255627947644f, -0.28346054949742161f }, 0 },
    PredictStep<2>{ { 0.34006620546483884f, -0.68804953986294359f }, -1 },
    UpdateStep<2>{ { 0.44177608999051077f, -0.47654089750600465f }, 0 },
    PredictStep<2>{ { 0.70703891331911339f, -0.75907217164310203f }, -1 },
    UpdateStep<2>{ { 0.48659207720135034f, -0.84973363446972072f }, 0 },
    PredictStep<2>{ { 0.75443979600733968f, -1.3242610683722593f }, -1 },
    UpdateStep<2>{ { 0.63901652987728963f, -2.6068989568960834f }, 0 },
    PredictStep<2>{ { 0.37945432735765922f, 85.518783211205772f }, -1 },
    UpdateStep<2>{ { -0.011693335048546416f, -5.7003885007819229e-05f }, 0 },
    PredictStep<2>{ { 17385.401457238353f, -35393.932109073336f }, -1 },
    UpdateStep<2>{ { 2.8191901920135326e-05f, -7.2142985211668528e-05f }, 0 },
    PredictStep<2>{ { 13856.743318033416f, -44223.930841064743f }, -1 },
    UpdateStep<2>{ { 2.2611451797902927e-05f, -9.3945970273874506e-05f }, 0 },
    PredictStep<2>{ { 10644.395939496944f, -61611.368893381004f }, -1 },
    UpdateStep<2>{ { 1.623076976512752e-05f, -0.00014747866693256927f }, 0 },
    PredictStep<2>{ { 6780.6417030603779f, -129004.45767463192f }, -1 },
    SwapStep{ {}, 0 },
    PredictStep<1>{ { 7.7516701207353075e-06f }, 0 },
    ScaleEvenStep<1>{ { 3.2658481961910174e-06f }, 0 },
    ScaleOddStep<1>{ { -306199.16785057774f }, 0 }
);

template <>
struct scheme_traits<coif9_tag> {
    using SchemeType = decltype(coif9_scheme);
    static constexpr const char* name = "coif9";
    static constexpr int id         = 31;
    static constexpr int tap_size   = 54;
    static constexpr int delay_even = 13;
    static constexpr int delay_odd  = 14;
    static constexpr int num_steps  = 31;
    static constexpr const auto& scheme = coif9_scheme;
};

}
