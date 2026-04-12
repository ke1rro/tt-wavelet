#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif9_tag {};

inline constexpr auto coif9_scheme = make_lifting_scheme(
    54,
    13,
    14,
    PredictStep<1>{{-1.4366106040844768f}, {0xbfb7e2dbu}, 0},
    UpdateStep<2>{{0.46889064120753954f, 0.81142028360392893f}, {0x3ef0126fu, 0x3f4fb93du}, 0},
    PredictStep<2>{{-1.0607755431465575f, 0.93774887822095332f}, {0xbf87c77eu, 0x3f70104fu}, -1},
    UpdateStep<2>{{-0.88347243010766474f, 0.65175183104612011f}, {0xbf622b40u, 0x3f26d935u}, 0},
    PredictStep<2>{{-1.1114886573150888f, 0.57965200680285367f}, {0xbf8e4543u, 0x3f146413u}, -1},
    UpdateStep<2>{{-0.71919287650226271f, 0.44627082153449682f}, {0xbf381d06u, 0x3ee47d9cu}, 0},
    PredictStep<2>{{-0.4836830286124284f, 0.45845033104482513f}, {0xbef7a54du, 0x3eeaba00u}, -1},
    UpdateStep<2>{{-0.43247566942454774f, 0.20120710349135787f}, {0xbedd6d73u, 0x3e4e093cu}, 0},
    PredictStep<2>{{-0.25252162716038695f, 0.08903848688946743f}, {0xbe814a84u, 0x3db659cfu}, -1},
    UpdateStep<2>{{-0.074248606992623714f, -0.065268286100430978f}, {0xbd980fa7u, 0xbd85ab61u}, 0},
    PredictStep<2>{{0.078387114523126952f, -0.25428686081518975f}, {0x3da0896cu, 0xbe8231e3u}, -1},
    UpdateStep<2>{{0.20189255627947644f, -0.28346054949742161f}, {0x3e4ebcecu, 0xbe9121beu}, 0},
    PredictStep<2>{{0.34006620546483884f, -0.68804953986294359f}, {0x3eae1d28u, 0xbf302404u}, -1},
    UpdateStep<2>{{0.44177608999051077f, -0.47654089750600465f}, {0x3ee2307au, 0xbef3fd2bu}, 0},
    PredictStep<2>{{0.70703891331911339f, -0.75907217164310203f}, {0x3f350081u, 0xbf42528eu}, -1},
    UpdateStep<2>{{0.48659207720135034f, -0.84973363446972072f}, {0x3ef92299u, 0xbf598825u}, 0},
    PredictStep<2>{{0.75443979600733968f, -1.3242610683722593f}, {0x3f4122f7u, 0xbfa98163u}, -1},
    UpdateStep<2>{{0.63901652987728963f, -2.6068989568960834f}, {0x3f239696u, 0xc026d76fu}, 0},
    PredictStep<2>{{0.37945432735765922f, 85.518783211205772f}, {0x3ec247d6u, 0x42ab099eu}, -1},
    UpdateStep<2>{{-0.011693335048546416f, -5.7003885007819229e-05f}, {0xbc3f9567u, 0xb86f1775u}, 0},
    PredictStep<2>{{17385.401457238353f, -35393.932109073336f}, {0x4687d2ceu, 0xc70a41efu}, -1},
    UpdateStep<2>{{2.8191901920135326e-05f, -7.2142985211668528e-05f}, {0x37ec7da6u, 0xb8974b78u}, 0},
    PredictStep<2>{{13856.743318033416f, -44223.930841064743f}, {0x465882f9u, 0xc72cbfeeu}, -1},
    UpdateStep<2>{{2.2611451797902927e-05f, -9.3945970273874506e-05f}, {0x37bdadb9u, 0xb8c504dcu}, 0},
    PredictStep<2>{{10644.395939496944f, -61611.368893381004f}, {0x46265195u, 0xc770ab5eu}, -1},
    UpdateStep<2>{{1.623076976512752e-05f, -0.00014747866693256927f}, {0x37882750u, 0xb91aa481u}, 0},
    PredictStep<2>{{6780.6417030603779f, -129004.45767463192f}, {0x45d3e522u, 0xc7fbf63bu}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{7.7516701207353075e-06f}, {0x37020d2bu}, 0},
    ScaleEvenStep<1>{{3.2658481961910174e-06f}, {0x365b2ad8u}, 0},
    ScaleOddStep<1>{{-306199.16785057774f}, {0xc89582e5u}, 0});

template <>
struct scheme_traits<coif9_tag> {
    using SchemeType = decltype(coif9_scheme);
    static constexpr const char* name = "coif9";
    static constexpr int id = 31;
    static constexpr int tap_size = 54;
    static constexpr int delay_even = 13;
    static constexpr int delay_odd = 14;
    static constexpr int num_steps = 31;
    static constexpr const auto& scheme = coif9_scheme;
};

}  // namespace ttwv
