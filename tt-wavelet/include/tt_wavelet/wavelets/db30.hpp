#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db30_tag {};

inline constexpr auto db30_scheme = make_lifting_scheme(
    60,
    15,
    15,
    PredictStep<1>{{-0.53414484604479395f}, {0xbf08bdb7u}, -1},
    UpdateStep<2>{{-0.69432752574341416f, 0.14254399260136924f}, {0xbf31bf73u, 0x3e11f70du}, 0},
    PredictStep<2>{{-0.72250275176740442f, 0.32976719773179131f}, {0xbf38f5f1u, 0x3ea8d73fu}, -1},
    UpdateStep<2>{{-0.8665129309935834f, 0.45710203078485367f}, {0xbf5dd3cbu, 0x3eea0947u}, 0},
    PredictStep<2>{{-1.9534260881568684f, 0.48601549525465487f}, {0xbffa09deu, 0x3ef8d706u}, -1},
    UpdateStep<2>{{0.02567027469152991f, 0.35517404617959375f}, {0x3cd24a78u, 0x3eb5d95fu}, 0},
    PredictStep<2>{{1.523509089348428f, 4.0130807453172865f}, {0x3fc30259u, 0x40806b28u}, -1},
    UpdateStep<2>{{-0.16077257867783445f, 0.10011417020738983f}, {0xbe24a191u, 0x3dcd08a8u}, 0},
    PredictStep<2>{{-6.6949487332580535f, 5.4153074173872184f}, {0xc0d63d05u, 0x40ad4a33u}, -1},
    UpdateStep<2>{{-0.13929663540029708f, 0.11279420076174587f}, {0xbe0ea3c7u, 0x3de700a5u}, 0},
    PredictStep<2>{{-7.3974514727513059f, 5.8377565862385721f}, {0xc0ecb7ecu, 0x40bacee7u}, -1},
    UpdateStep<2>{{-0.1548850981578121f, 0.10857014569899365f}, {0xbe1e9a33u, 0x3dde5a06u}, 0},
    PredictStep<2>{{-8.8406021556557448f, 1.9237816570101465f}, {0xc10d731bu, 0x3ff63e7au}, -1},
    UpdateStep<2>{{-0.54180704286292225f, 0.017968343861349958f}, {0xbf0ab3deu, 0x3c933259u}, 0},
    PredictStep<2>{{6.062596098367079f, 1.3963771367510212f}, {0x40c200cau, 0x3fb2bc7cu}, -1},
    UpdateStep<2>{{0.0014423294391606976f, -0.18482521225601184f}, {0x3abd0c8cu, 0xbe3d42d2u}, 0},
    PredictStep<2>{{3.9940690803119123f, 31.628742793479983f}, {0x407f9ed4u, 0x41fd07aau}, -1},
    UpdateStep<2>{{-0.033811988342148762f, 0.019420605586785971f}, {0xbd0a7e71u, 0x3c9f17f6u}, 0},
    PredictStep<2>{{-56.437573903266696f, 30.93846881135557f}, {0xc261c013u, 0x41f781fcu}, -1},
    UpdateStep<2>{{-0.036012115509342635f, 0.01770269501827286f}, {0xbd138171u, 0x3c91053eu}, 0},
    PredictStep<2>{{-63.828760207961452f, 27.761291152654298f}, {0xc27f50a7u, 0x41de1720u}, -1},
    UpdateStep<2>{{-0.041253111767859965f, 0.015666154943588931f}, {0xbd28f906u, 0x3c80564fu}, 0},
    PredictStep<2>{{-74.155717217918436f, 24.240425164809135f}, {0xc2944fbau, 0x41c1ec64u}, -1},
    UpdateStep<2>{{-0.048730505812005581f, 0.013485126786987633f}, {0xbd4799a4u, 0x3c5cf0b9u}, 0},
    PredictStep<2>{{-89.447306952221084f, 20.521025296174919f}, {0xc2b2e505u, 0x41a42b0fu}, -1},
    UpdateStep<2>{{-0.060452312754135824f, 0.011179766401528487f}, {0xbd779cd8u, 0x3c372b57u}, 0},
    PredictStep<2>{{-115.54591187549815f, 16.541964307693299f}, {0xc2e71782u, 0x418455f1u}, -1},
    UpdateStep<2>{{-0.083254437452455143f, 0.0086545684201738519f}, {0xbdaa814du, 0x3c0dcbe4u}, 0},
    PredictStep<2>{{-178.91156991183627f, 12.011371773078835f}, {0xc332e95du, 0x41402e94u}, -1},
    UpdateStep<2>{{1.8805111523718597e-20f, 0.0055893534470284854f}, {0x1eb19bf8u, 0x3bb726e5u}, 0},
    PredictStep<1>{{-5.8217989077221146f}, {0xc0ba4c2du}, 0},
    ScaleEvenStep<1>{{-132646.35538618942f}, {0xc8018997u}, 0},
    ScaleOddStep<1>{{-7.5388426398040019e-06f}, {0xb6fcf62au}, 0});

template <>
struct scheme_traits<db30_tag> {
    using SchemeType = decltype(db30_scheme);
    static constexpr const char* name = "db30";
    static constexpr int id = 55;
    static constexpr int tap_size = 60;
    static constexpr int delay_even = 15;
    static constexpr int delay_odd = 15;
    static constexpr int num_steps = 33;
    static constexpr const auto& scheme = db30_scheme;
};

}  // namespace ttwv
