#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db30_tag {};

inline constexpr auto db30_scheme = make_lifting_scheme(
    60,
    15,
    15,
    PredictStep<1>{{-0.53414484604479395f}, -1},
    UpdateStep<2>{{-0.69432752574341416f, 0.14254399260136924f}, 0},
    PredictStep<2>{{-0.72250275176740442f, 0.32976719773179131f}, -1},
    UpdateStep<2>{{-0.8665129309935834f, 0.45710203078485367f}, 0},
    PredictStep<2>{{-1.9534260881568684f, 0.48601549525465487f}, -1},
    UpdateStep<2>{{0.02567027469152991f, 0.35517404617959375f}, 0},
    PredictStep<2>{{1.523509089348428f, 4.0130807453172865f}, -1},
    UpdateStep<2>{{-0.16077257867783445f, 0.10011417020738983f}, 0},
    PredictStep<2>{{-6.6949487332580535f, 5.4153074173872184f}, -1},
    UpdateStep<2>{{-0.13929663540029708f, 0.11279420076174587f}, 0},
    PredictStep<2>{{-7.3974514727513059f, 5.8377565862385721f}, -1},
    UpdateStep<2>{{-0.1548850981578121f, 0.10857014569899365f}, 0},
    PredictStep<2>{{-8.8406021556557448f, 1.9237816570101465f}, -1},
    UpdateStep<2>{{-0.54180704286292225f, 0.017968343861349958f}, 0},
    PredictStep<2>{{6.062596098367079f, 1.3963771367510212f}, -1},
    UpdateStep<2>{{0.0014423294391606976f, -0.18482521225601184f}, 0},
    PredictStep<2>{{3.9940690803119123f, 31.628742793479983f}, -1},
    UpdateStep<2>{{-0.033811988342148762f, 0.019420605586785971f}, 0},
    PredictStep<2>{{-56.437573903266696f, 30.93846881135557f}, -1},
    UpdateStep<2>{{-0.036012115509342635f, 0.01770269501827286f}, 0},
    PredictStep<2>{{-63.828760207961452f, 27.761291152654298f}, -1},
    UpdateStep<2>{{-0.041253111767859965f, 0.015666154943588931f}, 0},
    PredictStep<2>{{-74.155717217918436f, 24.240425164809135f}, -1},
    UpdateStep<2>{{-0.048730505812005581f, 0.013485126786987633f}, 0},
    PredictStep<2>{{-89.447306952221084f, 20.521025296174919f}, -1},
    UpdateStep<2>{{-0.060452312754135824f, 0.011179766401528487f}, 0},
    PredictStep<2>{{-115.54591187549815f, 16.541964307693299f}, -1},
    UpdateStep<2>{{-0.083254437452455143f, 0.0086545684201738519f}, 0},
    PredictStep<2>{{-178.91156991183627f, 12.011371773078835f}, -1},
    UpdateStep<2>{{1.8805111523718597e-20f, 0.0055893534470284854f}, 0},
    PredictStep<1>{{-5.8217989077221146f}, 0},
    ScaleEvenStep<1>{{-132646.35538618942f}, 0},
    ScaleOddStep<1>{{-7.5388426398040019e-06f}, 0});

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
