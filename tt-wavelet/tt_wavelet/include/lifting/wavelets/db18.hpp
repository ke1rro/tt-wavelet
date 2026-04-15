#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct db18_tag {};

inline constexpr auto db18_scheme = make_lifting_scheme(
    36,
    9,
    9,
    PredictStep<1>{ { -0.44801197432740864f }, -1 },
    UpdateStep<2>{ { -0.65439494253147834f, 0.078836219876662034f }, 0 },
    PredictStep<2>{ { -0.94979565185576209f, 0.21893743829400378f }, -1 },
    UpdateStep<2>{ { 0.10827447145221548f, 0.33173269911001835f }, 0 },
    PredictStep<2>{ { 0.12237094103982626f, 0.9457783447201622f }, -1 },
    UpdateStep<2>{ { -0.52356363092208258f, 0.296982814658587f }, 0 },
    PredictStep<2>{ { -2.010333066554256f, 1.4610576544964944f }, -1 },
    UpdateStep<2>{ { -0.5149721320813373f, 0.36584145909473414f }, 0 },
    PredictStep<2>{ { -2.4291478945175928f, 1.6391219738866543f }, -1 },
    UpdateStep<2>{ { -0.60935313541912317f, 0.3797198464127472f }, 0 },
    PredictStep<2>{ { -2.8519359137520768f, 1.5891648679154167f }, -1 },
    UpdateStep<2>{ { -0.72144149784518841f, 0.34711775410898565f }, 0 },
    PredictStep<2>{ { -3.4513989187864977f, 1.3828662491141075f }, -1 },
    UpdateStep<2>{ { -0.90223749042783929f, 0.28962839830966192f }, 0 },
    PredictStep<2>{ { -4.5149088884059854f, 1.1083125704712173f }, -1 },
    UpdateStep<2>{ { -1.2618103141511194f, 0.22148789348274509f }, 0 },
    PredictStep<2>{ { -7.1081599770986728f, 0.79251208890441149f }, -1 },
    UpdateStep<2>{ { 6.7023466313744001e-12f, 0.14068338393131191f }, 0 },
    PredictStep<1>{ { -0.37767835630208652f }, 0 },
    ScaleEvenStep<1>{ { 487.21853672112877f }, 0 },
    ScaleOddStep<1>{ { 0.0020524670648407084f }, 0 }
);

template <>
struct scheme_traits<db18_tag> {
    using SchemeType = decltype(db18_scheme);
    static constexpr const char* name = "db18";
    static constexpr int id         = 41;
    static constexpr int tap_size   = 36;
    static constexpr int delay_even = 9;
    static constexpr int delay_odd  = 9;
    static constexpr int num_steps  = 21;
    static constexpr const auto& scheme = db18_scheme;
};

}
