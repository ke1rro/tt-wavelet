#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym18_tag {};

inline constexpr auto sym18_scheme = make_lifting_scheme(
    36,
    9,
    9,
    PredictStep<1>{{1.9282472241257345f}, {0x3ff6d0ceu}, -1},
    UpdateStep<2>{{-0.074401697497713234f, -0.40868823395123349f}, {0xbd985febu, 0xbed13f96u}, 0},
    PredictStep<2>{{8.8359211123803529f, 1.4745393384806844f}, {0x410d5fefu, 0x3fbcbdb4u}, -1},
    UpdateStep<2>{{0.0068976902652594973f, -0.092312145891622385f}, {0x3be20605u, 0xbdbd0e26u}, 0},
    PredictStep<2>{{-22.989096691332353f, -3.4953032352619702f}, {0xc1b7e9acu, 0xc05fb30cu}, -1},
    UpdateStep<2>{{0.054047262509234681f, 0.022471867672537602f}, {0x3d5d60aau, 0x3cb816ecu}, 0},
    PredictStep<2>{{4.6800512123533462f, -15.926125170134135f}, {0x4095c2fbu, 0xc17ed169u}, -1},
    UpdateStep<2>{{0.051094369230440945f, -0.074367084934276281f}, {0x3d514854u, 0xbd984dc5u}, 0},
    PredictStep<2>{{0.91066277484702562f, -3.9393055634775052f}, {0x3f692132u, 0xc07c1d95u}, -1},
    UpdateStep<2>{{-0.03362897625101275f, -0.01459166247090511f}, {0xbd09be8au, 0xbc6f11deu}, 0},
    PredictStep<2>{{-5.7590703287070451f, 1.9850508991854621f}, {0xc0b84a4eu, 0x3ffe1626u}, -1},
    UpdateStep<2>{{-0.10928576397066721f, 0.065255293490429231f}, {0xbddfd137u, 0x3d85a491u}, 0},
    PredictStep<2>{{-7.621740630521975f, 5.7470433863256041f}, {0xc0f3e54du, 0x40b7e7c8u}, -1},
    UpdateStep<2>{{0.014746657105318197f, 0.098155412084945784f}, {0x3c719bf6u, 0x3dc905b4u}, 0},
    PredictStep<2>{{-14.805693301798202f, -4.2604082842157238f}, {0xc16ce41fu, 0xc0885544u}, -1},
    UpdateStep<2>{{-0.0095743844553769653f, 0.030217908869191346f}, {0xbc1cdde1u, 0x3cf78b8cu}, 0},
    PredictStep<2>{{28.665817570145162f, 7.8509945988848537f}, {0x41e55398u, 0x40fb3b59u}, -1},
    UpdateStep<2>{{0.030091809247122184f, -0.018141878638098186f}, {0x3cf68319u, 0xbc949e47u}, 0},
    PredictStep<1>{{-24.884396014273594f}, {0xc1c7133eu}, 0},
    ScaleEvenStep<1>{{3.7963607857133135f}, {0x4072f793u}, 0},
    ScaleOddStep<1>{{0.26341015947779733f}, {0x3e86ddb2u}, 0});

template <>
struct scheme_traits<sym18_tag> {
    using SchemeType = decltype(sym18_scheme);
    static constexpr const char* name = "sym18";
    static constexpr int id = 95;
    static constexpr int tap_size = 36;
    static constexpr int delay_even = 9;
    static constexpr int delay_odd = 9;
    static constexpr int num_steps = 21;
    static constexpr const auto& scheme = sym18_scheme;
};

}  // namespace ttwv
