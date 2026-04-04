#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym18_tag {};

inline constexpr auto sym18_scheme = make_lifting_scheme(
    36,
    9,
    9,
    PredictStep<1>{{1.9282472241257345f}, -1},
    UpdateStep<2>{{-0.074401697497713234f, -0.40868823395123349f}, 0},
    PredictStep<2>{{8.8359211123803529f, 1.4745393384806844f}, -1},
    UpdateStep<2>{{0.0068976902652594973f, -0.092312145891622385f}, 0},
    PredictStep<2>{{-22.989096691332353f, -3.4953032352619702f}, -1},
    UpdateStep<2>{{0.054047262509234681f, 0.022471867672537602f}, 0},
    PredictStep<2>{{4.6800512123533462f, -15.926125170134135f}, -1},
    UpdateStep<2>{{0.051094369230440945f, -0.074367084934276281f}, 0},
    PredictStep<2>{{0.91066277484702562f, -3.9393055634775052f}, -1},
    UpdateStep<2>{{-0.03362897625101275f, -0.01459166247090511f}, 0},
    PredictStep<2>{{-5.7590703287070451f, 1.9850508991854621f}, -1},
    UpdateStep<2>{{-0.10928576397066721f, 0.065255293490429231f}, 0},
    PredictStep<2>{{-7.621740630521975f, 5.7470433863256041f}, -1},
    UpdateStep<2>{{0.014746657105318197f, 0.098155412084945784f}, 0},
    PredictStep<2>{{-14.805693301798202f, -4.2604082842157238f}, -1},
    UpdateStep<2>{{-0.0095743844553769653f, 0.030217908869191346f}, 0},
    PredictStep<2>{{28.665817570145162f, 7.8509945988848537f}, -1},
    UpdateStep<2>{{0.030091809247122184f, -0.018141878638098186f}, 0},
    PredictStep<1>{{-24.884396014273594f}, 0},
    ScaleEvenStep<1>{{3.7963607857133135f}, 0},
    ScaleOddStep<1>{{0.26341015947779733f}, 0});

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
