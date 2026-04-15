#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct sym13_tag {};

inline constexpr auto sym13_scheme = make_lifting_scheme(
    26,
    6,
    7,
    PredictStep<1>{ { 0.52400171276804186f }, 0 },
    UpdateStep<2>{ { -0.41111789571429869f, 0.089010065759606211f }, 0 },
    PredictStep<2>{ { -0.14276366028924103f, 541.2981652604617f }, -1 },
    UpdateStep<2>{ { -0.0018474006745758519f, -2.0598521719985809e-06f }, 0 },
    PredictStep<2>{ { 90791.287125446572f, 343518.2713911772f }, -1 },
    UpdateStep<2>{ { -2.2329820596110329e-06f, -2.5591108193455853e-07f }, 0 },
    PredictStep<2>{ { 162008.88366466324f, -1127687.9158900557f }, -1 },
    UpdateStep<2>{ { 6.0031289227665045e-07f, 1.6670119618807615e-07f }, 0 },
    PredictStep<2>{ { -834535.4509742338f, -2748139.5906104571f }, -1 },
    UpdateStep<2>{ { 2.3167595758389408e-07f, 9.2180210681794014e-08f }, 0 },
    PredictStep<2>{ { -2355966.8513249219f, 1807530.2364822316f }, -1 },
    UpdateStep<2>{ { -7.7660207198857414e-08f, -5.9949817750441097e-08f }, 0 },
    PredictStep<2>{ { 1479231.0351313897f, -4896789.0012990721f }, -1 },
    SwapStep{ {}, 0 },
    PredictStep<1>{ { 1.0538682754559374e-07f }, 0 },
    ScaleEvenStep<1>{ { 0.00031946057948789038f }, 0 },
    ScaleOddStep<1>{ { -3130.2766732691862f }, 0 }
);

template <>
struct scheme_traits<sym13_tag> {
    using SchemeType = decltype(sym13_scheme);
    static constexpr const char* name = "sym13";
    static constexpr int id         = 90;
    static constexpr int tap_size   = 26;
    static constexpr int delay_even = 6;
    static constexpr int delay_odd  = 7;
    static constexpr int num_steps  = 17;
    static constexpr const auto& scheme = sym13_scheme;
};

}
