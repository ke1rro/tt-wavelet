#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif8_tag {};

inline constexpr auto coif8_scheme = make_lifting_scheme(
    48,
    12,
    12,
    PredictStep<1>{{0.67631809034829105f}, -1},
    UpdateStep<2>{{1.5896745497507698f, -0.46405601212645581f}, 0},
    PredictStep<2>{{0.46035159964435418f, -0.53026841090191967f}, -1},
    UpdateStep<2>{{1.1609191018052829f, -1.7417146245201596f}, 0},
    PredictStep<2>{{0.26281400880970168f, -0.55344912932950996f}, -1},
    UpdateStep<2>{{0.75598932101149952f, -1.0973497600712532f}, 0},
    PredictStep<2>{{0.15961597437918118f, -0.21339181886871209f}, -1},
    UpdateStep<2>{{0.14054404564192516f, -0.60873454701888052f}, 0},
    PredictStep<2>{{-0.050803381611308041f, -0.040585187766643473f}, -1},
    UpdateStep<2>{{-0.37712108505884318f, 0.1753619491361415f}, 0},
    PredictStep<2>{{-0.24215291814877107f, 0.10583639964745155f}, -1},
    UpdateStep<2>{{-1.0399569289669657f, 0.73809172033061377f}, 0},
    PredictStep<2>{{-0.28707493145241247f, 0.29010416386124022f}, -1},
    UpdateStep<2>{{-1.7610904097191147f, 1.0355885292104703f}, 0},
    PredictStep<2>{{-0.59031009228845177f, 0.31245081691525128f}, -1},
    UpdateStep<2>{{-5.1748969234119979f, 1.3783936323316173f}, 0},
    PredictStep<2>{{-1.9704217317870354f, 0.19015134087314237f}, -1},
    UpdateStep<2>{{-0.066082289037217135f, 0.50742879545394337f}, 0},
    PredictStep<2>{{-32.877410759938073f, 14.998844007119581f}, -1},
    UpdateStep<2>{{-0.085941363178906563f, 0.030359653173309076f}, 0},
    PredictStep<2>{{-42.960475947506851f, 11.633145204948198f}, -1},
    UpdateStep<2>{{-0.11994479495255828f, 0.023276814069547704f}, 0},
    PredictStep<2>{{-67.642405756567896f, 8.3371634473372662f}, -1},
    UpdateStep<2>{{8.4262744082886485e-12f, 0.014783625435780282f}, 0},
    PredictStep<1>{{-3.966568114953072f}, 0},
    ScaleEvenStep<1>{{828.31367157113618f}, 0},
    ScaleOddStep<1>{{0.0012072721172200515f}, 0});

template <>
struct scheme_traits<coif8_tag> {
    using SchemeType = decltype(coif8_scheme);
    static constexpr const char* name = "coif8";
    static constexpr int id = 30;
    static constexpr int tap_size = 48;
    static constexpr int delay_even = 12;
    static constexpr int delay_odd = 12;
    static constexpr int num_steps = 27;
    static constexpr const auto& scheme = coif8_scheme;
};

}  // namespace ttwv
