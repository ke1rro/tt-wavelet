#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db36_tag {};

inline constexpr auto db36_scheme = make_lifting_scheme(
    72,
    18,
    18,
    PredictStep<1>{{-0.43729084596680701f}, -1},
    UpdateStep<2>{{-0.5682441736563687f, 0.20318716695292668f}, 0},
    PredictStep<2>{{-0.57657412862286439f, 0.45548562400651915f}, -1},
    UpdateStep<2>{{-0.65635967501468218f, 0.61984454537596489f}, 0},
    PredictStep<2>{{-0.68940846123628963f, 0.64043499365085454f}, -1},
    UpdateStep<2>{{-0.77732156727298074f, 0.70486450111120214f}, 0},
    PredictStep<2>{{-0.80430137627806997f, 0.71235772987088308f}, -1},
    UpdateStep<2>{{-0.89406871095461049f, 0.7583421809870623f}, 0},
    PredictStep<2>{{-0.92517033585029662f, 0.6502304897823582f}, -1},
    UpdateStep<2>{{-1.1298529023725459f, 0.39903224178093993f}, 0},
    PredictStep<2>{{-1.5949583617685441f, 0.29435125673367463f}, -1},
    UpdateStep<2>{{-0.70325603590447361f, 0.39091376251741544f}, 0},
    PredictStep<2>{{-0.10030469972335537f, 0.99757432318145356f}, -1},
    UpdateStep<2>{{-0.50545558261293921f, 1.8579086952987185f}, 0},
    PredictStep<2>{{-0.49236762942508133f, 0.44232504992188826f}, -1},
    UpdateStep<2>{{-2.1996439381008486f, 1.6678141786064977f}, 0},
    PredictStep<2>{{-0.60511491500223402f, 0.4362841496349808f}, -1},
    UpdateStep<2>{{-2.3780881979519504f, 1.6215110045146965f}, 0},
    PredictStep<2>{{-0.65341331922926227f, 0.42071681734135175f}, -1},
    UpdateStep<2>{{-2.5571754752854341f, 2.317757910244199f}, 0},
    PredictStep<2>{{-0.4537389893626857f, -0.015153220935781749f}, -1},
    UpdateStep<2>{{-2.2890199059985479f, -1.102171403235702f}, 0},
    PredictStep<2>{{-0.001803008093170392f, 0.12762459373834442f}, -1},
    UpdateStep<2>{{-9.9946081798656277f, 6.5179104930916152f}, 0},
    PredictStep<2>{{-0.17087666962943063f, 0.076022528629588859f}, -1},
    UpdateStep<2>{{-14.745230338129387f, 5.8484621372179326f}, 0},
    PredictStep<2>{{-0.19347355878603814f, 0.067818264298455416f}, -1},
    UpdateStep<2>{{-16.862739989511301f, 5.1686630235144921f}, 0},
    PredictStep<2>{{-0.2240627115634313f, 0.059302341490681594f}, -1},
    UpdateStep<2>{{-19.837180895382367f, 4.4630362213937671f}, 0},
    PredictStep<2>{{-0.26900809520455865f, 0.050410388711525124f}, -1},
    UpdateStep<2>{{-24.487788286185506f, 3.717360249829154f}, 0},
    PredictStep<2>{{-0.34575696280852558f, 0.04083668105559532f}, -1},
    UpdateStep<2>{{-33.556061446797337f, 2.892204951932615f}, 0},
    PredictStep<2>{{-0.53274607165149268f, 0.029800875218490282f}, -1},
    UpdateStep<2>{{1.6495973217223345e-21f, 1.877066867710611f}, 0},
    PredictStep<1>{{-0.014513477637542129f}, 0},
    ScaleEvenStep<1>{{54462.602001051411f}, 0},
    ScaleOddStep<1>{{1.8361223358015374e-05f}, 0});

template <>
struct scheme_traits<db36_tag> {
    using SchemeType = decltype(db36_scheme);
    static constexpr const char* name = "db36";
    static constexpr int id = 61;
    static constexpr int tap_size = 72;
    static constexpr int delay_even = 18;
    static constexpr int delay_odd = 18;
    static constexpr int num_steps = 39;
    static constexpr const auto& scheme = db36_scheme;
};

}  // namespace ttwv
