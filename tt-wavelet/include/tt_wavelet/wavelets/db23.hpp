#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db23_tag {};

inline constexpr auto db23_scheme = make_lifting_scheme(
    46,
    11,
    12,
    PredictStep<1>{{5.8087544093004917f}, {0x40b9e151u}, 0},
    UpdateStep<2>{{-0.16238009201981232f, -0.010733356002501724f}, {0xbe2646f8u, 0xbc2fdaf5u}, 0},
    PredictStep<2>{{15.804004115036234f, -10.604294330651369f}, {0x417cdd33u, 0xc129ab31u}, -1},
    UpdateStep<2>{{0.018063142219162838f, -0.0084474157690681943f}, {0x3c93f928u, 0xbc0a6708u}, 0},
    PredictStep<2>{{23.652811846874876f, -14.25360234149939f}, {0x41bd38f5u, 0xc1640ec1u}, -1},
    UpdateStep<2>{{0.023667946566689954f, -0.017505522960417831f}, {0x3cc1e348u, 0xbc8f67beu}, 0},
    PredictStep<2>{{29.890254388696619f, -23.071153351765158f}, {0x41ef1f3eu, 0xc1b891b9u}, -1},
    UpdateStep<2>{{0.028941860984958472f, -0.022224564292268534f}, {0x3ced177bu, 0xbcb6104au}, 0},
    PredictStep<2>{{35.677246584377912f, -26.560001957866348f}, {0x420eb580u, 0xc1d47ae2u}, -1},
    UpdateStep<2>{{0.74444402988591996f, -0.023974131008180025f}, {0x3f3e93e2u, 0xbcc46566u}, 0},
    PredictStep<2>{{-0.00065356228469041935f, -1.337993210269983f}, {0xba2b53d3u, 0xbfab435du}, -1},
    UpdateStep<2>{{-14.601898160194628f, -304.67752402843837f}, {0xc169a160u, 0xc39856b9u}, 0},
    PredictStep<2>{{0.0031735437042823872f, -0.0040285783757549355f}, {0x3b4ffb3au, 0xbb84022au}, -1},
    UpdateStep<2>{{244.0500852142703f, -422.79220030934113f}, {0x43740cd2u, 0xc3d36567u}, 0},
    PredictStep<2>{{0.0023518610695271031f, -0.0046142249004918695f}, {0x3b1a21aeu, 0xbb9732edu}, -1},
    UpdateStep<2>{{216.40161243911314f, -490.37262286901006f}, {0x435866d0u, 0xc3f52fb2u}, 0},
    PredictStep<2>{{0.0020386793636338698f, -0.0054588158893516796f}, {0x3b059b5du, 0xbbb2dfdeu}, -1},
    UpdateStep<2>{{183.18256511707997f, -594.16232694325515f}, {0x43372ebdu, 0xc4148a64u}, 0},
    PredictStep<2>{{0.0016830353300500654f, -0.0068162874445142595f}, {0x3adc994bu, 0xbbdf5b2au}, -1},
    UpdateStep<2>{{146.70739397908309f, -773.42597392397067f}, {0x4312b518u, 0xc4415b43u}, 0},
    PredictStep<2>{{0.00129294855015606f, -0.0094643016011484588f}, {0x3aa97827u, 0xbc1b1028u}, -1},
    UpdateStep<2>{{105.66019997953076f, -1207.5477732574536f}, {0x42d35206u, 0xc496f187u}, 0},
    PredictStep<2>{{0.00082812458616128104f, -0.01968724103709521f}, {0x3a591680u, 0xbca14723u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{50.794318925428499f}, {0x424b2d62u}, 0},
    ScaleEvenStep<1>{{0.0015282342839286233f}, {0x3ac84f09u}, 0},
    ScaleOddStep<1>{{-654.34993215130976f}, {0xc4239665u}, 0});

template <>
struct scheme_traits<db23_tag> {
    using SchemeType = decltype(db23_scheme);
    static constexpr const char* name = "db23";
    static constexpr int id = 47;
    static constexpr int tap_size = 46;
    static constexpr int delay_even = 11;
    static constexpr int delay_odd = 12;
    static constexpr int num_steps = 27;
    static constexpr const auto& scheme = db23_scheme;
};

}  // namespace ttwv
