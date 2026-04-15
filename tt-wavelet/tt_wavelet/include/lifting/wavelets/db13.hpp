#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct db13_tag {};

inline constexpr auto db13_scheme = make_lifting_scheme(
    26,
    6,
    7,
    PredictStep<1>{ { 9.0045687254486602f }, 0 },
    UpdateStep<2>{ { -0.10970177220050904f, -0.004185954768710822f }, 0 },
    PredictStep<2>{ { 25.2249697946562f, -44.861969329427488f }, -1 },
    UpdateStep<2>{ { 0.0060604849283940401f, -0.009436897897615662f }, 0 },
    PredictStep<2>{ { 50.352337092387899f, -76.026321623507826f }, -1 },
    UpdateStep<2>{ { 0.008860535046289671f, -0.013712458318624211f }, 0 },
    PredictStep<2>{ { 60.656329024856085f, -101.02103244521902f }, -1 },
    UpdateStep<2>{ { 0.0092260817255252724f, -0.017445846601730604f }, 0 },
    PredictStep<2>{ { 56.174486709096364f, -128.13692816901911f }, -1 },
    UpdateStep<2>{ { 0.0077736791632120232f, -0.022882269040755739f }, 0 },
    PredictStep<2>{ { 43.682197370364285f, -180.01673678535431f }, -1 },
    UpdateStep<2>{ { 0.0055548910595801465f, -0.036357709187444631f }, 0 },
    PredictStep<2>{ { 27.504466054487615f, -382.34555165636345f }, -1 },
    SwapStep{ {}, 0 },
    PredictStep<1>{ { 0.0026154351540118863f }, 0 },
    ScaleEvenStep<1>{ { 0.00038518061680364516f }, 0 },
    ScaleOddStep<1>{ { -2596.1846374781971f }, 0 }
);

template <>
struct scheme_traits<db13_tag> {
    using SchemeType = decltype(db13_scheme);
    static constexpr const char* name = "db13";
    static constexpr int id         = 36;
    static constexpr int tap_size   = 26;
    static constexpr int delay_even = 6;
    static constexpr int delay_odd  = 7;
    static constexpr int num_steps  = 17;
    static constexpr const auto& scheme = db13_scheme;
};

}
