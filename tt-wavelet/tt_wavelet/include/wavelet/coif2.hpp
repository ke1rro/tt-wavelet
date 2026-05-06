#pragma once

#include "wavelet/scheme.hpp"

namespace ttwv::schemes {

const LiftingScheme& coif2(
    LiftingStep(Predict, { -0.0156557281f }, 2),
    LiftingStep(Update, { -0.0727326195f }, 1),
    LiftingStep(Predict, { 0.3848648469f }, 2),
    LiftingStep(Update, { 0.8525720202f }, 1),
    LiftingStep(Predict, { 0.8525720202f }, 2),
    LiftingStep(Update, { 0.3848648469f }, 1),
);

}
