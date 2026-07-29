#pragma once

#include <cstdint>

#include "../../tt_wavelet/include/common/signal_extension.hpp"

#define ALWI inline __attribute__((always_inline))

namespace ttwv::kernels::primitives {

ALWI uint32_t positive_mod(const int32_t value, const uint32_t modulus) {
    return ttwv::extension_positive_mod(value, modulus);
}

using SignedPeriodIndex = ttwv::SignedExtensionPeriod;

ALWI SignedPeriodIndex decompose_signed_period(const int32_t value, const uint64_t period) {
    return ttwv::decompose_extension_period(value, period);
}

ALWI uint32_t symmetric_index(const int32_t index, const uint32_t length) {
    return ttwv::make_extended_index<ttwv::BoundaryMode::kSymmetric>(index, length).source_index;
}

ALWI uint32_t reflect_index(const int32_t index, const uint32_t length) {
    return ttwv::make_extended_index<ttwv::BoundaryMode::kReflect>(index, length).source_index;
}

struct AntisymmetricIndex {
    uint32_t source_index;
    bool negate;
};

ALWI AntisymmetricIndex antisymmetric_index(const int32_t index, const uint32_t length) {
    const ttwv::ExtendedIndex extended =
        ttwv::make_extended_index<ttwv::BoundaryMode::kAntisymmetric>(index, length);
    return AntisymmetricIndex{
        .source_index = extended.source_index,
        .negate = extended.operation == ttwv::ExtensionOperation::kNegatedSample,
    };
}

}  // namespace ttwv::kernels::primitives
