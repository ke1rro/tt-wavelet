#include <bit>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

#include "tt_wavelet/include/lifting/device.hpp"
#include "tt_wavelet/include/schemes/registry.hpp"

namespace {

bool fail(const std::string_view message) {
    std::cerr << message << '\n';
    return false;
}

bool check_registry() {
    if (ttwv::kSchemeRegistry.size() != 106) {
        return fail("generated registry must contain all 106 schemes");
    }
    if (ttwv::find_scheme("bior1.1") == nullptr || ttwv::find_scheme("coif17") == nullptr) {
        return fail("known generated scheme lookup failed");
    }
    if (ttwv::find_scheme("missing-scheme") != nullptr) {
        return fail("unknown generated scheme lookup must fail");
    }

    bool found_k1 = false;
    bool found_negative_shift = false;
    bool found_scale = false;
    bool found_swap = false;
    for (const auto& entry : ttwv::kSchemeRegistry) {
        if (!ttwv::validate_lifting_scheme(*entry.scheme)) {
            return fail("generated registry contains an invalid descriptor");
        }
        for (const auto& step : entry.scheme->steps()) {
            found_k1 |=
                (step.type() == ttwv::LiftingStepType::kPredict || step.type() == ttwv::LiftingStepType::kUpdate) &&
                step.coeffs().size() == 1;
            found_negative_shift |= step.shift() < 0;
            found_scale |=
                step.type() == ttwv::LiftingStepType::kScaleEven || step.type() == ttwv::LiftingStepType::kScaleOdd;
            found_swap |= step.type() == ttwv::LiftingStepType::kSwap;
        }
    }
    if (!found_k1 || !found_negative_shift || !found_scale || !found_swap) {
        return fail("registry coverage is missing a required step form");
    }
    return true;
}

bool check_packing() {
    const auto* scheme = ttwv::find_scheme("coif17");
    if (scheme == nullptr) {
        return fail("coif17 not found");
    }
    const std::vector<uint32_t> packed = ttwv::pack_compile_time_steps(*scheme);
    size_t cursor = 0;
    for (const auto& step : scheme->steps()) {
        if (cursor >= packed.size()) {
            return fail("packed stream ended before coif17 descriptor");
        }
        const uint32_t expected_meta =
            (static_cast<uint32_t>(step.coeffs().size()) << 3) | static_cast<uint32_t>(step.type());
        if (packed[cursor++] != expected_meta) {
            return fail("packed step metadata does not match descriptor");
        }
        if (step.type() == ttwv::LiftingStepType::kPredict || step.type() == ttwv::LiftingStepType::kUpdate) {
            if (cursor >= packed.size() || packed[cursor++] != static_cast<uint32_t>(step.shift())) {
                return fail("packed step shift does not match descriptor");
            }
        }
        for (const float coefficient : step.coeffs()) {
            if (cursor >= packed.size() || packed[cursor++] != std::bit_cast<uint32_t>(coefficient)) {
                return fail("packed coefficient does not match descriptor");
            }
        }
    }
    if (cursor != packed.size() || packed.empty()) {
        return fail("packed coif17 stream contains trailing or no words");
    }
    return true;
}

}  // namespace

int main() {
    if (!check_registry() || !check_packing()) {
        return 1;
    }
    const auto* bior13 = ttwv::find_scheme("bior1.3");
    if (bior13 == nullptr || ttwv::canonical_output_length(5, *bior13) != 5) {
        return 1;
    }
    std::cout << "scheme registry and compile-time packing regression passed\n";
    return 0;
}
