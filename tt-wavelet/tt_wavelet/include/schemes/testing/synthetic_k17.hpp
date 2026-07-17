#pragma once

#include "../../lifting/static_scheme.hpp"

namespace ttwv::schemes::testing {

struct synthetic_k17_inverse;

// Hardware acceptance scheme for the maximum SFPI coefficient capacity.  It
// is intentionally not part of the user-visible wavelet registry: its only
// purpose is to exercise all 16 cross-lane rotations of a K=17 stencil.
struct synthetic_k17 {
    static constexpr const char* name = "synthetic-k17";
    static constexpr uint32_t tap_size = 34U;
    static constexpr int32_t delay_even = 8;
    static constexpr int32_t delay_odd = 9;
    static constexpr uint32_t num_steps = 3U;
    static constexpr const char* compute_scheme_header =
        "\"../../tt_wavelet/include/schemes/testing/synthetic_k17.hpp\"";
    static constexpr const char* compute_scheme_type = "ttwv::schemes::testing::synthetic_k17";
    using inverse = synthetic_k17_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct synthetic_k17::step<0> {
    using type = StaticStep<
        StepType::kPredict,
        -8,
        0x3c800000U,
        0x3c800000U,
        0x3c800000U,
        0x3c800000U,
        0x3c800000U,
        0x3c800000U,
        0x3c800000U,
        0x3c800000U,
        0x3c800000U,
        0x3c800000U,
        0x3c800000U,
        0x3c800000U,
        0x3c800000U,
        0x3c800000U,
        0x3c800000U,
        0x3c800000U,
        0x3c800000U>;
    static_assert(type::k == 17U);
};

template <>
struct synthetic_k17::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3f800000U>;
};

template <>
struct synthetic_k17::step<2> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3f800000U>;
};

struct synthetic_k17_inverse {
    static constexpr const char* name = "synthetic-k17-inverse";
    static constexpr uint32_t tap_size = 34U;
    static constexpr uint32_t num_steps = 3U;
    static constexpr const char* compute_scheme_header =
        "\"../../tt_wavelet/include/schemes/testing/synthetic_k17.hpp\"";
    static constexpr const char* compute_scheme_type = "ttwv::schemes::testing::synthetic_k17_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct synthetic_k17_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3f800000U>;
};

template <>
struct synthetic_k17_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3f800000U>;
};

template <>
struct synthetic_k17_inverse::step<2> {
    using type = StaticStep<
        StepType::kPredict,
        -8,
        0xbc800000U,
        0xbc800000U,
        0xbc800000U,
        0xbc800000U,
        0xbc800000U,
        0xbc800000U,
        0xbc800000U,
        0xbc800000U,
        0xbc800000U,
        0xbc800000U,
        0xbc800000U,
        0xbc800000U,
        0xbc800000U,
        0xbc800000U,
        0xbc800000U,
        0xbc800000U,
        0xbc800000U>;
    static_assert(type::k == 17U);
};

}  // namespace ttwv::schemes::testing
