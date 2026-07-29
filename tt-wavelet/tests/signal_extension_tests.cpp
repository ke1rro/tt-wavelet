#include <bit>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "tt_wavelet/include/common/signal_extension.hpp"

namespace {

constexpr std::array<ttwv::BoundaryMode, 8> kModes = {
    ttwv::BoundaryMode::kZero,
    ttwv::BoundaryMode::kConstant,
    ttwv::BoundaryMode::kSymmetric,
    ttwv::BoundaryMode::kReflect,
    ttwv::BoundaryMode::kPeriodic,
    ttwv::BoundaryMode::kSmooth,
    ttwv::BoundaryMode::kAntisymmetric,
    ttwv::BoundaryMode::kAntireflect,
};

[[nodiscard]] int64_t positive_mod(const int64_t value, const int64_t modulus) {
    const int64_t remainder = value % modulus;
    return remainder < 0 ? remainder + modulus : remainder;
}

[[nodiscard]] float independent_extension(
    const std::vector<float>& values, const int64_t index, const ttwv::BoundaryMode mode) {
    const int64_t length = static_cast<int64_t>(values.size());
    if (index >= 0 && index < length) {
        return values[static_cast<size_t>(index)];
    }
    switch (mode) {
        case ttwv::BoundaryMode::kZero: return 0.0F;
        case ttwv::BoundaryMode::kConstant: return index < 0 ? values.front() : values.back();
        case ttwv::BoundaryMode::kPeriodic:
            return values[static_cast<size_t>(positive_mod(index, length))];
        case ttwv::BoundaryMode::kSymmetric: {
            const int64_t phase = positive_mod(index, 2 * length);
            return values[static_cast<size_t>(phase < length ? phase : 2 * length - 1 - phase)];
        }
        case ttwv::BoundaryMode::kReflect: {
            if (length == 1) {
                return values.front();
            }
            const int64_t period = 2 * (length - 1);
            const int64_t phase = positive_mod(index, period);
            return values[static_cast<size_t>(phase < length ? phase : period - phase)];
        }
        case ttwv::BoundaryMode::kAntisymmetric: {
            const int64_t phase = positive_mod(index, 4 * length);
            if (phase < length) {
                return values[static_cast<size_t>(phase)];
            }
            if (phase < 2 * length) {
                return -values[static_cast<size_t>(2 * length - 1 - phase)];
            }
            if (phase < 3 * length) {
                return values[static_cast<size_t>(phase - 2 * length)];
            }
            return -values[static_cast<size_t>(4 * length - 1 - phase)];
        }
        case ttwv::BoundaryMode::kSmooth: {
            if (length == 1) {
                return values.front();
            }
            if (index < 0) {
                return values.front() + static_cast<float>(-index) * (values.front() - values[1]);
            }
            return values.back() +
                   static_cast<float>(index - length + 1) * (values.back() - values[values.size() - 2]);
        }
        case ttwv::BoundaryMode::kAntireflect: {
            if (length == 1) {
                return values.front();
            }
            if (index < 0) {
                return 2.0F * values.front() - independent_extension(values, -index, mode);
            }
            return 2.0F * values.back() - independent_extension(values, 2 * length - 2 - index, mode);
        }
    }
    throw std::runtime_error("unsupported mode");
}

struct VectorReader {
    const std::vector<float>& values;
    [[nodiscard]] float operator()(const uint32_t index) const { return values[index]; }
};

[[nodiscard]] float canonical_extension(
    const std::vector<float>& values, const int64_t index, const ttwv::BoundaryMode mode) {
    const ttwv::ExtendedIndex extended =
        ttwv::make_extended_index(mode, index, static_cast<uint32_t>(values.size()));
    return ttwv::evaluate_extended_index(extended, static_cast<uint32_t>(values.size()), VectorReader{values});
}

void require_equal(const float expected, const float actual, const char* label) {
    if (std::bit_cast<uint32_t>(expected) != std::bit_cast<uint32_t>(actual) &&
        std::abs(expected - actual) > 1.0e-6F) {
        throw std::runtime_error(
            std::string{label} + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual));
    }
}

void validate_1d() {
    for (const uint32_t length : {1U, 2U, 3U, 4U, 5U, 8U, 9U}) {
        std::vector<float> values(length);
        for (uint32_t index = 0; index < length; ++index) {
            values[index] = static_cast<float>(3 * index + 1);
        }
        for (const ttwv::BoundaryMode mode : kModes) {
            for (int64_t index = -3 * static_cast<int64_t>(length);
                 index <= 4 * static_cast<int64_t>(length);
                 ++index) {
                require_equal(
                    independent_extension(values, index, mode),
                    canonical_extension(values, index, mode),
                    "1D extension");
            }
        }
    }
}

void validate_compact_antireflect() {
    constexpr std::array<int32_t, 11> kExtremeIndices = {
        std::numeric_limits<int32_t>::min(),
        std::numeric_limits<int32_t>::min() + 1,
        -65537,
        -1025,
        -1,
        0,
        1,
        1025,
        65537,
        std::numeric_limits<int32_t>::max() - 1,
        std::numeric_limits<int32_t>::max(),
    };
    for (const uint32_t length : {1U, 2U, 3U, 4U, 5U, 8U, 9U, 1024U}) {
        for (const int32_t index : kExtremeIndices) {
            const ttwv::ExtendedIndex canonical =
                ttwv::make_extended_index<ttwv::BoundaryMode::kAntireflect>(index, length);
            const ttwv::AntireflectIndexI32 compact =
                ttwv::make_antireflect_index_i32(index, length);
            if (compact.source_index != canonical.source_index ||
                compact.period_quotient != canonical.period_quotient ||
                compact.reflected != canonical.reflected ||
                compact.affine !=
                    (canonical.operation == ttwv::ExtensionOperation::kAntireflect)) {
                throw std::runtime_error("compact antireflect descriptor differs from canonical mapping");
            }
        }
    }
}

void validate_compact_smooth() {
    constexpr std::array<int32_t, 11> kExtremeIndices = {
        std::numeric_limits<int32_t>::min(),
        std::numeric_limits<int32_t>::min() + 1,
        -65537,
        -1025,
        -1,
        0,
        1,
        1025,
        65537,
        std::numeric_limits<int32_t>::max() - 1,
        std::numeric_limits<int32_t>::max(),
    };
    for (const uint32_t length : {1U, 2U, 3U, 4U, 5U, 8U, 9U, 1024U}) {
        for (const int32_t index : kExtremeIndices) {
            const ttwv::ExtendedIndex canonical =
                ttwv::make_extended_index<ttwv::BoundaryMode::kSmooth>(index, length);
            const ttwv::SmoothIndexI32 compact =
                ttwv::make_smooth_index_i32(index, length);
            if (compact.source_index != canonical.source_index ||
                compact.auxiliary_index != canonical.auxiliary_index ||
                compact.distance != canonical.distance ||
                compact.affine != (canonical.operation == ttwv::ExtensionOperation::kSmooth)) {
                throw std::runtime_error("compact smooth descriptor differs from canonical mapping");
            }
        }
    }
}

void validate_compact_symmetric() {
    constexpr std::array<int32_t, 11> kExtremeIndices = {
        std::numeric_limits<int32_t>::min(),
        std::numeric_limits<int32_t>::min() + 1,
        -65537,
        -1025,
        -1,
        0,
        1,
        1025,
        65537,
        std::numeric_limits<int32_t>::max() - 1,
        std::numeric_limits<int32_t>::max(),
    };
    for (const uint32_t length : {1U, 2U, 3U, 4U, 5U, 8U, 9U, 1024U}) {
        for (const int32_t index : kExtremeIndices) {
            const ttwv::ExtendedIndex canonical =
                ttwv::make_extended_index<ttwv::BoundaryMode::kSymmetric>(index, length);
            const uint32_t compact = ttwv::make_symmetric_index_i32(index, length);
            if (compact != canonical.source_index) {
                throw std::runtime_error("compact symmetric index differs from canonical mapping");
            }
        }
    }
}

void validate_corners() {
    constexpr uint32_t kHeight = 3;
    constexpr uint32_t kWidth = 5;
    std::vector<float> matrix(kHeight * kWidth);
    for (uint32_t index = 0; index < matrix.size(); ++index) {
        matrix[index] = static_cast<float>(index + 1);
    }
    for (const ttwv::BoundaryMode mode : kModes) {
        for (int64_t y = -3 * kHeight; y <= 4 * kHeight; ++y) {
            std::vector<float> horizontal_rows(kHeight);
            for (uint32_t row = 0; row < kHeight; ++row) {
                const std::vector<float> source_row(
                    matrix.begin() + static_cast<std::ptrdiff_t>(row * kWidth),
                    matrix.begin() + static_cast<std::ptrdiff_t>((row + 1) * kWidth));
                horizontal_rows[row] = independent_extension(source_row, -7, mode);
            }
            const float expected = independent_extension(horizontal_rows, y, mode);

            const ttwv::ExtendedIndex y_extension = ttwv::make_extended_index(mode, y, kHeight);
            struct RowReader {
                const std::vector<float>& matrix;
                ttwv::BoundaryMode mode;
                [[nodiscard]] float operator()(const uint32_t row) const {
                    const std::vector<float> values(
                        matrix.begin() + static_cast<std::ptrdiff_t>(row * kWidth),
                        matrix.begin() + static_cast<std::ptrdiff_t>((row + 1) * kWidth));
                    return canonical_extension(values, -7, mode);
                }
            };
            const float actual = ttwv::evaluate_extended_index(
                y_extension, kHeight, RowReader{.matrix = matrix, .mode = mode});
            require_equal(expected, actual, "2D corner extension");
        }
    }
}

}  // namespace

int main() {
    try {
        validate_1d();
        validate_compact_antireflect();
        validate_compact_smooth();
        validate_compact_symmetric();
        validate_corners();
        std::cout << "signal extension tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
