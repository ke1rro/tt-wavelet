#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ttwv {

/**
 * @brief Signal extension mode used when accessing out-of-bounds indices during padding.
 *
 * @note Padding in this library is a physical operation. Boundary extensions are
 * resolved explicitly by a dedicated padding preprocessing kernel that physically copies
 * the padded and reflected/wrapped values into a new, larger DRAM buffer before the main
 * computation kernels run.
 *
 * Mirrors the modes supported by NumPy / PyWavelets: Zero, Constant, Symmetric, Periodic
 */
enum class BoundaryMode : uint8_t {
    Zero = 0,       ///< Pad with zeros (numpy `constant`, constant_values=0)
    Constant = 1,   ///< Clamp to edge value (numpy `edge`)
    Symmetric = 2,  ///< Mirror reflection (numpy `symmetric` / PyWavelets `sym`)
    Periodic = 3    ///< Wrap-around (numpy `wrap` / PyWavelets `per`)
};

/**
 * @brief Computes @p value modulo @p modulus, always returning a non-negative result.
 *
 * The built-in % operator yield negative results for negative operands,
 * this function maps the result into [0, modulus).
 *
 * @param value   Signed input value.
 * @param modulus Positive modulus; if 0 the function returns 0.
 * @return value mod modulus in the range [0, modulus).
 */
[[nodiscard]] constexpr size_t positive_mod(const int64_t value, const size_t modulus) noexcept {
    if (modulus == 0) {
        return 0;
    }

    const auto signed_modulus{static_cast<int64_t>(modulus)};
    const auto remainder{value % signed_modulus};
    return static_cast<size_t>(remainder < 0 ? remainder + signed_modulus : remainder);
}

/**
 * @brief Maps an arbitrary (possibly negative) index into [0, length) using symmetric reflection.
 *
 * The signal is treated as if it were extended by mirroring at both edges with period 2 * length.
 * This matches NumPy's symmetric and PyWavelets' sym padding.
 *
 * @par Example (length = 5, signal indices 0..4):
 * @code
 * index:  -2  -1   0  1  2  3  4   5   6
 * result:  1   0   0  1  2  3  4   4   3
 * @endcode
 *
 * @param index  Logical sample index (may be negative or beyond `length`).
 * @param length Number of samples in the original signal. Must be > 0
 *               (returns 0 immediately when `length <= 1`).
 * @return Reflected index in `[0, length)`.
 */
[[nodiscard]] constexpr size_t symmetric_index(const int64_t index, const size_t length) noexcept {
    if (length <= 1) {
        return 0;
    }

    const auto period = length * 2;
    const auto reflected = positive_mod(index, period);
    return reflected < length ? reflected : period - 1 - reflected;
}

/**
 * @brief Resolves an out-of-bounds logical index to a valid source index according to
 *        the specified @ref BoundaryMode.
 *
 * This is the single entry-point for boundary handling used by the padding pipeline.
 * It returns std::nullopt whenever the requested position should yield a zero sample
 * (i.e. @ref BoundaryMode::Zero, or an empty signal).
 *
 * | Mode       | In-bounds [0, length) | Negative index        | Index >= length        |
 * |------------|--------------------------|-----------------------|------------------------|
 * | Zero       | nullopt (treat as 0)  | nullopt             | nullopt              |
 * | Constant   | index                 | 0                   | length - 1           |
 * | Symmetric  | index                 | reflected             | reflected              |
 * | Periodic   | index                 | wrapped               | wrapped                |
 *
 * @param index  Logical sample offset (may be negative when inside the left padding region).
 * @param length Number of samples in the original signal.
 * @param mode   Extension mode to apply.
 * @return Source index in [0, length), or std::nullopt if the output should be zero.
 */
[[nodiscard]] constexpr std::optional<size_t> boundary_index(
    const int64_t index, const size_t length, const BoundaryMode mode) noexcept {
    if (length == 0) {
        return std::nullopt;
    }

    switch (mode) {
        case BoundaryMode::Zero: return std::nullopt;
        case BoundaryMode::Constant:
            if (index < 0) {
                return size_t{0};
            }
            return static_cast<size_t>(index) >= length ? length - 1 : static_cast<size_t>(index);
        case BoundaryMode::Symmetric: return symmetric_index(index, length);
        case BoundaryMode::Periodic: return positive_mod(index, length);
    }

    return std::nullopt;
}

// static_assert(symmetric_index(-1, 5) == 0);
// static_assert(symmetric_index(-2, 5) == 1);
// static_assert(symmetric_index(5, 5) == 4);
// static_assert(symmetric_index(6, 5) == 3);
// static_assert(boundary_index(-4, 5, BoundaryMode::Periodic).value() == 1);
// static_assert(boundary_index(-1, 5, BoundaryMode::Constant).value() == 0);
// static_assert(!boundary_index(-1, 5, BoundaryMode::Zero).has_value());

}  // namespace ttwv
