#pragma once

#include <cstddef>
#include <cstdint>

namespace ttwv {

/**
 * @brief Integer ceiling division: ceil(numerator / denominator).
 *
 * Equivalent to std::ceil((double)numerator / denominator) only in integer arithmetic.
 *
 * @param numerator   Dividend.
 * @param denominator Divisor. Returns 0 if this is 0 (divide-by-zero guard).
 * @return Smallest integer q such that q * denominator >= numerator.
 */
[[nodiscard]] constexpr size_t ceil_div(const size_t numerator, const size_t denominator) noexcept {
    return denominator == 0 ? 0 : (numerator + denominator - 1) / denominator;
}

/**
 * @brief Rounds @p value up to the nearest multiple of @p alignment.
 *
 * If @p value is already a multiple of @p alignment it is returned unchanged.
 * If @p alignment is 0 the function is a no-op and returns @p value.
 *
 * @param value     The value to align.
 * @param alignment The required alignment granularity (e.g. 32 bytes for NOC transfers).
 * @return Smallest multiple of @p alignment that is >= @p value.
 */
[[nodiscard]] constexpr size_t round_up(const std::size_t value, const std::size_t alignment) noexcept {
    return alignment == 0 ? value : ceil_div(value, alignment) * alignment;
}

/**
 * @brief Contiguous 1D signal buffer stored in DRAM as row-major sticks.
 *
 * A **stick** is a fixed-width row of scalar samples that forms the minimum addressable
 * transfer unit for the NOC. For the current fp32 device path `stick_width` is 32,
 * meaning each stick is 128 bytes (32 x 4 B).
 *
 * ### Logical vs physical size
 *
 * The logical signal @ref length may not be a multiple of @ref stick_width.
 * The remaining slots in the last partial stick are zero-padded in DRAM.
 *
 * @code
 * // Example: length = 70, stick_width = 32
 * //
 * // [s0 ... s31] [s32 ... s63] [s64 ... s69 | 0 ... 0]
 * //  stick 0      stick 1      stick 2  (22 padding zeros)
 * //
 * // physical_length() = 3 * 32 = 96
 * // stick_count()     = 3
 * @endcode
 */
struct SignalBuffer {
    uint64_t dram_address{0};                    ///< Base DRAM address of the first stick.
    size_t length{0};                            ///< Logical number of scalar samples in the signal.
    uint32_t stick_width{32};                    ///< Number of scalar samples per physical stick.
    uint32_t element_size_bytes{sizeof(float)};  ///< Size of one scalar element in bytes.

    /**
     * @brief Number of full or partial sticks required to hold the logical signal.
     *
     * Computed as ceil(length / stick_width).
     *
     * @return Number of sticks in DRAM.
     */
    [[nodiscard]] constexpr size_t stick_count() const noexcept {
        return ceil_div(length, static_cast<size_t>(stick_width));
    }

    /**
     * @brief Raw byte size of one stick, without any alignment padding.
     *
     * Equal to stick_width * element_size_bytes.
     *
     * @return Stick size in bytes.
     */
    [[nodiscard]] constexpr uint32_t stick_bytes() const noexcept { return stick_width * element_size_bytes; }

    /**
     * @brief Stick size in bytes rounded up to the nearest multiple of @p alignment.
     *
     * NOC async reads/writes require the transfer size to be a multiple of 32 bytes.
     * Use this value when allocating DRAM pages or passing stick_nbytes to the kernel.
     *
     * @param alignment Byte alignment granularity (default: 32, the NOC minimum).
     * @return Aligned stick size in bytes.
     */
    [[nodiscard]] constexpr uint32_t aligned_stick_bytes(const uint32_t alignment = 32) const noexcept {
        return static_cast<uint32_t>(round_up(static_cast<size_t>(stick_bytes()), alignment));
    }

    /**
     * @brief Total number of scalar slots actually present in DRAM, including zero-padding
     *        in the last stick.
     *
     * Always a multiple of @ref stick_width. May be larger than @ref length when the
     * signal does not fill the last stick completely.
     *
     * @return stick_count() * stick_width.
     */
    [[nodiscard]] constexpr size_t physical_length() const noexcept {
        return stick_count() * static_cast<size_t>(stick_width);
    }

    /**
     * @brief Total byte footprint of the buffer in DRAM, including alignment padding.
     *
     * Use this value to allocate the DRAM buffer (e.g. as the size argument to
     * tt::tt_metal::Buffer).
     *
     * @param alignment Byte alignment per stick (default: 32).
     * @return stick_count() * aligned_stick_bytes(alignment).
     */
    [[nodiscard]] constexpr size_t physical_nbytes(const uint32_t alignment = 32) const noexcept {
        return stick_count() * static_cast<size_t>(aligned_stick_bytes(alignment));
    }
};

/**
 * @brief Even/odd split of a 1D signal stored as two separate DRAM buffers.
 *
 * After splitting, the two sub-signals have half the length of the original:
 * - even[k] holds signal[2k]
 * - odd[k]  holds signal[2k + 1]
 *
 * This split is the first step of the Wavelet Lifting Transform.
 */
struct Signal {
    SignalBuffer even;  ///< Buffer of even-indexed samples: signal[0], signal[2], signal[4], ...
    SignalBuffer odd;   ///< Buffer of odd-indexed samples:  signal[1], signal[3], signal[5], ...
};

/**
 * @brief Builds the even/odd signal descriptors produced by splitting a 1D source signal.
 *
 * @param input         Descriptor of the source signal.
 * @param source_length Logical length of the signal being split into even/odd streams.
 * @param even_addr     DRAM base address for the even-indexed output.
 * @param odd_addr      DRAM base address for the odd-indexed output.
 * @return Even/odd output descriptors for the split signal.
 */
[[nodiscard]] constexpr Signal make_split_signal(
    const SignalBuffer& input, const size_t source_length, const uint64_t even_addr, const uint64_t odd_addr) noexcept {
    const size_t even_len = ceil_div(source_length, size_t{2});
    const size_t odd_len = source_length / 2;

    return Signal{
        .even =
            SignalBuffer{
                .dram_address = even_addr,
                .length = even_len,
                .stick_width = input.stick_width,
                .element_size_bytes = input.element_size_bytes},
        .odd = SignalBuffer{
            .dram_address = odd_addr,
            .length = odd_len,
            .stick_width = input.stick_width,
            .element_size_bytes = input.element_size_bytes}};
}

}  // namespace ttwv
