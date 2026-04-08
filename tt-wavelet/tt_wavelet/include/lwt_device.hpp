#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "tt-metalium/host_api.hpp"

namespace ttwv::lwt {

// ---------------------------------------------------------------------------
// Program bundle
// ---------------------------------------------------------------------------

struct LwtProgram {
    tt::tt_metal::Program program;
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
};

// ---------------------------------------------------------------------------
// Step configuration
// ---------------------------------------------------------------------------

/// Configuration for one predict/update lifting step executed on device.
///
/// The reader kernel uses this to produce the correct halo+current tile pair
/// for the stencil: it reads from the sub-band selected by @p split_phase,
/// applies @p source_offset to shift the stencil window, and zero-pads to
/// the 17-lane stencil contract.
struct LwtStepConfig {
    // ---- signal geometry ----
    uint32_t signal_length;   ///< Logical length of the input sub-band (even or odd)
    uint32_t stick_width{32}; ///< Scalar samples per stick (fp32 path: 32)
    uint32_t element_size_bytes{sizeof(float)};

    // ---- reader control ----
    uint32_t split_phase;    ///< 0 = even sub-band, 1 = odd sub-band
    int32_t  source_offset;  ///< Stencil shift (from lifting step's `shift` field)
    uint32_t stencil_k;      ///< Number of non-zero stencil taps

    // ---- pad-split passthrough (set is_pad_split=false for pre-split input) ----
    bool is_pad_split{false};
    uint32_t input_length{0};   ///< Original signal length (only used when is_pad_split)
    uint32_t padded_length{0};  ///< Padded length (only used when is_pad_split)
    uint32_t left_pad{0};       ///< Left symmetric pad (only used when is_pad_split)

    // ---- output ----
    uint32_t num_tiles;  ///< Number of halo/cur tile pairs (= ceil(signal_length / stick_width))

    [[nodiscard]] uint32_t stick_bytes() const noexcept {
        return stick_width * element_size_bytes;
    }
    [[nodiscard]] uint32_t aligned_stick_bytes(uint32_t alignment = 32) const noexcept {
        return ((stick_bytes() + alignment - 1) / alignment) * alignment;
    }
    [[nodiscard]] uint32_t output_stick_count() const noexcept {
        return (signal_length + stick_width - 1) / stick_width;
    }
};

// ---------------------------------------------------------------------------
// Factory & runtime-arg setter
// ---------------------------------------------------------------------------

/// Build and return a complete LWT device program (reader + writer) for one
/// lifting step.  The program is ready to enqueue; runtime args are already
/// set for the provided buffers and config.
[[nodiscard]] LwtProgram create_lwt_step_program(
    const std::filesystem::path&        kernel_root,
    const tt::tt_metal::CoreCoord&      core,
    const tt::tt_metal::Buffer&         input_buffer,
    const tt::tt_metal::Buffer&         output_buffer,
    const LwtStepConfig&                config);

/// Update only the runtime args of an existing @ref LwtProgram (avoids
/// re-compilation when iterating over lifting steps with the same kernel).
void set_lwt_step_runtime_args(
    const LwtProgram&                   program_bundle,
    const tt::tt_metal::CoreCoord&      core,
    const tt::tt_metal::Buffer&         input_buffer,
    const tt::tt_metal::Buffer&         output_buffer,
    const LwtStepConfig&                config);

}  // namespace ttwv::lwt
