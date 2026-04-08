#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "tt-metalium/host_api.hpp"

namespace ttwv::lwt {
struct LwtProgram {
    tt::tt_metal::Program program;
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
};

struct LwtStepConfig {
    uint32_t signal_length;
    uint32_t stick_width{32};
    uint32_t element_size_bytes{sizeof(float)};

    uint32_t split_phase;
    int32_t source_offset;
    uint32_t stencil_k;

    bool is_pad_split{false};
    uint32_t input_length{0};
    uint32_t padded_length{0};
    uint32_t left_pad{0};
    uint32_t num_tiles;

    [[nodiscard]] uint32_t stick_bytes() const noexcept { return stick_width * element_size_bytes; }
    [[nodiscard]] uint32_t aligned_stick_bytes(uint32_t alignment = 32) const noexcept {
        return ((stick_bytes() + alignment - 1) / alignment) * alignment;
    }
    [[nodiscard]] uint32_t output_stick_count() const noexcept {
        return (signal_length + stick_width - 1) / stick_width;
    }
};

[[nodiscard]] LwtProgram create_lwt_step_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& output_buffer,
    const LwtStepConfig& config);

void set_lwt_step_runtime_args(
    const LwtProgram& program_bundle,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& output_buffer,
    const LwtStepConfig& config);

}  // namespace ttwv::lwt
