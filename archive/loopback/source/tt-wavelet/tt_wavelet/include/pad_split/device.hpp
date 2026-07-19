#pragma once

#include <filesystem>

#include "tt-metalium/host_api.hpp"
#include "tt_wavelet/include/pad_split/layout.hpp"

namespace ttwv {

struct PadSplit1DDeviceProgram {
    tt::tt_metal::Program program;
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
};

[[nodiscard]] PadSplit1DDeviceProgram create_pad_split_1d_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& even_buffer,
    const tt::tt_metal::Buffer& odd_buffer,
    const PadSplit1DLayout& layout);

void set_pad_split_1d_runtime_args(
    const PadSplit1DDeviceProgram& program_bundle,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& even_buffer,
    const tt::tt_metal::Buffer& odd_buffer,
    const PadSplit1DLayout& layout);

}  // namespace ttwv
