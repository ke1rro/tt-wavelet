#pragma once

#include <filesystem>

#include "tt-metalium/host_api.hpp"
#include "tt_wavelet/include/split.hpp"

namespace ttwv {

struct Split1DDeviceProgram {
    tt::tt_metal::Program program;
    tt::tt_metal::KernelHandle reader_kernel;
    tt::tt_metal::KernelHandle writer_kernel;
};

[[nodiscard]] Split1DDeviceProgram create_split_1d_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& even_buffer,
    const tt::tt_metal::Buffer& odd_buffer,
    const Split1DLayout& layout);

void set_split_1d_runtime_args(
    const Split1DDeviceProgram& program_bundle,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& even_buffer,
    const tt::tt_metal::Buffer& odd_buffer,
    const Split1DLayout& layout);

}  // namespace ttwv
