#pragma once

#include <filesystem>

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_wavelet/include/pad_split/layout.hpp"

namespace ttwv {

struct PadSplit1DDeviceProgram {
    tt::tt_metal::Program program;
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
};

[[nodiscard]] PadSplit1DDeviceProgram create_pad_split_1d_program(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& even_buffer,
    const tt::tt_metal::Buffer& odd_buffer,
    const PadSplit1DLayout& layout);

}  // namespace ttwv
