#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "tt-metalium/host_api.hpp"
#include "stencil_device.hpp"


namespace ttwv::lwt {
    struct LWTProgram{
        tt::tt_metal::Program program;
        tt::tt_metal::KernelHandle reader;
        tt::tt_metal::KernelHandle compute;
        tt::tt_metal::KernelHandle writer;
    };

    [[nodiscard]] LWTProgram create_1dlwt_program(
        const std::filesystem::path& kernel_root,
        const tt::tt_metal::CoreCoord& core,
        const tt::tt_metal::Buffer& input_buffer,
        const tt::tt_metal::Buffer& output_buffer,
        const StencilConfig& config);
} // namespace ttwv::lwt
