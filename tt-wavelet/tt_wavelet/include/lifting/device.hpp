#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_wavelet/include/schemes/scheme.hpp"

namespace ttwv {

struct WaveletCoefficients {
    std::vector<float> approximation;
    std::vector<float> detail;
};

class WaveletProgram {
public:
    WaveletProgram(
        const std::filesystem::path& kernel_root,
        tt::tt_metal::distributed::MeshDevice& mesh_device,
        tt::tt_metal::distributed::MeshCommandQueue& command_queue,
        const tt::tt_metal::CoreCoord& core,
        const LiftingScheme& scheme,
        size_t input_length);
    ~WaveletProgram();

    WaveletProgram(WaveletProgram&&) noexcept;
    WaveletProgram& operator=(WaveletProgram&&) noexcept;
    WaveletProgram(const WaveletProgram&) = delete;
    WaveletProgram& operator=(const WaveletProgram&) = delete;

    [[nodiscard]] WaveletCoefficients execute(std::span<const float> signal);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::vector<uint32_t> pack_compile_time_steps(const LiftingScheme& scheme);

[[nodiscard]] size_t canonical_output_length(size_t input_length, const LiftingScheme& scheme) noexcept;

[[nodiscard]] bool validate_lifting_scheme(const LiftingScheme& scheme);

}  // namespace ttwv
