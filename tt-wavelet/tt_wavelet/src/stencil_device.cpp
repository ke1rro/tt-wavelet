#include "tt_wavelet/include/stencil_device.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <tt_stl/assert.hpp>
#include <vector>

#include "tt-metalium/tensor_accessor_args.hpp"

namespace ttwv {

namespace {

// CB indices matching the compute kernel expectations
constexpr uint32_t kCbHalo = tt::CBIndex::c_0;        // reader → compute (halo sticks)
constexpr uint32_t kCbCur = tt::CBIndex::c_1;         // reader → compute (current sticks)
constexpr uint32_t kCbTilizedH = tt::CBIndex::c_2;    // compute internal (tilized halo)
constexpr uint32_t kCbTilizedC = tt::CBIndex::c_3;    // compute internal (tilized current)
constexpr uint32_t kCbTilizedOut = tt::CBIndex::c_4;  // compute internal (tilized output)
constexpr uint32_t kCbOut = tt::CBIndex::c_16;        // compute → writer (output sticks)
constexpr uint32_t kCbCache = tt::CBIndex::c_5;       // reader DRAM cache

constexpr uint32_t kAlignment = 32;

constexpr const char* kReaderKernelPath = "kernels/dataflow/stencil_reader.cpp";
constexpr const char* kComputeKernelPath = "kernels/compute/stencil_compute.cpp";
constexpr const char* kWriterKernelPath = "kernels/dataflow/stencil_writer.cpp";

[[nodiscard]] std::filesystem::path resolve(const std::filesystem::path& root, const char* rel) { return root / rel; }

/// Bit-cast a float to uint32_t for passing as compile-time arg.
[[nodiscard]] uint32_t float_to_u32(const float v) noexcept {
    uint32_t bits{};
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

}  // namespace

StencilDeviceProgram create_stencil_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& output_buffer,
    const StencilConfig& config) {
    TT_FATAL(config.k() >= 1 && config.k() <= 17, "stencil length K={} must be in [1,17]", config.k());
    TT_FATAL(config.element_size_bytes == sizeof(float), "only fp32 supported");
    TT_FATAL(config.stick_width == 32, "kernel expects 32 samples per stick");

    const auto page_size = config.aligned_stick_bytes(kAlignment);
    const auto tile_size = page_size * 32;  // 32 sticks per tile

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // ── Circular buffers ──
    // Halo input sticks: double-buffered, 32 sticks per tile
    auto halo_cb = tt::tt_metal::CircularBufferConfig(2 * tile_size, {{kCbHalo, tt::DataFormat::Float32}})
                       .set_page_size(kCbHalo, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, halo_cb);

    // Current input sticks: double-buffered
    auto cur_cb = tt::tt_metal::CircularBufferConfig(2 * tile_size, {{kCbCur, tt::DataFormat::Float32}})
                      .set_page_size(kCbCur, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cur_cb);

    // Tilized halo (1 tile)
    auto tilized_h_cb = tt::tt_metal::CircularBufferConfig(tile_size, {{kCbTilizedH, tt::DataFormat::Float32}})
                            .set_page_size(kCbTilizedH, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core, tilized_h_cb);

    // Tilized current (1 tile)
    auto tilized_c_cb = tt::tt_metal::CircularBufferConfig(tile_size, {{kCbTilizedC, tt::DataFormat::Float32}})
                            .set_page_size(kCbTilizedC, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core, tilized_c_cb);

    // Tilized output (1 tile)
    auto tilized_out_cb = tt::tt_metal::CircularBufferConfig(tile_size, {{kCbTilizedOut, tt::DataFormat::Float32}})
                              .set_page_size(kCbTilizedOut, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core, tilized_out_cb);

    // Output sticks: double-buffered
    auto out_cb = tt::tt_metal::CircularBufferConfig(2 * tile_size, {{kCbOut, tt::DataFormat::Float32}})
                      .set_page_size(kCbOut, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, out_cb);

    // DRAM read cache (1 stick)
    auto cache_cb = tt::tt_metal::CircularBufferConfig(page_size, {{kCbCache, tt::DataFormat::Float32}})
                        .set_page_size(kCbCache, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cache_cb);

    // ── Reader kernel ──
    std::vector<uint32_t> reader_compile_args = {kCbHalo, kCbCur, page_size, kCbCache};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_args);

    const auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        resolve(kernel_root, kReaderKernelPath),
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    // ── Compute kernel ──
    // Compile-time args: K, num_tiles, h_packed[0..K-1]
    std::vector<uint32_t> compute_compile_args;
    compute_compile_args.push_back(config.k());
    compute_compile_args.push_back(config.num_tiles());
    for (const auto& coeff : config.coefficients) {
        compute_compile_args.push_back(float_to_u32(coeff));
    }

    const auto compute_kernel = tt::tt_metal::CreateKernel(
        program,
        resolve(kernel_root, kComputeKernelPath),
        core,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .compile_args = compute_compile_args});

    // ── Writer kernel ──
    std::vector<uint32_t> writer_compile_args = {kCbOut, page_size};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_args);

    const auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        resolve(kernel_root, kWriterKernelPath),
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    // ── Runtime args ──
    // Reader: src_addr, signal_length, stencil_k, num_tiles
    tt::tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        std::array<uint32_t, 4>{
            static_cast<uint32_t>(input_buffer.address()), config.signal_length, config.k(), config.num_tiles()});

    // Writer: dst_addr, num_output_sticks
    const uint32_t num_output_sticks = config.num_tiles() * 32;  // 32 sticks per tile
    tt::tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        std::array<uint32_t, 2>{static_cast<uint32_t>(output_buffer.address()), num_output_sticks});

    return StencilDeviceProgram{
        .program = std::move(program),
        .reader = reader_kernel,
        .compute = compute_kernel,
        .writer = writer_kernel,
    };
}

std::vector<float> reference_stencil_1d(const std::vector<float>& signal, const std::vector<float>& coefficients) {
    const auto k = coefficients.size();
    const auto n = signal.size();

    if (k == 0 || n == 0) {
        return std::vector<float>(n, 0.0F);
    }

    // Direct convolution: g[i] = sum_{j=0}^{k-1} h[j] * f[i-j]
    // With zero-padding: f[i] = 0 for i < 0 or i >= n
    std::vector<float> result(n, 0.0F);
    for (size_t i = 0; i < n; i++) {
        float sum = 0.0F;
        for (size_t j = 0; j < k; j++) {
            if (i >= j) {
                sum += coefficients[j] * signal[i - j];
            }
        }
        result[i] = sum;
    }
    return result;
}

}  // namespace ttwv
