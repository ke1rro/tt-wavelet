#include "tt_wavelet/include/padding_device.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <tt_stl/assert.hpp>
#include <vector>

#include "tt-metalium/tensor_accessor_args.hpp"

namespace ttwv {

namespace {

constexpr uint32_t kOutputCbId = tt::CBIndex::c_0;
constexpr uint32_t kCacheCbId = tt::CBIndex::c_1;
constexpr uint32_t kCircularBufferAlignmentBytes = 32;
constexpr const char* kSymmetricReaderKernelPath = "kernels/dataflow/pad_symmetric_1d_reader.cpp";
constexpr const char* kWriterKernelPath = "kernels/dataflow/pad_1d_writer.cpp";

[[nodiscard]] std::filesystem::path reader_kernel_path(const std::filesystem::path& kernel_root) {
    return kernel_root / kSymmetricReaderKernelPath;
}

[[nodiscard]] std::filesystem::path writer_kernel_path(const std::filesystem::path& kernel_root) {
    return kernel_root / kWriterKernelPath;
}

}  // namespace

Pad1DDeviceProgram create_symmetric_pad_1d_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& output_buffer,
    const Pad1DLayout& layout) {
    TT_FATAL(layout.config.mode == BoundaryMode::Symmetric, "Padding kernel currently supports symmetric mode only");
    TT_FATAL(layout.input.element_size_bytes == sizeof(float), "Padding kernel is fp32-only in this first pass");
    TT_FATAL(layout.output.element_size_bytes == sizeof(float), "Output buffer must be fp32");
    TT_FATAL(layout.input.stick_width == 32, "Padding kernel expects 32 scalar samples per stick");
    TT_FATAL(layout.output.stick_width == layout.input.stick_width, "Input and output stick widths must match");

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    const auto page_size = layout.output.aligned_stick_bytes(kCircularBufferAlignmentBytes);

    const auto output_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * page_size, {{kOutputCbId, tt::DataFormat::Float32}})
            .set_page_size(kOutputCbId, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, output_cb_config);

    const auto cache_cb_config = tt::tt_metal::CircularBufferConfig(page_size, {{kCacheCbId, tt::DataFormat::Float32}})
                                     .set_page_size(kCacheCbId, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cache_cb_config);

    std::vector<uint32_t> reader_compile_args = {kOutputCbId, page_size, kCacheCbId};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_args);

    std::vector<uint32_t> writer_compile_args = {kOutputCbId, page_size};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_args);

    const auto reader_kernel = tt::tt_metal::CreateKernel(
        program, reader_kernel_path(kernel_root), core, tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));
    const auto writer_kernel = tt::tt_metal::CreateKernel(
        program, writer_kernel_path(kernel_root), core, tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    Pad1DDeviceProgram bundle{
        .program = std::move(program),
        .reader_kernel = reader_kernel,
        .writer_kernel = writer_kernel,
    };

    set_pad_1d_runtime_args(bundle, core, input_buffer, output_buffer, layout);

    return bundle;
}

void set_pad_1d_runtime_args(
    const Pad1DDeviceProgram& program_bundle,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& output_buffer,
    const Pad1DLayout& layout) {
    TT_FATAL(
        layout.input.length <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
        "input.length ({}) overflows uint32_t kernel receives truncated value",
        layout.input.length);
    TT_FATAL(
        layout.output.length <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
        "output.length ({}) overflows uint32_t kernel receives truncated value",
        layout.output.length);

    const auto total_output_sticks = static_cast<uint32_t>(layout.output_stick_count());

    tt::tt_metal::SetRuntimeArgs(
        program_bundle.program,
        program_bundle.reader_kernel,
        core,
        std::array<uint32_t, 6>{
            static_cast<uint32_t>(input_buffer.address()),
            static_cast<uint32_t>(layout.input.length),
            static_cast<uint32_t>(layout.output.length),
            layout.config.left,
            total_output_sticks,
            0U,
        });

    tt::tt_metal::SetRuntimeArgs(
        program_bundle.program,
        program_bundle.writer_kernel,
        core,
        std::array<uint32_t, 3>{
            static_cast<uint32_t>(output_buffer.address()),
            total_output_sticks,
            0U,
        });
}

}  // namespace ttwv
