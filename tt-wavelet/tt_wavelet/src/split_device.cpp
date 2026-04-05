#include "tt_wavelet/include/split_device.hpp"

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

constexpr uint32_t kCbIdIn = tt::CBIndex::c_0;
constexpr uint32_t kCbIdEven = tt::CBIndex::c_16;
constexpr uint32_t kCbIdOdd = tt::CBIndex::c_17;
constexpr uint32_t kCircularBufferAlignmentBytes = 32;

constexpr const char* kReaderKernelPath = "kernels/dataflow/split_1d_reader.cpp";
constexpr const char* kWriterKernelPath = "kernels/dataflow/split_1d_writer.cpp";

[[nodiscard]] std::filesystem::path reader_kernel_path(const std::filesystem::path& kernel_root) {
    return kernel_root / kReaderKernelPath;
}

[[nodiscard]] std::filesystem::path writer_kernel_path(const std::filesystem::path& kernel_root) {
    return kernel_root / kWriterKernelPath;
}

}  // namespace

Split1DDeviceProgram create_split_1d_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& even_buffer,
    const tt::tt_metal::Buffer& odd_buffer,
    const Split1DLayout& layout) {

    TT_FATAL(layout.input.element_size_bytes == sizeof(float), "Split kernel is fp32-only");
    TT_FATAL(layout.output.even.element_size_bytes == sizeof(float), "Even buffer must be fp32");
    TT_FATAL(layout.output.odd.element_size_bytes == sizeof(float), "Odd buffer must be fp32");
    TT_FATAL(layout.input.stick_width == 32, "Split kernel expects 32 scalar samples per stick");

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    const auto page_size = layout.input.aligned_stick_bytes(kCircularBufferAlignmentBytes);

    // CB for interleaved input signal
    const auto in_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * page_size, {{kCbIdIn, tt::DataFormat::Float32}})
            .set_page_size(kCbIdIn, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, in_cb_config);

    // CBs for extracted even and odd signals
    const auto even_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * page_size, {{kCbIdEven, tt::DataFormat::Float32}})
            .set_page_size(kCbIdEven, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, even_cb_config);

    const auto odd_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * page_size, {{kCbIdOdd, tt::DataFormat::Float32}})
            .set_page_size(kCbIdOdd, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, odd_cb_config);

    std::vector<uint32_t> reader_compile_args = {kCbIdIn, page_size};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_args);

    std::vector<uint32_t> writer_compile_args = {kCbIdIn, kCbIdEven, kCbIdOdd, page_size};
    tt::tt_metal::TensorAccessorArgs(even_buffer).append_to(writer_compile_args);

    const auto reader_kernel = tt::tt_metal::CreateKernel(
        program, reader_kernel_path(kernel_root), core, tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));
    const auto writer_kernel = tt::tt_metal::CreateKernel(
        program, writer_kernel_path(kernel_root), core, tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    Split1DDeviceProgram bundle{
        .program = std::move(program),
        .reader_kernel = reader_kernel,
        .writer_kernel = writer_kernel,
    };

    set_split_1d_runtime_args(bundle, core, input_buffer, even_buffer, odd_buffer, layout);

    return bundle;
}

void set_split_1d_runtime_args(
    const Split1DDeviceProgram& program_bundle,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& even_buffer,
    const tt::tt_metal::Buffer& odd_buffer,
    const Split1DLayout& layout) {

    TT_FATAL(
        layout.input.length <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
        "input.length ({}) overflows uint32_t kernel receives truncated value",
        layout.input.length);

    tt::tt_metal::SetRuntimeArgs(
        program_bundle.program,
        program_bundle.reader_kernel,
        core,
        std::array<uint32_t, 3>{
            static_cast<uint32_t>(input_buffer.address()),
            static_cast<uint32_t>(layout.input.stick_count()),
            0U  // start stick
        });

    tt::tt_metal::SetRuntimeArgs(
        program_bundle.program,
        program_bundle.writer_kernel,
        core,
        std::array<uint32_t, 7>{
            static_cast<uint32_t>(even_buffer.address()),
            static_cast<uint32_t>(odd_buffer.address()),
            static_cast<uint32_t>(layout.input.length),
            static_cast<uint32_t>(layout.even_stick_count()),
            static_cast<uint32_t>(layout.odd_stick_count()),
            0U, // even start stick
            0U  // odd start stick
        });
}

}  // namespace ttwv
