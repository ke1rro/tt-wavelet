#include "tt_wavelet/include/pad_split/device.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <tt_stl/assert.hpp>
#include <vector>

#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt_wavelet/include/device_protocol/lwt_config.hpp"

namespace ttwv {

namespace {

constexpr uint32_t kCbIdEven = tt::CBIndex::c_0;
constexpr uint32_t kCbIdOdd = tt::CBIndex::c_1;
constexpr uint32_t kCbIdCache = tt::CBIndex::c_2;

constexpr const char* kReaderKernelPath = "kernels/dataflow/pad_split_1d_reader.cpp";
constexpr const char* kWriterKernelPath = "kernels/dataflow/pad_split_1d_writer.cpp";

[[nodiscard]] std::filesystem::path kernel_path(const std::filesystem::path& kernel_root, const char* relative_path) {
    return kernel_root / relative_path;
}

}  // namespace

PadSplit1DDeviceProgram create_pad_split_1d_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& even_buffer,
    const tt::tt_metal::Buffer& odd_buffer,
    const PadSplit1DLayout& layout) {
    TT_FATAL(layout.pad_config.mode == BoundaryMode::kSymmetric, "pad split currently supports symmetric mode only");
    TT_FATAL(layout.input.element_size_bytes == sizeof(float), "pad split supports fp32 only");
    TT_FATAL(layout.input.stick_width == kStickWidth, "kernel expects 32 scalar samples per stick");
    TT_FATAL(
        layout.input.length <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
        "input.length ({}) overflows uint32_t",
        layout.input.length);

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    constexpr uint32_t page_size = device_protocol::kStickBytes;

    const auto even_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * page_size, {{kCbIdEven, tt::DataFormat::Float32}})
            .set_page_size(kCbIdEven, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, even_cb_config);

    const auto odd_cb_config = tt::tt_metal::CircularBufferConfig(2 * page_size, {{kCbIdOdd, tt::DataFormat::Float32}})
                                   .set_page_size(kCbIdOdd, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, odd_cb_config);

    const auto cache_cb_config =
        tt::tt_metal::CircularBufferConfig(
            device_protocol::kPadSplitCacheStickCount * page_size, {{kCbIdCache, tt::DataFormat::Float32}})
            .set_page_size(kCbIdCache, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cache_cb_config);

    std::vector<uint32_t> reader_compile_args = {kCbIdEven, kCbIdOdd, kCbIdCache};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_args);

    const auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kReaderKernelPath),
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    std::vector<uint32_t> writer_compile_args = {kCbIdEven, kCbIdOdd};
    tt::tt_metal::TensorAccessorArgs(even_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(odd_buffer).append_to(writer_compile_args);
    const auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kWriterKernelPath),
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    PadSplit1DDeviceProgram bundle{
        .program = std::move(program),
        .reader = reader_kernel,
        .writer = writer_kernel,
    };

    tt::tt_metal::SetRuntimeArgs(
        bundle.program,
        bundle.reader,
        core,
        std::array<uint32_t, 4>{
            static_cast<uint32_t>(input_buffer.address()),
            static_cast<uint32_t>(layout.input.length),
            static_cast<uint32_t>(layout.padded_length()),
            layout.pad_config.left,
        });

    tt::tt_metal::SetRuntimeArgs(
        bundle.program,
        bundle.writer,
        core,
        std::array<uint32_t, 4>{
            static_cast<uint32_t>(even_buffer.address()),
            static_cast<uint32_t>(odd_buffer.address()),
            static_cast<uint32_t>(layout.even_stick_count()),
            static_cast<uint32_t>(layout.odd_stick_count()),
        });

    return bundle;
}

}  // namespace ttwv
