#include "tt_wavelet/include/pad_split/device.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <tt_stl/assert.hpp>
#include <vector>

#include "tt-metalium/tensor_accessor_args.hpp"

namespace ttwv {

namespace {

constexpr uint32_t kCbIdEven = tt::CBIndex::c_0;
constexpr uint32_t kCbIdOdd = tt::CBIndex::c_1;
constexpr uint32_t kCbIdCache = tt::CBIndex::c_2;
constexpr uint32_t kCircularBufferAlignmentBytes = 32;
constexpr uint32_t kSourceCacheStickCount = 8;

constexpr const char* kReaderKernelPath = "kernels/dataflow/pad_split_1d_reader.cpp";
constexpr const char* kWriterKernelPath = "kernels/dataflow/pad_split_1d_writer.cpp";

[[nodiscard]] std::filesystem::path reader_kernel_path(const std::filesystem::path& kernel_root) {
    return kernel_root / kReaderKernelPath;
}

[[nodiscard]] std::filesystem::path writer_kernel_path(const std::filesystem::path& kernel_root) {
    return kernel_root / kWriterKernelPath;
}

[[nodiscard]] uint32_t checked_stick_count(const size_t value, const char* label) {
    TT_FATAL(
        value <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()), "{} {} overflows uint32_t", label, value);
    return static_cast<uint32_t>(value);
}

[[nodiscard]] uint32_t at_least_one_page(const uint32_t page_count) { return page_count == 0 ? 1 : page_count; }

[[nodiscard]] bool is_single_core_l1_buffer(const tt::tt_metal::Buffer& buffer) {
    return buffer.is_l1() && buffer.has_shard_spec() && buffer.num_cores().value_or(0) == 1;
}

}  // namespace

PadSplit1DDeviceProgram create_pad_split_1d_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& even_buffer,
    const tt::tt_metal::Buffer& odd_buffer,
    const PadSplit1DLayout& layout) {
    TT_FATAL(layout.pad_config.mode == BoundaryMode::kSymmetric, "currently supports symmetric mode only");
    TT_FATAL(layout.input.element_size_bytes == sizeof(float), "is fp32-only");
    TT_FATAL(layout.input.stick_width == 32, "kernel expects 32 scalar samples per stick");

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    const auto page_size = layout.input.aligned_stick_bytes(kCircularBufferAlignmentBytes);
    const bool direct_l1_output = is_single_core_l1_buffer(even_buffer) && is_single_core_l1_buffer(odd_buffer);
    TT_FATAL(
        even_buffer.is_l1() == odd_buffer.is_l1(),
        "pad_split even/odd outputs must both be L1-backed or both use the writer path");

    const uint32_t even_cb_pages =
        direct_l1_output ? at_least_one_page(checked_stick_count(layout.even_stick_count(), "even sticks")) : 2;
    const uint32_t odd_cb_pages =
        direct_l1_output ? at_least_one_page(checked_stick_count(layout.odd_stick_count(), "odd sticks")) : 2;

    auto even_cb_config =
        tt::tt_metal::CircularBufferConfig(even_cb_pages * page_size, {{kCbIdEven, tt::DataFormat::Float32}})
            .set_page_size(kCbIdEven, page_size);
    if (direct_l1_output) {
        even_cb_config.set_globally_allocated_address(even_buffer);
    }
    tt::tt_metal::CreateCircularBuffer(program, core, even_cb_config);

    auto odd_cb_config =
        tt::tt_metal::CircularBufferConfig(odd_cb_pages * page_size, {{kCbIdOdd, tt::DataFormat::Float32}})
            .set_page_size(kCbIdOdd, page_size);
    if (direct_l1_output) {
        odd_cb_config.set_globally_allocated_address(odd_buffer);
    }
    tt::tt_metal::CreateCircularBuffer(program, core, odd_cb_config);

    const auto cache_cb_config =
        tt::tt_metal::CircularBufferConfig(kSourceCacheStickCount * page_size, {{kCbIdCache, tt::DataFormat::Float32}})
            .set_page_size(kCbIdCache, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cache_cb_config);

    // Reader compile args: cb_even, cb_odd, stick_nbytes, cb_cache, stick_width, cache_stick_count.
    const uint32_t stick_width = page_size / sizeof(float);
    std::vector<uint32_t> reader_compile_args = {
        kCbIdEven, kCbIdOdd, page_size, kCbIdCache, stick_width, kSourceCacheStickCount};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_args);

    const auto reader_kernel = tt::tt_metal::CreateKernel(
        program, reader_kernel_path(kernel_root), core, tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    std::optional<tt::tt_metal::KernelHandle> writer_kernel;
    if (!direct_l1_output) {
        // Writer compile args: cb_even, cb_odd, stick_nbytes
        // The writer kernel consumes TensorAccessorArgs starting at CTA index 3.
        // Do not append extra compile-time args before the accessor payload.
        std::vector<uint32_t> writer_compile_args = {kCbIdEven, kCbIdOdd, page_size};
        tt::tt_metal::TensorAccessorArgs(even_buffer).append_to(writer_compile_args);
        tt::tt_metal::TensorAccessorArgs(odd_buffer).append_to(writer_compile_args);
        writer_kernel = tt::tt_metal::CreateKernel(
            program,
            writer_kernel_path(kernel_root),
            core,
            tt::tt_metal::WriterDataMovementConfig(writer_compile_args));
    }

    PadSplit1DDeviceProgram bundle{
        .program = std::move(program),
        .reader = reader_kernel,
        .writer = writer_kernel,
        .writes_direct_to_l1 = direct_l1_output,
    };

    set_pad_split_1d_runtime_args(bundle, core, input_buffer, even_buffer, odd_buffer, layout);

    return bundle;
}

void set_pad_split_1d_runtime_args(
    const PadSplit1DDeviceProgram& program_bundle,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& even_buffer,
    const tt::tt_metal::Buffer& odd_buffer,
    const PadSplit1DLayout& layout) {
    TT_FATAL(
        layout.input.length <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
        "input.length ({}) overflows uint32_t",
        layout.input.length);

    // Reader runtime args
    tt::tt_metal::SetRuntimeArgs(
        program_bundle.program,
        program_bundle.reader,
        core,
        std::array<uint32_t, 4>{
            static_cast<uint32_t>(input_buffer.address()),
            static_cast<uint32_t>(layout.input.length),
            static_cast<uint32_t>(layout.padded_length()),
            layout.pad_config.left,
        });

    if (program_bundle.writer.has_value()) {
        tt::tt_metal::SetRuntimeArgs(
            program_bundle.program,
            *program_bundle.writer,
            core,
            std::array<uint32_t, 4>{
                static_cast<uint32_t>(even_buffer.address()),
                static_cast<uint32_t>(odd_buffer.address()),
                static_cast<uint32_t>(layout.even_stick_count()),
                static_cast<uint32_t>(layout.odd_stick_count()),
            });
    }
}

}  // namespace ttwv
