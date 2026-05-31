#include "tt_wavelet/include/pad_split/device.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <tt_stl/assert.hpp>
#include <vector>

#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt_wavelet/include/device_protocol/lwt_config.hpp"

namespace ttwv {

namespace {

constexpr uint32_t kCbIdEven = tt::CBIndex::c_0;
constexpr uint32_t kCbIdOdd = tt::CBIndex::c_1;
constexpr uint32_t kCbIdCache = tt::CBIndex::c_2;

constexpr const char* kReaderKernelPath = "kernels/dataflow/pad_split_1d_reader.cpp";
constexpr const char* kWriterKernelPath = "kernels/dataflow/pad_split_1d_writer.cpp";
constexpr const char* kPadSplitCoreCountEnv = "TT_WAVELET_PAD_SPLIT_CORES";

[[nodiscard]] std::filesystem::path kernel_path(const std::filesystem::path& kernel_root, const char* relative_path) {
    return kernel_root / relative_path;
}

[[nodiscard]] uint32_t ceil_div(const uint32_t value, const uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

[[nodiscard]] uint32_t pad_split_core_count(const tt::tt_metal::distributed::MeshDevice& mesh_device) {
    const auto grid = mesh_device.compute_with_storage_grid_size();
    const uint32_t hw_cores = static_cast<uint32_t>(grid.x * grid.y);
    TT_FATAL(hw_cores > 0, "pad split requires at least one compute core");

    const char* raw = std::getenv(kPadSplitCoreCountEnv);
    if (raw != nullptr && raw[0] != '\0') {
        char* end = nullptr;
        const unsigned long requested_cores = std::strtoul(raw, &end, 10);
        if (end != raw && *end == '\0' && requested_cores > 0 && requested_cores <= hw_cores) {
            return static_cast<uint32_t>(requested_cores);
        }
    }

    return hw_cores;
}

struct CoreChunk {
    uint32_t padded_begin;
    uint32_t padded_end;
    uint32_t even_stick_begin;
    uint32_t even_stick_count;
    uint32_t odd_stick_begin;
    uint32_t odd_stick_count;
};

[[nodiscard]] std::vector<CoreChunk> partition_output(
    const uint32_t padded_length, const uint32_t stick_width, const uint32_t core_count) {
    const uint32_t chunk_alignment = 2 * stick_width;
    const uint32_t total_chunks = padded_length / chunk_alignment;
    const uint32_t remainder = padded_length % chunk_alignment;
    const uint32_t active_cores = std::min(core_count, std::max(total_chunks, 1U));

    std::vector<CoreChunk> chunks;
    chunks.reserve(active_cores);

    const uint32_t base_chunks_per_core = total_chunks / active_cores;
    const uint32_t extra_chunks = total_chunks % active_cores;

    uint32_t padded_offset = 0;
    uint32_t even_stick_offset = 0;
    uint32_t odd_stick_offset = 0;

    for (uint32_t c = 0; c < active_cores; ++c) {
        const uint32_t chunk_count = base_chunks_per_core + (c < extra_chunks ? 1U : 0U);
        const uint32_t chunk_begin = padded_offset;
        uint32_t chunk_end = padded_offset + chunk_count * chunk_alignment;

        if (c == active_cores - 1) {
            chunk_end += remainder;
        }

        const uint32_t chunk_length = chunk_end - chunk_begin;
        const uint32_t even_stick_count = ceil_div((chunk_length + 1) / 2, stick_width);
        const uint32_t odd_stick_count = ceil_div(chunk_length / 2, stick_width);

        chunks.push_back(
            CoreChunk{
                .padded_begin = chunk_begin,
                .padded_end = chunk_end,
                .even_stick_begin = even_stick_offset,
                .even_stick_count = even_stick_count,
                .odd_stick_begin = odd_stick_offset,
                .odd_stick_count = odd_stick_count,
            });

        padded_offset = chunk_end;
        even_stick_offset += even_stick_count;
        odd_stick_offset += odd_stick_count;
    }

    return chunks;
}

}  // namespace

PadSplit1DDeviceProgram create_pad_split_1d_program(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
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
    TT_FATAL(
        layout.padded_length() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
        "padded_length ({}) overflows uint32_t",
        layout.padded_length());

    const uint32_t padded_length = static_cast<uint32_t>(layout.padded_length());
    const uint32_t requested_cores = pad_split_core_count(mesh_device);
    const std::vector<CoreChunk> chunks = partition_output(padded_length, kStickWidth, requested_cores);
    const uint32_t active_cores = static_cast<uint32_t>(chunks.size());

    const auto grid = mesh_device.compute_with_storage_grid_size();
    const auto cores =
        tt::tt_metal::grid_to_cores(active_cores, static_cast<uint32_t>(grid.x), static_cast<uint32_t>(grid.y), true);

    std::vector<tt::tt_metal::CoreRange> ranges;
    ranges.reserve(cores.size());
    for (const auto& core : cores) {
        ranges.emplace_back(core);
    }
    const auto core_range_set = tt::tt_metal::CoreRangeSet(std::move(ranges)).merge_ranges();

    tt::tt_metal::Program program{tt::tt_metal::CreateProgram()};
    constexpr uint32_t page_size{device_protocol::kStickBytes};

    const auto even_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * page_size, {{kCbIdEven, tt::DataFormat::Float32}})
            .set_page_size(kCbIdEven, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, even_cb_config);

    const auto odd_cb_config = tt::tt_metal::CircularBufferConfig(2 * page_size, {{kCbIdOdd, tt::DataFormat::Float32}})
                                   .set_page_size(kCbIdOdd, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, odd_cb_config);

    const auto cache_cb_config =
        tt::tt_metal::CircularBufferConfig(
            device_protocol::kPadSplitCacheStickCount * page_size, {{kCbIdCache, tt::DataFormat::Float32}})
            .set_page_size(kCbIdCache, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cache_cb_config);

    std::vector<uint32_t> reader_compile_args;
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_args);

    const auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kReaderKernelPath),
        core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(
            reader_compile_args, {}, {{"cb_even", kCbIdEven}, {"cb_odd", kCbIdOdd}, {"cb_cache", kCbIdCache}}));

    std::vector<uint32_t> writer_compile_args;
    tt::tt_metal::TensorAccessorArgs(even_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(odd_buffer).append_to(writer_compile_args);
    const auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kWriterKernelPath),
        core_range_set,
        tt::tt_metal::WriterDataMovementConfig(
            writer_compile_args, {}, {{"cb_even", kCbIdEven}, {"cb_odd", kCbIdOdd}}));

    for (uint32_t c = 0; c < active_cores; ++c) {
        const auto& chunk = chunks[c];

        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_kernel,
            cores[c],
            std::array<uint32_t, 6>{
                static_cast<uint32_t>(input_buffer.address()),
                static_cast<uint32_t>(layout.input.length),
                padded_length,
                layout.pad_config.left,
                chunk.padded_begin,
                chunk.padded_end,
            });

        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_kernel,
            cores[c],
            std::array<uint32_t, 6>{
                static_cast<uint32_t>(even_buffer.address()),
                static_cast<uint32_t>(odd_buffer.address()),
                chunk.even_stick_count,
                chunk.odd_stick_count,
                chunk.even_stick_begin,
                chunk.odd_stick_begin,
            });
    }

    PadSplit1DDeviceProgram bundle{
        .program = std::move(program),
        .reader = reader_kernel,
        .writer = writer_kernel,
    };

    return bundle;
}

}  // namespace ttwv
