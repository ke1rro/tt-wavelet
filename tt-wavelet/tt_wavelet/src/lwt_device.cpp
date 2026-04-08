#include "tt_wavelet/include/lwt_device.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <tt_stl/assert.hpp>
#include <vector>

#include "tt-metalium/tensor_accessor_args.hpp"

namespace ttwv::lwt {

namespace {

constexpr uint32_t kCbHalo = tt::CBIndex::c_0;   // reader -> compute: halo sticks
constexpr uint32_t kCbCur = tt::CBIndex::c_1;    // reader -> compute: current sticks
constexpr uint32_t kCbOut = tt::CBIndex::c_16;   // compute -> writer: output sticks
constexpr uint32_t kCbCache = tt::CBIndex::c_5;  // reader DRAM cache

constexpr uint32_t kAlignment = 32;  // NOC minimum transfer alignment (bytes)

constexpr const char* kReaderPath = "kernels/dataflow/lwt_reader.cpp";
constexpr const char* kWriterPath = "kernels/dataflow/lwt_writer.cpp";

}  // namespace

LwtProgram create_lwt_step_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& output_buffer,
    const LwtStepConfig& config) {
    TT_FATAL(config.stick_width == 32, "fp32 path requires stick_width == 32");
    TT_FATAL(config.element_size_bytes == sizeof(float), "fp32-only path");

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const uint32_t page_size = config.aligned_stick_bytes(kAlignment);

    const auto halo_cb_cfg = tt::tt_metal::CircularBufferConfig(2 * page_size, {{kCbHalo, tt::DataFormat::Float32}})
                                 .set_page_size(kCbHalo, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, halo_cb_cfg);

    const auto cur_cb_cfg = tt::tt_metal::CircularBufferConfig(2 * page_size, {{kCbCur, tt::DataFormat::Float32}})
                                .set_page_size(kCbCur, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cur_cb_cfg);

    const auto out_cb_cfg = tt::tt_metal::CircularBufferConfig(2 * page_size, {{kCbOut, tt::DataFormat::Float32}})
                                .set_page_size(kCbOut, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, out_cb_cfg);

    const auto cache_cb_cfg = tt::tt_metal::CircularBufferConfig(page_size, {{kCbCache, tt::DataFormat::Float32}})
                                  .set_page_size(kCbCache, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cache_cb_cfg);

    std::vector<uint32_t> reader_ct_args = {kCbHalo, kCbCur, page_size, kCbCache, config.stick_width};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_ct_args);

    const auto reader_kernel = tt::tt_metal::CreateKernel(
        program, kernel_root / kReaderPath, core, tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    std::vector<uint32_t> writer_ct_args = {kCbCur, page_size};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_ct_args);

    const auto writer_kernel = tt::tt_metal::CreateKernel(
        program, kernel_root / kWriterPath, core, tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    LwtProgram bundle{
        .program = std::move(program),
        .reader = reader_kernel,
        .writer = writer_kernel,
    };

    set_lwt_step_runtime_args(bundle, core, input_buffer, output_buffer, config);

    return bundle;
}

void set_lwt_step_runtime_args(
    const LwtProgram& program_bundle,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& output_buffer,
    const LwtStepConfig& config) {
    tt::tt_metal::SetRuntimeArgs(
        program_bundle.program,
        program_bundle.reader,
        core,
        std::array<uint32_t, 9>{
            static_cast<uint32_t>(input_buffer.address()),
            config.is_pad_split ? config.input_length : config.signal_length,
            config.is_pad_split ? config.padded_length : config.signal_length,
            config.is_pad_split ? config.left_pad : 0u,
            config.split_phase,
            static_cast<uint32_t>(static_cast<int32_t>(config.source_offset)),
            config.stencil_k,
            config.num_tiles,
            config.is_pad_split ? 1u : 0u,
        });

    // Writer runtime args
    //   0: dst_addr
    //   1: stick_count
    tt::tt_metal::SetRuntimeArgs(
        program_bundle.program,
        program_bundle.writer,
        core,
        std::array<uint32_t, 2>{
            static_cast<uint32_t>(output_buffer.address()),
            config.output_stick_count(),
        });
}

}  // namespace ttwv::lwt