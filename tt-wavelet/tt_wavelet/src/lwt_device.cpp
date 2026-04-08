#include "tt_wavelet/include/lwt_device.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <tt_stl/assert.hpp>

#include "tt-metalium/tensor_accessor_args.hpp"

namespace ttwv::lwt {

namespace {

// Circular-buffer indices
constexpr uint32_t kCbHalo  = tt::CBIndex::c_0;  // reader → compute: halo sticks
constexpr uint32_t kCbCur   = tt::CBIndex::c_1;  // reader → compute: current sticks
constexpr uint32_t kCbOut   = tt::CBIndex::c_16; // compute → writer: output sticks
constexpr uint32_t kCbCache = tt::CBIndex::c_5;  // reader DRAM cache

constexpr uint32_t kAlignment = 32;  // NOC minimum transfer alignment (bytes)

constexpr const char* kReaderPath = "kernels/dataflow/lwt_reader.cpp";
constexpr const char* kWriterPath = "kernels/dataflow/lwt_writer.cpp";

}  // namespace

// ---------------------------------------------------------------------------

LwtProgram create_lwt_step_program(
    const std::filesystem::path&    kernel_root,
    const tt::tt_metal::CoreCoord&  core,
    const tt::tt_metal::Buffer&     input_buffer,
    const tt::tt_metal::Buffer&     output_buffer,
    const LwtStepConfig&            config) {

    TT_FATAL(config.stick_width == 32, "fp32 path requires stick_width == 32");
    TT_FATAL(config.element_size_bytes == sizeof(float), "fp32-only path");

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const uint32_t page_size = config.aligned_stick_bytes(kAlignment);

    // halo CB (2 pages so reader can double-buffer)
    const auto halo_cb_cfg =
        tt::tt_metal::CircularBufferConfig(2 * page_size, {{kCbHalo, tt::DataFormat::Float32}})
            .set_page_size(kCbHalo, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, halo_cb_cfg);

    // current CB (2 pages)
    const auto cur_cb_cfg =
        tt::tt_metal::CircularBufferConfig(2 * page_size, {{kCbCur, tt::DataFormat::Float32}})
            .set_page_size(kCbCur, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cur_cb_cfg);

    // output CB — the writer drains from here.
    // For now (no compute kernel) we route the output directly from the reader
    // by aliasing kCbOut = kCbCur in the writer; but to keep the indices
    // clean we allocate a separate CB that main.cpp can wire up as needed.
    const auto out_cb_cfg =
        tt::tt_metal::CircularBufferConfig(2 * page_size, {{kCbOut, tt::DataFormat::Float32}})
            .set_page_size(kCbOut, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, out_cb_cfg);

    // DRAM cache for the reader (one stick)
    const auto cache_cb_cfg =
        tt::tt_metal::CircularBufferConfig(page_size, {{kCbCache, tt::DataFormat::Float32}})
            .set_page_size(kCbCache, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cache_cb_cfg);

    // ---- reader kernel ----
    // Compile-time args: cb_halo, cb_cur, stick_nbytes, cb_cache, stick_width,
    //                    TensorAccessorArgs(input)
    std::vector<uint32_t> reader_ct_args = {
        kCbHalo, kCbCur, page_size, kCbCache, config.stick_width};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_ct_args);

    const auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        kernel_root / kReaderPath,
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    // ---- writer kernel ----
    // Compile-time args: cb_id_out, stick_nbytes, TensorAccessorArgs(output)
    std::vector<uint32_t> writer_ct_args = {kCbCur, page_size};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_ct_args);

    const auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        kernel_root / kWriterPath,
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    LwtProgram bundle{
        .program = std::move(program),
        .reader  = reader_kernel,
        .writer  = writer_kernel,
    };

    set_lwt_step_runtime_args(bundle, core, input_buffer, output_buffer, config);

    return bundle;
}

// ---------------------------------------------------------------------------

void set_lwt_step_runtime_args(
    const LwtProgram&               program_bundle,
    const tt::tt_metal::CoreCoord&  core,
    const tt::tt_metal::Buffer&     input_buffer,
    const tt::tt_metal::Buffer&     output_buffer,
    const LwtStepConfig&            config) {

    // Reader runtime args (mirror LwtReaderConfig fields, see lwt_reader.cpp)
    //   0: src_addr
    //   1: input_length
    //   2: padded_length
    //   3: left_pad
    //   4: split_phase
    //   5: source_offset
    //   6: stencil_k
    //   7: num_tiles
    //   8: is_pad_split
    tt::tt_metal::SetRuntimeArgs(
        program_bundle.program,
        program_bundle.reader,
        core,
        std::array<uint32_t, 9>{
            static_cast<uint32_t>(input_buffer.address()),
            config.is_pad_split ? config.input_length   : config.signal_length,
            config.is_pad_split ? config.padded_length  : config.signal_length,
            config.is_pad_split ? config.left_pad       : 0u,
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