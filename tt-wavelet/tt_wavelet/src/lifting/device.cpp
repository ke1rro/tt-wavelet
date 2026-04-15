#include "tt_wavelet/include/lifting/device.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <tt_stl/assert.hpp>
#include <vector>

#include "tracy/Tracy.hpp"

#include "tt-metalium/circular_buffer_constants.h"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"

namespace ttwv {

namespace {

constexpr uint32_t kNocAlignmentBytes = 32;
constexpr uint32_t kCircularBufferAlignmentBytes = 32;
constexpr tt::DataFormat kDataFormat = tt::DataFormat::Float32;

constexpr uint32_t kStencilInput0Cb = tt::CBIndex::c_0;
constexpr uint32_t kStencilInput1Cb = tt::CBIndex::c_1;
constexpr uint32_t kStencilBaseCb = tt::CBIndex::c_2;
constexpr uint32_t kStencilOutputCb = tt::CBIndex::c_16;
constexpr uint32_t kStencilSrcCacheCb = tt::CBIndex::c_4;
constexpr uint32_t kStencilBaseCacheCb = tt::CBIndex::c_5;

constexpr uint32_t kScaleInputCb = tt::CBIndex::c_0;
constexpr uint32_t kScaleCoeffCb = tt::CBIndex::c_1;
constexpr uint32_t kScaleOutputCb = tt::CBIndex::c_16;
constexpr uint32_t kScaleSrcCacheCb = tt::CBIndex::c_2;

constexpr const char* kScaleReaderKernelPath = "kernels/dataflow/lwt_scale_reader.cpp";
constexpr const char* kRowMajorWriterKernelPath = "kernels/dataflow/lwt_row_major_writer.cpp";
constexpr const char* kMulComputeKernelPath = "kernels/compute/lwt_mul_compute.cpp";

constexpr const char* kLwtReaderKernelPath = "kernels/dataflow/lwt_reader.cpp";
constexpr const char* kLwtComputeKernelPath = "kernels/compute/lwt_compute.cpp";
constexpr const char* kLwtWriterKernelPath = "kernels/dataflow/lwt_writer.cpp";

constexpr uint32_t kLwtSrcTile0Cb = tt::CBIndex::c_0;
constexpr uint32_t kLwtSrcTile1Cb = tt::CBIndex::c_1;
constexpr uint32_t kLwtBaseTileCb = tt::CBIndex::c_2;
constexpr uint32_t kLwtOutputCb = tt::CBIndex::c_16;
constexpr uint32_t kLwtSrcCacheCb = tt::CBIndex::c_3;
constexpr uint32_t kLwtBaseCacheCb = tt::CBIndex::c_4;
constexpr uint32_t kLwtSyncCb = tt::CBIndex::c_5;
constexpr uint32_t kLwtSyncPageSize = 32;

struct ScaleStepProgram {
    tt::tt_metal::Program program;
    tt::tt_metal::KernelHandle reader{};
    tt::tt_metal::KernelHandle compute{};
    tt::tt_metal::KernelHandle writer{};
};

struct PredictUpdateProgram {
    tt::tt_metal::Program program;
    tt::tt_metal::KernelHandle reader{};
    tt::tt_metal::KernelHandle compute{};
    tt::tt_metal::KernelHandle writer{};
};

[[nodiscard]] std::filesystem::path kernel_path(const std::filesystem::path& kernel_root, const char* relative_path) {
    return kernel_root / relative_path;
}

[[nodiscard]] SignalBuffer with_address(const SignalBuffer& buffer, const uint64_t dram_address) {
    SignalBuffer out = buffer;
    out.dram_address = dram_address;
    return out;
}

[[nodiscard]] std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_signal_mesh_buffer(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const SignalBuffer& buffer) {
    const tt::tt_metal::distributed::DeviceLocalBufferConfig local_config{
        .page_size = buffer.aligned_stick_bytes(kNocAlignmentBytes),
        .buffer_type = tt::tt_metal::BufferType::DRAM,
    };
    const tt::tt_metal::distributed::ReplicatedBufferConfig replicated_config{
        .size = static_cast<uint64_t>(buffer.physical_nbytes(kNocAlignmentBytes)),
    };
    return tt::tt_metal::distributed::MeshBuffer::create(replicated_config, local_config, &mesh_device);
}

void create_circular_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreCoord& core,
    const uint32_t cb_index,
    const uint32_t entry_count,
    const uint32_t page_size,
    const tt::DataFormat data_format = kDataFormat) {
    const auto config = tt::tt_metal::CircularBufferConfig(entry_count * page_size, {{cb_index, data_format}})
                            .set_page_size(cb_index, page_size);
    static_cast<void>(tt::tt_metal::CreateCircularBuffer(program, core, config));
}

[[nodiscard]] std::vector<uint32_t> to_runtime_args(const device_protocol::DeviceStepDesc& desc) {
    std::vector<uint32_t> args(device_protocol::step_desc_word_count, 0);
    std::memcpy(args.data(), &desc, sizeof(desc));
    return args;
}

[[nodiscard]] uint32_t checked_length(const size_t value, const char* label) {
    TT_FATAL(
        value <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()), "{} {} overflows uint32_t", label, value);
    return static_cast<uint32_t>(value);
}

[[nodiscard]] uint32_t stencil_source_left_pad(const device_protocol::DeviceStepDesc& desc) {
    TT_FATAL(desc.k > 0U && desc.k <= device_protocol::step_coeff_capacity, "invalid stencil width {}", desc.k);
    return device_protocol::step_coeff_capacity - desc.k;
}

[[nodiscard]] const SignalBuffer& resolve_signal_buffer(const LiftingForwardPlan& plan, const StreamRef stream) {
    return plan.resolve_stream_buffer(stream);
}

[[nodiscard]] const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& resolve_mesh_buffer(
    const LiftingWorkingBuffers& buffers, const StreamRef stream) {
    return stream.family == LogicalStream::kEven ? buffers.even.at(stream.slot) : buffers.odd.at(stream.slot);
}

[[nodiscard]] const SignalBuffer& resolve_output_buffer_desc(
    const LiftingForwardPlan& plan, const LiftingStepRoute& route) {
    return plan.resolve_stream_buffer(route.output);
}

[[nodiscard]] const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& resolve_output_mesh_buffer(
    const LiftingWorkingBuffers& buffers, const LiftingStepRoute& route) {
    return resolve_mesh_buffer(buffers, route.output);
}

void run_program(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    tt::tt_metal::Program&& program) {
    ZoneScopedN("Enqueue Mesh Workload");
    tt::tt_metal::distributed::MeshWorkload workload;
    const auto device_range = tt::tt_metal::distributed::MeshCoordinateRange(mesh_device.shape());
    workload.add_program(device_range, std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
    {
        ZoneScopedN("Device Finish");
        tt::tt_metal::distributed::Finish(command_queue);
    }
}

[[nodiscard]] ScaleStepProgram create_scale_step_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& source_buffer,
    const tt::tt_metal::Buffer& output_buffer,
    const SignalBuffer& output_desc) {
    TT_FATAL(output_desc.element_size_bytes == sizeof(float), "LWT scale path is fp32-only");
    TT_FATAL(output_desc.stick_width == 32, "LWT scale path expects 32-element row-major sticks");

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const uint32_t tile_size = tt::tile_size(kDataFormat);
    const uint32_t stick_nbytes = output_desc.aligned_stick_bytes(kCircularBufferAlignmentBytes);

    create_circular_buffer(program, core, kScaleInputCb, 1, tile_size);
    create_circular_buffer(program, core, kScaleCoeffCb, 1, tile_size);
    create_circular_buffer(program, core, kScaleOutputCb, 1, tile_size);
    create_circular_buffer(program, core, kScaleSrcCacheCb, 1, stick_nbytes);

    std::vector<uint32_t> reader_compile_args = {
        kScaleInputCb,
        kScaleCoeffCb,
        stick_nbytes,
        kScaleSrcCacheCb,
        output_desc.stick_width,
    };
    tt::tt_metal::TensorAccessorArgs(source_buffer).append_to(reader_compile_args);

    std::vector<uint32_t> writer_compile_args = {kScaleOutputCb, stick_nbytes};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_args);
    const std::vector<uint32_t> compute_compile_args = {kScaleInputCb, kScaleCoeffCb, kScaleOutputCb};
    std::vector<UnpackToDestMode> scale_unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    // Keep scale path input/coeff tile unpack in full FP32 precision.
    scale_unpack_to_dest_mode[kScaleInputCb] = UnpackToDestMode::UnpackToDestFp32;
    scale_unpack_to_dest_mode[kScaleCoeffCb] = UnpackToDestMode::UnpackToDestFp32;

    const auto reader = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kScaleReaderKernelPath),
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));
    const auto writer = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kRowMajorWriterKernelPath),
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));
    const auto compute = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kMulComputeKernelPath),
        core,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = scale_unpack_to_dest_mode,
            .compile_args = compute_compile_args,
        });

    return ScaleStepProgram{
        .program = std::move(program),
        .reader = reader,
        .compute = compute,
        .writer = writer,
    };
}

void set_scale_step_runtime_args(
    const ScaleStepProgram& program_bundle,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& source_buffer,
    const tt::tt_metal::Buffer& output_buffer,
    const SignalBuffer& source_desc,
    const LiftingStepRoute& route,
    const device_protocol::DeviceStepDesc& desc) {
    TT_FATAL(desc.k == 1U, "Scale steps must provide exactly one coefficient");
    const uint32_t output_stick_count = checked_length(
        (route.output_length + source_desc.stick_width - 1) / source_desc.stick_width, "output stick count");

    tt::tt_metal::SetRuntimeArgs(
        program_bundle.program,
        program_bundle.reader,
        core,
        std::array<uint32_t, 4>{
            static_cast<uint32_t>(source_buffer.address()),
            checked_length(route.source_length, "source length"),
            output_stick_count,
            desc.coeffs_packed[0],
        });
    tt::tt_metal::SetRuntimeArgs(
        program_bundle.program, program_bundle.compute, core, std::array<uint32_t, 1>{output_stick_count});
    tt::tt_metal::SetRuntimeArgs(
        program_bundle.program,
        program_bundle.writer,
        core,
        std::array<uint32_t, 2>{
            static_cast<uint32_t>(output_buffer.address()),
            output_stick_count,
        });
}

}  // namespace

LiftingPreprocessDeviceProgram create_lifting_preprocess_program(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const SignalBuffer& input_desc,
    const RuntimeLiftingScheme& scheme) {
    ZoneScoped;
    TT_FATAL(input_desc.length > 0, "Input signal must be non-empty");
    TT_FATAL(input_desc.element_size_bytes == sizeof(float), "Lifting preprocess currently supports fp32 only");

    const SignalBuffer planned_input = with_address(input_desc, input_buffer.address());
    const uint32_t wavelet_pad = static_cast<uint32_t>(scheme.tap_size - 1);
    const PadSplit1DLayout provisional_layout = make_pad_split_1d_layout(
        planned_input, 0, 0, Pad1DConfig{.mode = scheme.mode, .left = wavelet_pad, .right = wavelet_pad});

    auto even_ping = create_signal_mesh_buffer(mesh_device, provisional_layout.output.even);
    auto even_pong = create_signal_mesh_buffer(mesh_device, provisional_layout.output.even);
    auto odd_ping = create_signal_mesh_buffer(mesh_device, provisional_layout.output.odd);
    auto odd_pong = create_signal_mesh_buffer(mesh_device, provisional_layout.output.odd);

    const LiftingForwardPlan plan = make_forward_lifting_plan(
        planned_input,
        scheme,
        even_ping->get_backing_buffer()->address(),
        even_pong->get_backing_buffer()->address(),
        odd_ping->get_backing_buffer()->address(),
        odd_pong->get_backing_buffer()->address());

    PadSplit1DDeviceProgram preprocess = create_pad_split_1d_program(
        kernel_root,
        core,
        input_buffer,
        *(even_ping->get_backing_buffer()),
        *(odd_ping->get_backing_buffer()),
        plan.preprocess_layout);

    return LiftingPreprocessDeviceProgram{
        .scheme = scheme,
        .plan = plan,
        .buffers =
            LiftingWorkingBuffers{
                .even = MeshBufferPair{.ping = std::move(even_ping), .pong = std::move(even_pong)},
                .odd = MeshBufferPair{.ping = std::move(odd_ping), .pong = std::move(odd_pong)},
            },
        .preprocess = std::move(preprocess),
    };
}

void run_preprocess(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    LiftingPreprocessDeviceProgram& bundle) {
    run_program(mesh_device, command_queue, std::move(bundle.preprocess.program));
}

[[nodiscard]] bool debug_step_readback_enabled() {
    const char* raw = std::getenv("TT_WAVELET_DEBUG_STEP_READBACK");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
}

void debug_print_signal(const char* label, const std::vector<float>& values, const size_t logical_length) {
    std::cout << label << " (" << logical_length << "): [";
    for (size_t i = 0; i < logical_length; ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << std::scientific << std::setprecision(8) << static_cast<double>(values[i]);
    }
    std::cout << std::defaultfloat << "]\n";
}

[[nodiscard]] PredictUpdateProgram create_predict_update_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& any_working_buffer,
    const SignalBuffer& output_desc) {
    TT_FATAL(output_desc.element_size_bytes == sizeof(float), "LWT path is fp32-only");
    TT_FATAL(output_desc.stick_width == 32, "LWT path expects 32-element sticks");

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const uint32_t tile_size = tt::tile_size(kDataFormat);
    const uint32_t stick_nbytes = output_desc.aligned_stick_bytes(kCircularBufferAlignmentBytes);

    create_circular_buffer(program, core, kLwtSrcTile0Cb, 1, tile_size);
    create_circular_buffer(program, core, kLwtSrcTile1Cb, 1, tile_size);
    create_circular_buffer(program, core, kLwtBaseTileCb, 1, tile_size);
    create_circular_buffer(program, core, kLwtOutputCb, 1, tile_size);
    create_circular_buffer(program, core, kLwtSrcCacheCb, 1, stick_nbytes);
    create_circular_buffer(program, core, kLwtBaseCacheCb, 1, stick_nbytes);
    create_circular_buffer(program, core, kLwtSyncCb, 1, kLwtSyncPageSize);

    std::vector<uint32_t> reader_ct_args = {
        kLwtSrcTile0Cb,
        kLwtSrcTile1Cb,
        kLwtBaseTileCb,
        stick_nbytes,
        kLwtSrcCacheCb,
        kLwtBaseCacheCb,
        output_desc.stick_width,
        kLwtSyncCb,
    };
    tt::tt_metal::TensorAccessorArgs(any_working_buffer).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {
        kLwtOutputCb,
        stick_nbytes,
        kLwtSyncCb,
    };
    tt::tt_metal::TensorAccessorArgs(any_working_buffer).append_to(writer_ct_args);

    const std::vector<uint32_t> compute_ct_args = {
        kLwtSrcTile0Cb,
        kLwtSrcTile1Cb,
        kLwtBaseTileCb,
        kLwtOutputCb,
    };
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    // copy_tile(cb_input0/1/base, ...) should unpack directly to Dest in FP32.
    unpack_to_dest_mode[kLwtSrcTile0Cb] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kLwtSrcTile1Cb] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kLwtBaseTileCb] = UnpackToDestMode::UnpackToDestFp32;

    const auto reader = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kLwtReaderKernelPath),
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));
    const auto writer = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kLwtWriterKernelPath),
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));
    const auto compute = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kLwtComputeKernelPath),
        core,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = compute_ct_args,
        });

    return PredictUpdateProgram{
        .program = std::move(program),
        .reader = reader,
        .compute = compute,
        .writer = writer,
    };
}

void set_predict_update_runtime_args(
    const PredictUpdateProgram& program_bundle,
    const tt::tt_metal::CoreCoord& core,
    const LiftingPreprocessDeviceProgram& bundle,
    const std::vector<size_t>& pu_step_indices) {
    const size_t num_pu_steps = pu_step_indices.size();

    // Reader runtime args:
    // [num_steps, (src_addr, src_len, base_addr, base_len, out_len, out_sticks, src_off, base_off, src_left_pad) * N]
    constexpr uint32_t kReaderArgsPerStep = 9;
    std::vector<uint32_t> reader_rt(1 + num_pu_steps * kReaderArgsPerStep, 0);
    reader_rt[0] = static_cast<uint32_t>(num_pu_steps);

    for (size_t i = 0; i < num_pu_steps; ++i) {
        const size_t step_index = pu_step_indices[i];
        const auto& route = bundle.plan.routes[step_index];
        const auto& source_desc = resolve_signal_buffer(bundle.plan, route.source);
        const auto& src_buf = resolve_mesh_buffer(bundle.buffers, route.source);
        const auto& base_buf = resolve_mesh_buffer(bundle.buffers, route.base);
        const uint32_t output_stick_count = checked_length(
            (route.output_length + source_desc.stick_width - 1) / source_desc.stick_width, "output sticks");

        const uint32_t off = static_cast<uint32_t>(1 + i * kReaderArgsPerStep);
        reader_rt[off + 0] = static_cast<uint32_t>(src_buf->address());
        reader_rt[off + 1] = checked_length(route.source_length, "source length");
        reader_rt[off + 2] = static_cast<uint32_t>(base_buf->address());
        reader_rt[off + 3] = checked_length(route.base_length, "base length");
        reader_rt[off + 4] = checked_length(route.output_length, "output length");
        reader_rt[off + 5] = output_stick_count;
        reader_rt[off + 6] = checked_length(route.source_offset, "source offset");
        reader_rt[off + 7] = checked_length(route.base_offset, "base offset");
        reader_rt[off + 8] = stencil_source_left_pad(bundle.plan.packed_steps[step_index]);
    }
    tt::tt_metal::SetRuntimeArgs(program_bundle.program, program_bundle.reader, core, reader_rt);

    // Compute runtime args: [num_steps, (output_sticks, DeviceStepDesc[19]) * N]
    constexpr uint32_t kComputeArgsPerStep = 1 + device_protocol::step_desc_word_count;
    std::vector<uint32_t> compute_rt(1 + num_pu_steps * kComputeArgsPerStep, 0);
    compute_rt[0] = static_cast<uint32_t>(num_pu_steps);

    for (size_t i = 0; i < num_pu_steps; ++i) {
        const size_t step_index = pu_step_indices[i];
        const auto& route = bundle.plan.routes[step_index];
        const auto& packed = bundle.plan.packed_steps[step_index];
        const auto& source_desc = resolve_signal_buffer(bundle.plan, route.source);
        const uint32_t output_stick_count = checked_length(
            (route.output_length + source_desc.stick_width - 1) / source_desc.stick_width, "output sticks");

        const uint32_t off = static_cast<uint32_t>(1 + i * kComputeArgsPerStep);
        compute_rt[off] = output_stick_count;
        auto desc_words = to_runtime_args(packed);
        for (size_t w = 0; w < desc_words.size(); ++w) {
            compute_rt[off + 1 + w] = desc_words[w];
        }
    }
    tt::tt_metal::SetRuntimeArgs(program_bundle.program, program_bundle.compute, core, compute_rt);

    // Writer runtime args: [num_steps, (output_addr, output_sticks) * N]
    constexpr uint32_t kWriterArgsPerStep = 2;
    std::vector<uint32_t> writer_rt(1 + num_pu_steps * kWriterArgsPerStep, 0);
    writer_rt[0] = static_cast<uint32_t>(num_pu_steps);

    for (size_t i = 0; i < num_pu_steps; ++i) {
        const size_t step_index = pu_step_indices[i];
        const auto& route = bundle.plan.routes[step_index];
        const auto& source_desc = resolve_signal_buffer(bundle.plan, route.source);
        const auto& out_buf = resolve_output_mesh_buffer(bundle.buffers, route);
        const uint32_t output_stick_count = checked_length(
            (route.output_length + source_desc.stick_width - 1) / source_desc.stick_width, "output sticks");

        const uint32_t off = static_cast<uint32_t>(1 + i * kWriterArgsPerStep);
        writer_rt[off + 0] = static_cast<uint32_t>(out_buf->address());
        writer_rt[off + 1] = output_stick_count;
    }
    tt::tt_metal::SetRuntimeArgs(program_bundle.program, program_bundle.writer, core, writer_rt);
}

LiftingActiveStreams execute_forward_lifting(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const tt::tt_metal::CoreCoord& core,
    const LiftingPreprocessDeviceProgram& bundle) {
    ZoneScoped;
    TT_FATAL(
        bundle.plan.routes.size() == bundle.plan.packed_steps.size(),
        "Route count {} must match packed step count {}",
        bundle.plan.routes.size(),
        bundle.plan.packed_steps.size());

    for (size_t step_index = 0; step_index < bundle.plan.routes.size();) {
        const auto& step = bundle.scheme.steps[step_index];
        const auto& route = bundle.plan.routes[step_index];
        const auto& packed_step = bundle.plan.packed_steps[step_index];

        if (step.type == StepType::kSwap) {
            ++step_index;
            continue;
        }

        if (step.type == StepType::kPredict || step.type == StepType::kUpdate) {
            std::vector<size_t> pu_step_indices;
            size_t scan = step_index;
            while (scan < bundle.plan.routes.size()) {
                const auto& scan_step = bundle.scheme.steps[scan];
                if (scan_step.type == StepType::kSwap) {
                    ++scan;
                    continue;
                }
                if (scan_step.type == StepType::kPredict || scan_step.type == StepType::kUpdate) {
                    pu_step_indices.push_back(scan);
                    ++scan;
                    continue;
                }
                break;
            }

            TT_FATAL(!pu_step_indices.empty(), "empty predict/update segment at step {}", step_index);

            const auto& first_route = bundle.plan.routes[pu_step_indices.front()];
            const auto& first_output_desc = resolve_output_buffer_desc(bundle.plan, first_route);
            auto program_bundle = create_predict_update_program(
                kernel_root, core, *(bundle.buffers.even.ping->get_backing_buffer()), first_output_desc);
            set_predict_update_runtime_args(program_bundle, core, bundle, pu_step_indices);
            run_program(mesh_device, command_queue, std::move(program_bundle.program));

            step_index = scan;
            continue;
        }

        const auto& source_desc = resolve_signal_buffer(bundle.plan, route.source);
        const auto& output_desc = resolve_output_buffer_desc(bundle.plan, route);
        const auto& source_buffer = resolve_mesh_buffer(bundle.buffers, route.source);
        const auto& output_buffer = resolve_output_mesh_buffer(bundle.buffers, route);

        TT_FATAL(
            step.type == StepType::kScaleEven || step.type == StepType::kScaleOdd,
            "unexpected step type {} in scale path",
            static_cast<int>(step.type));

        auto program_bundle = create_scale_step_program(
            kernel_root,
            core,
            *(source_buffer->get_backing_buffer()),
            *(output_buffer->get_backing_buffer()),
            output_desc);
        set_scale_step_runtime_args(
            program_bundle,
            core,
            *(source_buffer->get_backing_buffer()),
            *(output_buffer->get_backing_buffer()),
            source_desc,
            route,
            packed_step);
        run_program(mesh_device, command_queue, std::move(program_bundle.program));

        if (debug_step_readback_enabled()) {
            std::vector<float> device_output;
            auto debug_buffer = output_buffer;
            tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, device_output, debug_buffer, true);
            const auto logical_length = route.output_length;
            std::cout << "Step " << step_index
                      << " output family=" << (route.output.family == LogicalStream::kEven ? "even" : "odd")
                      << " slot=" << (route.output.slot == StreamSlot::kPing ? "ping" : "pong") << '\n';
            debug_print_signal("step output", device_output, logical_length);
        }

        ++step_index;
    }

    return bundle.plan.final_active;
}

}  // namespace ttwv
