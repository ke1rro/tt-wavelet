#include "tt_wavelet/include/lifting/lifting_device.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <vector>

#include <tt_stl/assert.hpp>

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

constexpr const char* kStencilReaderKernelPath = "kernels/dataflow/lwt_stencil_step_reader.cpp";
constexpr const char* kScaleReaderKernelPath = "kernels/dataflow/lwt_scale_reader.cpp";
constexpr const char* kRowMajorWriterKernelPath = "kernels/dataflow/lwt_row_major_writer.cpp";
constexpr const char* kStencilComputeKernelPath = "kernels/lwt_stencil_compute.cpp";
constexpr const char* kMulComputeKernelPath = "kernels/lwt_mul_compute.cpp";

struct StencilStepProgram {
    tt::tt_metal::Program program;
    tt::tt_metal::KernelHandle reader{};
    tt::tt_metal::KernelHandle stencil_compute{};
    tt::tt_metal::KernelHandle writer{};
};

struct ScaleStepProgram {
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

[[nodiscard]] tt::tt_metal::CBHandle create_circular_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreCoord& core,
    const uint32_t cb_index,
    const uint32_t entry_count,
    const uint32_t page_size,
    const tt::DataFormat data_format = kDataFormat) {
    const auto config = tt::tt_metal::CircularBufferConfig(entry_count * page_size, {{cb_index, data_format}})
                            .set_page_size(cb_index, page_size);
    return tt::tt_metal::CreateCircularBuffer(program, core, config);
}

[[nodiscard]] std::vector<uint32_t> to_runtime_args(const device::DeviceStepDesc& desc) {
    std::vector<uint32_t> args(device::step_desc_word_count, 0);
    std::memcpy(args.data(), &desc, sizeof(desc));
    return args;
}

[[nodiscard]] uint32_t checked_length(const size_t value, const char* label) {
    TT_FATAL(value <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()), "{} {} overflows uint32_t", label, value);
    return static_cast<uint32_t>(value);
}

[[nodiscard]] const SignalBuffer& resolve_signal_buffer(
    const LiftingForwardPlan& plan, const StreamRef stream) {
    return plan.resolve_stream_buffer(stream);
}

[[nodiscard]] const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& resolve_mesh_buffer(
    const LiftingWorkingBuffers& buffers, const StreamRef stream) {
    return stream.family == LogicalStream::Even ? buffers.even.at(stream.slot) : buffers.odd.at(stream.slot);
}

[[nodiscard]] const SignalBuffer& resolve_output_buffer_desc(const LiftingForwardPlan& plan, const LiftingStepRoute& route) {
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
    tt::tt_metal::distributed::MeshWorkload workload;
    const auto device_range = tt::tt_metal::distributed::MeshCoordinateRange(mesh_device.shape());
    workload.add_program(device_range, std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
    tt::tt_metal::distributed::Finish(command_queue);
}

[[nodiscard]] StencilStepProgram create_stencil_step_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& source_buffer,
    const tt::tt_metal::Buffer& base_buffer,
    const tt::tt_metal::Buffer& output_buffer,
    const SignalBuffer& output_desc) {
    TT_FATAL(output_desc.element_size_bytes == sizeof(float), "LWT stencil path is fp32-only");
    TT_FATAL(output_desc.stick_width == 32, "LWT stencil path expects 32-element row-major sticks");

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const uint32_t tile_size = tt::tile_size(kDataFormat);
    const uint32_t stick_nbytes = output_desc.aligned_stick_bytes(kCircularBufferAlignmentBytes);

    create_circular_buffer(program, core, kStencilInput0Cb, 1, tile_size);
    create_circular_buffer(program, core, kStencilInput1Cb, 1, tile_size);
    create_circular_buffer(program, core, kStencilBaseCb, 1, tile_size);
    create_circular_buffer(program, core, kStencilOutputCb, 1, tile_size);
    create_circular_buffer(program, core, kStencilSrcCacheCb, 1, stick_nbytes);
    create_circular_buffer(program, core, kStencilBaseCacheCb, 1, stick_nbytes);

    std::vector<uint32_t> reader_compile_args = {
        kStencilInput0Cb,
        kStencilInput1Cb,
        kStencilBaseCb,
        stick_nbytes,
        kStencilSrcCacheCb,
        kStencilBaseCacheCb,
        output_desc.stick_width,
    };
    tt::tt_metal::TensorAccessorArgs(source_buffer).append_to(reader_compile_args);

    std::vector<uint32_t> writer_compile_args = {kStencilOutputCb, stick_nbytes};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_args);

    const std::vector<uint32_t> stencil_compute_compile_args = {
        kStencilInput0Cb,
        kStencilInput1Cb,
        kStencilBaseCb,
        kStencilOutputCb,
    };

    const auto reader = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kStencilReaderKernelPath),
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));
    const auto writer = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kRowMajorWriterKernelPath),
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));
    const auto stencil_compute = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kStencilComputeKernelPath),
        core,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .compile_args = stencil_compute_compile_args,
        });

    return StencilStepProgram{
        .program = std::move(program),
        .reader = reader,
        .stencil_compute = stencil_compute,
        .writer = writer,
    };
}

void set_stencil_step_runtime_args(
    const StencilStepProgram& program_bundle,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& source_buffer,
    const tt::tt_metal::Buffer& base_buffer,
    const tt::tt_metal::Buffer& output_buffer,
    const SignalBuffer& source_desc,
    const SignalBuffer& output_desc,
    const device::DeviceStepDesc& desc) {
    tt::tt_metal::SetRuntimeArgs(
        program_bundle.program,
        program_bundle.reader,
        core,
        std::array<uint32_t, 6>{
            static_cast<uint32_t>(source_buffer.address()),
            static_cast<uint32_t>(base_buffer.address()),
            checked_length(source_desc.length, "source length"),
            checked_length(output_desc.length, "output length"),
            checked_length(output_desc.stick_count(), "output stick count"),
            desc.k,
        });
    tt::tt_metal::SetRuntimeArgs(program_bundle.program, program_bundle.stencil_compute, core, to_runtime_args(desc));
    tt::tt_metal::SetRuntimeArgs(
        program_bundle.program,
        program_bundle.writer,
        core,
        std::array<uint32_t, 2>{
            static_cast<uint32_t>(output_buffer.address()),
            checked_length(output_desc.stick_count(), "output stick count"),
        });
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
    const SignalBuffer& output_desc,
    const device::DeviceStepDesc& desc) {
    TT_FATAL(desc.k == 1U, "Scale steps must provide exactly one coefficient");

    tt::tt_metal::SetRuntimeArgs(
        program_bundle.program,
        program_bundle.reader,
        core,
        std::array<uint32_t, 4>{
            static_cast<uint32_t>(source_buffer.address()),
            checked_length(source_desc.length, "source length"),
            checked_length(output_desc.stick_count(), "output stick count"),
            desc.coeffs_packed[0],
        });
    tt::tt_metal::SetRuntimeArgs(
        program_bundle.program,
        program_bundle.compute,
        core,
        std::array<uint32_t, 1>{checked_length(output_desc.stick_count(), "output stick count")});
    tt::tt_metal::SetRuntimeArgs(
        program_bundle.program,
        program_bundle.writer,
        core,
        std::array<uint32_t, 2>{
            static_cast<uint32_t>(output_buffer.address()),
            checked_length(output_desc.stick_count(), "output stick count"),
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

LiftingActiveStreams execute_forward_lifting(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const tt::tt_metal::CoreCoord& core,
    const LiftingPreprocessDeviceProgram& bundle) {
    TT_FATAL(
        bundle.plan.routes.size() == bundle.plan.packed_steps.size(),
        "Route count {} must match packed step count {}",
        bundle.plan.routes.size(),
        bundle.plan.packed_steps.size());

    for (size_t step_index = 0; step_index < bundle.plan.routes.size(); ++step_index) {
        const auto& step = bundle.scheme.steps[step_index];
        const auto& route = bundle.plan.routes[step_index];
        const auto& packed_step = bundle.plan.packed_steps[step_index];

        if (step.type == StepType::Swap) {
            continue;
        }

        const auto& source_desc = resolve_signal_buffer(bundle.plan, route.source);
        const auto& output_desc = resolve_output_buffer_desc(bundle.plan, route);
        const auto& source_buffer = resolve_mesh_buffer(bundle.buffers, route.source);
        const auto& output_buffer = resolve_output_mesh_buffer(bundle.buffers, route);

        if (step.type == StepType::Predict || step.type == StepType::Update) {
            const auto& base_buffer = resolve_mesh_buffer(bundle.buffers, route.base);
            auto program_bundle = create_stencil_step_program(
                kernel_root,
                core,
                *(source_buffer->get_backing_buffer()),
                *(base_buffer->get_backing_buffer()),
                *(output_buffer->get_backing_buffer()),
                output_desc);
            set_stencil_step_runtime_args(
                program_bundle,
                core,
                *(source_buffer->get_backing_buffer()),
                *(base_buffer->get_backing_buffer()),
                *(output_buffer->get_backing_buffer()),
                source_desc,
                output_desc,
                packed_step);
            run_program(mesh_device, command_queue, std::move(program_bundle.program));
            continue;
        }

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
            output_desc,
            packed_step);
        run_program(mesh_device, command_queue, std::move(program_bundle.program));
    }

    return bundle.plan.final_active;
}

}  // namespace ttwv
