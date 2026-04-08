#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_wavelet/include/lifting/lifting_scheme.hpp"
#include "tt_wavelet/include/lifting/lifting_step.hpp"
#include "tt_wavelet/include/lwt_device.hpp"
#include "tt_wavelet/include/pad_split.hpp"
#include "tt_wavelet/include/pad_split_device.hpp"

namespace fs = std::filesystem;
using namespace ttwv;
using namespace ttwv::lwt;

static constexpr auto kScheme = make_lifting_scheme(
    /*tap_size=*/6,
    /*delay_even=*/1,
    /*delay_odd=*/2,
    UpdateStep<2>{.coefficients = {-0.5f, -0.5f}, .shift = -1},
    PredictStep<2>{.coefficients = {0.25f, 0.25f}, .shift = 0},
    SwapStep{.coefficients = {}, .shift = 0},
    ScaleEvenStep<1>{.coefficients = {1.414213562373095f}, .shift = 0},
    ScaleOddStep<1>{.coefficients = {-0.7071067811865476f}, .shift = 0});

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using MeshBuffer = tt::tt_metal::distributed::MeshBuffer;
using MeshCQ = tt::tt_metal::distributed::MeshCommandQueue;

static std::shared_ptr<MeshBuffer> alloc_signal_buffer(
    MeshDevice* dev, const SignalBuffer& sig, uint32_t alignment = 32) {
    tt::tt_metal::distributed::DeviceLocalBufferConfig local{
        .page_size = sig.aligned_stick_bytes(alignment),
        .buffer_type = tt::tt_metal::BufferType::DRAM,
    };
    tt::tt_metal::distributed::ReplicatedBufferConfig rep{
        .size = static_cast<uint64_t>(sig.physical_nbytes(alignment)),
    };
    return MeshBuffer::create(rep, local, dev);
}

static void print_signal(const std::string& label, const std::vector<float>& data, size_t length) {
    std::cout << label << " [";
    for (size_t i = 0; i < length; ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << data[i];
    }
    std::cout << "]\n";
}

static void run_step(
    MeshDevice* dev,
    MeshCQ& cq,
    const fs::path& kernel_root,
    const tt::tt_metal::CoreCoord& core,
    const SignalBuffer& in_desc,
    const std::shared_ptr<MeshBuffer>& in_buf,
    SignalBuffer& out_desc,
    std::shared_ptr<MeshBuffer>& out_buf,
    uint32_t split_phase,
    int32_t source_offset,
    uint32_t stencil_k,
    const std::string& label) {
    out_buf = alloc_signal_buffer(dev, out_desc);
    out_desc.dram_address = out_buf->get_backing_buffer()->address();

    const uint32_t signal_len = static_cast<uint32_t>(in_desc.length);
    const uint32_t num_tiles = (signal_len + 31u) / 32u;

    LwtStepConfig cfg{
        .signal_length = signal_len,
        .split_phase = split_phase,
        .source_offset = source_offset,
        .stencil_k = stencil_k,
        .is_pad_split = false,
        .num_tiles = num_tiles,
    };

    LwtProgram prog =
        create_lwt_step_program(kernel_root, core, *in_buf->get_backing_buffer(), *out_buf->get_backing_buffer(), cfg);

    tt::tt_metal::distributed::MeshWorkload workload;
    const auto dev_range = tt::tt_metal::distributed::MeshCoordinateRange(dev->shape());
    workload.add_program(dev_range, std::move(prog.program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
    tt::tt_metal::distributed::Finish(cq);

    std::vector<float> out_data;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, out_data, out_buf, true);
    print_signal(label, out_data, out_desc.length);
}

int main() {
    constexpr int kDeviceId = 0;
    auto mesh_device = MeshDevice::create_unit_mesh(kDeviceId);
    MeshCQ& cq = mesh_device->mesh_command_queue();
    constexpr tt::tt_metal::CoreCoord kCore{0, 0};
    const fs::path kernel_root = TT_WAVELET_SOURCE_DIR;

    constexpr size_t kSignalLength = 8;
    std::vector<float> original_signal(kSignalLength);
    for (size_t i = 0; i < kSignalLength; ++i) {
        original_signal[i] = static_cast<float>(i + 1);
    }

    print_signal("Input:", original_signal, kSignalLength);
    std::cout << "\n";
    const uint32_t pad = static_cast<uint32_t>(kScheme.tap_size) - 1u;
    const Pad1DConfig pad_cfg{.mode = BoundaryMode::Symmetric, .left = pad, .right = pad};

    SignalBuffer input_desc{
        .dram_address = 0, .length = kSignalLength, .stick_width = 32, .element_size_bytes = sizeof(float)};
    PadSplit1DLayout ps_layout = make_pad_split_1d_layout(input_desc, 0, 0, pad_cfg);

    auto input_buf = alloc_signal_buffer(mesh_device.get(), ps_layout.input);
    auto even_buf = alloc_signal_buffer(mesh_device.get(), ps_layout.output.even);
    auto odd_buf = alloc_signal_buffer(mesh_device.get(), ps_layout.output.odd);

    ps_layout.input.dram_address = input_buf->get_backing_buffer()->address();
    ps_layout.output.even.dram_address = even_buf->get_backing_buffer()->address();
    ps_layout.output.odd.dram_address = odd_buf->get_backing_buffer()->address();

    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, input_buf, original_signal, false);

    PadSplit1DDeviceProgram ps_prog = create_pad_split_1d_program(
        kernel_root,
        kCore,
        *input_buf->get_backing_buffer(),
        *even_buf->get_backing_buffer(),
        *odd_buf->get_backing_buffer(),
        ps_layout);

    {
        tt::tt_metal::distributed::MeshWorkload workload;
        const auto dev_range = tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());
        workload.add_program(dev_range, std::move(ps_prog.program));
        tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
        tt::tt_metal::distributed::Finish(cq);
    }

    {
        std::vector<float> even_data, odd_data;
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, even_data, even_buf, true);
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, odd_data, odd_buf, true);
        print_signal("pad+split even:", even_data, ps_layout.output.even.length);
        print_signal("pad+split odd: ", odd_data, ps_layout.output.odd.length);
        std::cout << "\n";
    }

    SignalBuffer cur_even_desc = ps_layout.output.even;
    SignalBuffer cur_odd_desc = ps_layout.output.odd;

    int step_idx = 0;
    kScheme.for_each_step([&](const auto& step) {
        ++step_idx;
        using Step = std::decay_t<decltype(step)>;

        if constexpr (is_predict_step<Step> && !is_scale_step<Step>) {
            const uint32_t K = static_cast<uint32_t>(step.n_coeffs);
            const std::string label = "step " + std::to_string(step_idx) +
                                      " predict (shift=" + std::to_string(step.shift) + ", K=" + std::to_string(K) +
                                      ") odd:";
            run_step(
                mesh_device.get(),
                cq,
                kernel_root,
                kCore,
                cur_even_desc,
                even_buf,
                cur_odd_desc,
                odd_buf,
                /*split_phase=*/0,
                step.shift,
                K,
                label);

        } else if constexpr (is_update_step<Step> && !is_scale_step<Step>) {
            const uint32_t K = static_cast<uint32_t>(step.n_coeffs);
            const std::string label = "step " + std::to_string(step_idx) +
                                      " update (shift=" + std::to_string(step.shift) + ", K=" + std::to_string(K) +
                                      ") even:";
            run_step(
                mesh_device.get(),
                cq,
                kernel_root,
                kCore,
                cur_odd_desc,
                odd_buf,
                cur_even_desc,
                even_buf,
                /*split_phase=*/1,
                step.shift,
                K,
                label);

        } else {
            std::cout << "step " << step_idx << " (" << static_cast<int>(Step::type) << ") — skipped\n";
        }
    });

    std::cout << "\nDone.\n";
    return 0;
}
