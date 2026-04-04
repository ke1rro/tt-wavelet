#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_wavelet/includes/padding.hpp"
#include "tt_wavelet/includes/padding_device.hpp"

using namespace ttwv;

int main() {
    constexpr int device_id = 0;
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    tt::tt_metal::distributed::MeshCommandQueue& command_queue = mesh_device->mesh_command_queue();
    constexpr tt::tt_metal::CoreCoord core{0, 0};

    // 1. Prepare Host Data
    const size_t signal_length = 100;
    std::vector<float> original_signal(signal_length);
    for (size_t i = 0; i < signal_length; ++i) {
        original_signal[i] = static_cast<float>(i + 1);
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    Pad1DConfig pad_config{.mode = BoundaryMode::Symmetric, .left = 16, .right = 16};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    SignalBuffer input_desc{
        .dram_address = 0, .length = signal_length, .stick_width = 32, .element_size_bytes = sizeof(float)};

    Pad1DLayout layout = make_pad_1d_layout(input_desc, 0, pad_config);

    // 2. Allocate Mesh Buffers
    tt::tt_metal::distributed::DeviceLocalBufferConfig input_local_conf{
        .page_size = layout.input.aligned_stick_bytes(32), .buffer_type = tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::distributed::ReplicatedBufferConfig input_rep_conf{
        .size = static_cast<uint64_t>(layout.input.physical_nbytes())};

    tt::tt_metal::distributed::DeviceLocalBufferConfig output_local_conf{
        .page_size = layout.output.aligned_stick_bytes(32), .buffer_type = tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::distributed::ReplicatedBufferConfig output_rep_conf{
        .size = static_cast<uint64_t>(layout.output.physical_nbytes())};

    auto input_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(input_rep_conf, input_local_conf, mesh_device.get());
    auto output_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(output_rep_conf, output_local_conf, mesh_device.get());

    // Update layout with actual backing buffer DRAM addresses
    layout.input.dram_address = input_buffer->get_backing_buffer()->address();
    layout.output.dram_address = output_buffer->get_backing_buffer()->address();

    // 3. Write Input
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, input_buffer, original_signal, false);

    // 4. Create and Enqueue Padding Program
    Pad1DDeviceProgram pad_program = create_symmetric_pad_1d_program(
        TT_WAVELET_SOURCE_DIR,
        core,
        *(input_buffer->get_backing_buffer()),
        *(output_buffer->get_backing_buffer()),
        layout);

    tt::tt_metal::distributed::MeshWorkload workload;
    auto dev_range = tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(dev_range, std::move(pad_program.program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
    tt::tt_metal::distributed::Finish(command_queue);

    // 5. Read Output and Verify
    std::vector<float> device_result;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, device_result, output_buffer, true);

    std::vector<float> host_reference = materialize_reference_padding(original_signal, layout);

    bool success = true;
    for (size_t i = 0; i < layout.output.length; ++i) {
        if (std::abs(device_result[i] - host_reference[i]) > 1e-5) {
            std::cerr << "Mismatch at index " << i << ": device=" << device_result[i] << ", host=" << host_reference[i]
                      << std::endl;
            success = false;
        }
    }

    if (success) {
        std::cout << "SUCCESS!\n";
    }

    return success ? 0 : 1;
}
