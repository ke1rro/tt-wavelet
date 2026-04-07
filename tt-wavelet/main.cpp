#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_wavelet/include/pad_split.hpp"
#include "tt_wavelet/include/pad_split_device.hpp"

using namespace ttwv;

int main() {
    constexpr int device_id = 0;
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    tt::tt_metal::distributed::MeshCommandQueue& command_queue = mesh_device->mesh_command_queue();
    constexpr tt::tt_metal::CoreCoord core{0, 0};

    // 1. Create Layouts and Inputs
    constexpr size_t signal_length = 3;
    std::vector<float> original_signal(signal_length);
    for (size_t i = 0; i < signal_length; ++i) {
        original_signal[i] = static_cast<float>(i + 1);
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    Pad1DConfig pad_config{.mode = BoundaryMode::Symmetric, .left = 2, .right = 2};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    SignalBuffer input_desc{
        .dram_address = 0, .length = signal_length, .stick_width = 32, .element_size_bytes = sizeof(float)};

    PadSplit1DLayout layout = make_pad_split_1d_layout(input_desc, 0, 0, pad_config);

    tt::tt_metal::distributed::DeviceLocalBufferConfig input_local_conf{
        .page_size = layout.input.aligned_stick_bytes(32), .buffer_type = tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::distributed::ReplicatedBufferConfig input_rep_conf{
        .size = static_cast<uint64_t>(layout.input.physical_nbytes())};

    auto input_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(input_rep_conf, input_local_conf, mesh_device.get());

    tt::tt_metal::distributed::DeviceLocalBufferConfig even_local_conf{
        .page_size = layout.output.even.aligned_stick_bytes(32), .buffer_type = tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::distributed::ReplicatedBufferConfig even_rep_conf{
        .size = static_cast<uint64_t>(layout.output.even.physical_nbytes())};

    auto even_buffer = tt::tt_metal::distributed::MeshBuffer::create(even_rep_conf, even_local_conf, mesh_device.get());

    tt::tt_metal::distributed::DeviceLocalBufferConfig odd_local_conf{
        .page_size = layout.output.odd.aligned_stick_bytes(32), .buffer_type = tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::distributed::ReplicatedBufferConfig odd_rep_conf{
        .size = static_cast<uint64_t>(layout.output.odd.physical_nbytes())};

    auto odd_buffer = tt::tt_metal::distributed::MeshBuffer::create(odd_rep_conf, odd_local_conf, mesh_device.get());

    layout.input.dram_address = input_buffer->get_backing_buffer()->address();
    layout.output.even.dram_address = even_buffer->get_backing_buffer()->address();
    layout.output.odd.dram_address = odd_buffer->get_backing_buffer()->address();
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, input_buffer, original_signal, false);

    PadSplit1DDeviceProgram pad_split_program = create_pad_split_1d_program(
        TT_WAVELET_SOURCE_DIR,
        core,
        *(input_buffer->get_backing_buffer()),
        *(even_buffer->get_backing_buffer()),
        *(odd_buffer->get_backing_buffer()),
        layout);

    tt::tt_metal::distributed::MeshWorkload workload;
    auto dev_range = tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(dev_range, std::move(pad_split_program.program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
    tt::tt_metal::distributed::Finish(command_queue);
    std::vector<float> device_even_result;
    std::vector<float> device_odd_result;

    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, device_even_result, even_buffer, true);
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, device_odd_result, odd_buffer, true);

    SplitResult host_reference = materialize_reference_pad_split(original_signal, layout);

    std::cout << "Original Signal: [ ";
    for (float v : original_signal) {
        std::cout << v << " ";
    }
    std::cout << "]\n\n";

    std::cout << "Padded length: " << layout.padded_length() << "\n\n";

    std::cout << "Device Even Signal (length " << layout.output.even.length << "): [ ";
    for (size_t i = 0; i < layout.output.even.length; ++i) {
        std::cout << device_even_result[i] << " ";
    }
    std::cout << "]\n\n";

    std::cout << "Device Odd Signal (length " << layout.output.odd.length << "): [ ";
    for (size_t i = 0; i < layout.output.odd.length; ++i) {
        std::cout << device_odd_result[i] << " ";
    }
    std::cout << "]\n\n";

    bool success = true;
    for (size_t i = 0; i < layout.output.even.length; ++i) {
        if (std::abs(device_even_result[i] - host_reference.even[i]) > 1e-5) {
            std::cerr << "Even mismatch at index " << i << ": device=" << device_even_result[i]
                      << ", host=" << host_reference.even[i] << std::endl;
            success = false;
        }
    }

    for (size_t i = 0; i < layout.output.odd.length; ++i) {
        if (std::abs(device_odd_result[i] - host_reference.odd[i]) > 1e-5) {
            std::cerr << "Odd mismatch at index " << i << ": device=" << device_odd_result[i]
                      << ", host=" << host_reference.odd[i] << std::endl;
            success = false;
        }
    }

    if (success) {
        std::cout << "SUCCESS!\n";
    }

    return success ? 0 : 1;
}