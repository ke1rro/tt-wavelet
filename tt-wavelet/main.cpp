#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_wavelet/include/pad_split.hpp"
#include "tt_wavelet/include/pad_split_device.hpp"
#include "tt_wavelet/include/stencil_device.hpp"

using namespace ttwv;

void print_vec(const char* label, const std::vector<float>& v, size_t count) {
    std::cout << label << " [ ";
    for (size_t i = 0; i < count && i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << "]\n";
}

bool compare_vec(
    const char* label,
    const std::vector<float>& device,
    const std::vector<float>& host,
    size_t count,
    float tol = 1e-5F) {
    bool ok = true;
    for (size_t i = 0; i < count && i < device.size() && i < host.size(); ++i) {
        if (std::abs(device[i] - host[i]) > tol) {
            std::cerr << label << " mismatch at " << i << ": device=" << device[i] << ", host=" << host[i]
                      << " (diff=" << std::abs(device[i] - host[i]) << ")\n";
            ok = false;
        }
    }
    return ok;
}

bool test_pad_split_stencil(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const tt::tt_metal::CoreCoord& core) {
    std::cout << "=== Test: Pad+Split → Stencil (db2 predict) ===\n";

    // ── 1. Original signal ──
    constexpr size_t signal_length = 8;
    std::vector<float> original_signal(signal_length);
    for (size_t i = 0; i < signal_length; ++i) {
        original_signal[i] = static_cast<float>(i + 1);
    }
    print_vec("Original Signal:", original_signal, signal_length);

    // ── 2. db2 wavelet parameters ──
    // tap_size=4 → pad by L-1=3 on each side (symmetric boundary)
    // First step: predict, shift=-1, coefficients=[-0.5773502691896258]
    constexpr uint32_t tap_size = 4;
    const uint32_t pad_amount = tap_size - 1;  // 3
    // NOLINT
    const std::vector<float> predict_coeffs = {-0.5773502691896258F};  // NOLINT

    // ── 3. Pad + Split on device ──
    Pad1DConfig pad_config{.mode = BoundaryMode::Symmetric, .left = pad_amount, .right = pad_amount};
    SignalBuffer input_desc{
        .dram_address = 0, .length = signal_length, .stick_width = 32, .element_size_bytes = sizeof(float)};

    PadSplit1DLayout ps_layout = make_pad_split_1d_layout(input_desc, 0, 0, pad_config);

    const auto page_size = ps_layout.input.aligned_stick_bytes(32);

    // Input buffer
    auto input_buffer = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{
            .size = static_cast<uint64_t>(ps_layout.input.physical_nbytes())},
        tt::tt_metal::distributed::DeviceLocalBufferConfig{
            .page_size = page_size, .buffer_type = tt::tt_metal::BufferType::DRAM},
        &mesh_device);

    // Even output buffer
    auto even_buffer = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{
            .size = static_cast<uint64_t>(ps_layout.output.even.physical_nbytes())},
        tt::tt_metal::distributed::DeviceLocalBufferConfig{
            .page_size = ps_layout.output.even.aligned_stick_bytes(32), .buffer_type = tt::tt_metal::BufferType::DRAM},
        &mesh_device);

    // Odd output buffer
    auto odd_buffer = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{
            .size = static_cast<uint64_t>(ps_layout.output.odd.physical_nbytes())},
        tt::tt_metal::distributed::DeviceLocalBufferConfig{
            .page_size = ps_layout.output.odd.aligned_stick_bytes(32), .buffer_type = tt::tt_metal::BufferType::DRAM},
        &mesh_device);

    ps_layout.input.dram_address = input_buffer->get_backing_buffer()->address();
    ps_layout.output.even.dram_address = even_buffer->get_backing_buffer()->address();
    ps_layout.output.odd.dram_address = odd_buffer->get_backing_buffer()->address();

    // Write original signal, pad to stick width
    std::vector<float> signal_padded(ps_layout.input.physical_length(), 0.0F);
    for (size_t i = 0; i < signal_length; ++i) {
        signal_padded[i] = original_signal[i];
    }
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, input_buffer, signal_padded, false);

    // Run pad+split
    PadSplit1DDeviceProgram ps_program = create_pad_split_1d_program(
        TT_WAVELET_SOURCE_DIR,
        core,
        *(input_buffer->get_backing_buffer()),
        *(even_buffer->get_backing_buffer()),
        *(odd_buffer->get_backing_buffer()),
        ps_layout);

    {
        tt::tt_metal::distributed::MeshWorkload workload;
        workload.add_program(
            tt::tt_metal::distributed::MeshCoordinateRange(mesh_device.shape()), std::move(ps_program.program));
        tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
        tt::tt_metal::distributed::Finish(command_queue);
    }

    // Read back even/odd from device
    std::vector<float> device_even, device_odd;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, device_even, even_buffer, true);
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, device_odd, odd_buffer, true);

    // Host reference for pad+split
    SplitResult host_ps = materialize_reference_pad_split(original_signal, ps_layout);

    print_vec("Pad+Split Even (device):", device_even, ps_layout.output.even.length);
    print_vec("Pad+Split Even (host):  ", host_ps.even, ps_layout.output.even.length);
    print_vec("Pad+Split Odd  (device):", device_odd, ps_layout.output.odd.length);
    print_vec("Pad+Split Odd  (host):  ", host_ps.odd, ps_layout.output.odd.length);

    bool ps_ok = true;
    ps_ok &= compare_vec("Even", device_even, host_ps.even, ps_layout.output.even.length);
    ps_ok &= compare_vec("Odd", device_odd, host_ps.odd, ps_layout.output.odd.length);
    std::cout << "Pad+Split: " << (ps_ok ? "PASS" : "FAIL") << "\n\n";

    // ── 4. Stencil on the even stream ──
    // For predict step: conv(even, h) where h=[-0.5773502691896258], k=1
    // The stencil reader reads from the even_buffer that pad+split just wrote.
    // 17-k = 16 padding applied by the stencil reader.
    std::cout << "--- Stencil on even stream ---\n";
    std::cout << "Filter h (k=" << predict_coeffs.size() << "): [ ";
    for (float c : predict_coeffs) {
        std::cout << c << " ";
    }
    std::cout << "]\n";
    std::cout << "Stencil padding (17-k): " << (17 - predict_coeffs.size()) << "\n";

    StencilConfig stencil_config;
    stencil_config.coefficients = predict_coeffs;
    stencil_config.signal_length = static_cast<uint32_t>(ps_layout.output.even.length);
    stencil_config.stick_width = 32;
    stencil_config.element_size_bytes = sizeof(float);

    // Output buffer for stencil result
    const uint32_t stencil_output_sticks = stencil_config.num_tiles() * 32;
    auto stencil_out_buffer = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{
            .size = static_cast<uint64_t>(stencil_config.aligned_stick_bytes(32) * stencil_output_sticks)},
        tt::tt_metal::distributed::DeviceLocalBufferConfig{
            .page_size = stencil_config.aligned_stick_bytes(32), .buffer_type = tt::tt_metal::BufferType::DRAM},
        &mesh_device);

    // Create stencil program — reads from EVEN buffer (output of pad+split)
    StencilDeviceProgram stencil_program = create_stencil_program(
        TT_WAVELET_SOURCE_DIR,
        core,
        *(even_buffer->get_backing_buffer()),  // ← INPUT: even stream from pad+split
        *(stencil_out_buffer->get_backing_buffer()),
        stencil_config);

    {
        tt::tt_metal::distributed::MeshWorkload workload;
        workload.add_program(
            tt::tt_metal::distributed::MeshCoordinateRange(mesh_device.shape()), std::move(stencil_program.program));
        tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
        tt::tt_metal::distributed::Finish(command_queue);
    }

    // Read stencil output
    std::vector<float> device_stencil;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, device_stencil, stencil_out_buffer, true);

    // Host reference: conv(even, h)
    // Use the host pad+split even as the signal for convolution
    std::vector<float> even_signal(
        host_ps.even.begin(), host_ps.even.begin() + static_cast<long>(ps_layout.output.even.length));
    std::vector<float> host_stencil = reference_stencil_1d(even_signal, predict_coeffs);

    print_vec("Stencil Host Reference:", host_stencil, host_stencil.size());
    print_vec("Stencil Device (row 0):", device_stencil, 32);

    // ── 5. Verify: odd_new = odd + stencil(even) ──
    // This is what the full predict step should produce.
    std::cout << "\n--- Full predict step verification (host only) ---\n";
    std::vector<float> odd_after_predict(ps_layout.output.odd.length);
    for (size_t i = 0; i < ps_layout.output.odd.length; ++i) {
        odd_after_predict[i] = host_ps.odd[i] + (i < host_stencil.size() ? host_stencil[i] : 0.0F);
    }
    print_vec("odd_new = odd + conv(even, h):", odd_after_predict, ps_layout.output.odd.length);

    bool overall = ps_ok;
    std::cout << "\n=== Overall: " << (overall ? "PASS" : "FAIL") << " ===\n\n";
    return overall;
}

int main() {
    constexpr int device_id = 0;
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    tt::tt_metal::distributed::MeshCommandQueue& command_queue = mesh_device->mesh_command_queue();
    constexpr tt::tt_metal::CoreCoord core{0, 0};

    bool all_pass = test_pad_split_stencil(*mesh_device, command_queue, core);

    if (all_pass) {
        std::cout << "ALL TESTS PASSED!\n";
    } else {
        std::cout << "SOME TESTS FAILED!\n";
    }

    return all_pass ? 0 : 1;
}
