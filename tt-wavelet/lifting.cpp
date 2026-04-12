#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <span>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_wavelet/include/boundary.hpp"
#include "tt_wavelet/include/lifting/lifting_device.hpp"
#include "tt_wavelet/include/lifting/lifting_plan.hpp"
#include "tt_wavelet/include/pad_split.hpp"

namespace {

constexpr float kTolerance = 1e-5F;

[[nodiscard]] std::filesystem::path default_scheme_path() {
    return std::filesystem::path(TT_WAVELET_SOURCE_DIR) / "../ttnn-wavelet/lifting_schemes/bior3.9.json";
}

[[nodiscard]] bool matches_reference(
    const std::vector<float>& device_values, const std::vector<float>& reference, const size_t logical_length) {
    for (size_t i = 0; i < logical_length; ++i) {
        if (std::fabs(device_values[i] - reference[i]) > kTolerance) {
            std::cerr << "Mismatch at index " << i << ": device=" << device_values[i] << " reference=" << reference[i]
                      << '\n';
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::vector<float> apply_stencil_reference(
    const std::span<const float> source, const std::vector<float>& coefficients, const size_t output_length) {
    std::vector<float> output(output_length, 0.0F);
    const int64_t k = static_cast<int64_t>(coefficients.size());

    for (size_t i = 0; i < output_length; ++i) {
        float sum = 0.0F;
        for (size_t j = 0; j < coefficients.size(); ++j) {
            const int64_t logical_index = static_cast<int64_t>(i) + k - 1 - static_cast<int64_t>(j);
            if (logical_index >= 0 && logical_index < static_cast<int64_t>(source.size())) {
                sum += coefficients[j] * source[static_cast<size_t>(logical_index)];
            }
        }
        output[i] = sum;
    }

    return output;
}

struct HostLiftingReference {
    std::vector<float> even;
    std::vector<float> odd;
};

[[nodiscard]] HostLiftingReference materialize_reference_forward_lifting(
    const std::span<const float> input, const ttwv::LiftingPreprocessDeviceProgram& bundle) {
    const auto split = ttwv::materialize_reference_pad_split(input, bundle.plan.preprocess_layout);

    std::vector<float> even(split.even.begin(), split.even.begin() + static_cast<std::ptrdiff_t>(bundle.plan.preprocess_layout.output.even.length));
    std::vector<float> odd(split.odd.begin(), split.odd.begin() + static_cast<std::ptrdiff_t>(bundle.plan.preprocess_layout.output.odd.length));

    for (size_t step_index = 0; step_index < bundle.scheme.steps.size(); ++step_index) {
        const auto& step = bundle.scheme.steps[step_index];
        switch (step.type) {
            case ttwv::StepType::Predict: {
                const auto stencil = apply_stencil_reference(even, step.coefficients, odd.size());
                for (size_t i = 0; i < odd.size(); ++i) {
                    odd[i] += stencil[i];
                }
                break;
            }
            case ttwv::StepType::Update: {
                const auto stencil = apply_stencil_reference(odd, step.coefficients, even.size());
                for (size_t i = 0; i < even.size(); ++i) {
                    even[i] += stencil[i];
                }
                break;
            }
            case ttwv::StepType::ScaleEven:
                for (auto& value : even) {
                    value *= step.coefficients[0];
                }
                break;
            case ttwv::StepType::ScaleOdd:
                for (auto& value : odd) {
                    value *= step.coefficients[0];
                }
                break;
            case ttwv::StepType::Swap:
                std::swap(even, odd);
                break;
        }
    }

    return HostLiftingReference{.even = std::move(even), .odd = std::move(odd)};
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path scheme_path = argc > 1 ? std::filesystem::path(argv[1]) : default_scheme_path();

    constexpr int device_id = 0;
    constexpr tt::tt_metal::CoreCoord core{0, 0};
    constexpr size_t signal_length = 19;

    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    tt::tt_metal::distributed::MeshCommandQueue& command_queue = mesh_device->mesh_command_queue();

    std::vector<float> original_signal(signal_length);
    for (size_t i = 0; i < signal_length; ++i) {
        original_signal[i] = static_cast<float>(i + 1);
    }

    ttwv::SignalBuffer input_desc{
        .dram_address = 0,
        .length = signal_length,
        .stick_width = 32,
        .element_size_bytes = sizeof(float),
    };

    const tt::tt_metal::distributed::DeviceLocalBufferConfig input_local_config{
        .page_size = input_desc.aligned_stick_bytes(32),
        .buffer_type = tt::tt_metal::BufferType::DRAM,
    };
    const tt::tt_metal::distributed::ReplicatedBufferConfig input_replicated_config{
        .size = static_cast<uint64_t>(input_desc.physical_nbytes()),
    };
    auto input_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(input_replicated_config, input_local_config, mesh_device.get());
    input_desc.dram_address = input_buffer->get_backing_buffer()->address();

    const auto scheme = ttwv::load_runtime_lifting_scheme(scheme_path);
    auto bundle = ttwv::create_lifting_preprocess_program(
        TT_WAVELET_SOURCE_DIR, *mesh_device, core, *(input_buffer->get_backing_buffer()), input_desc, scheme);

    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, input_buffer, original_signal, false);
    ttwv::run_preprocess(command_queue, *mesh_device, bundle);
    const auto active_streams =
        ttwv::execute_forward_lifting(TT_WAVELET_SOURCE_DIR, *mesh_device, command_queue, core, bundle);

    std::vector<float> device_even_result;
    std::vector<float> device_odd_result;

    auto even_buffer = active_streams.even.family == ttwv::LogicalStream::Even
                           ? bundle.buffers.even.at(active_streams.even.slot)
                           : bundle.buffers.odd.at(active_streams.even.slot);
    auto odd_buffer = active_streams.odd.family == ttwv::LogicalStream::Even
                          ? bundle.buffers.even.at(active_streams.odd.slot)
                          : bundle.buffers.odd.at(active_streams.odd.slot);
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, device_even_result, even_buffer, true);
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, device_odd_result, odd_buffer, true);

    const auto reference = materialize_reference_forward_lifting(original_signal, bundle);

    const bool even_matches =
        matches_reference(device_even_result, reference.even, bundle.plan.even_active_buffer().length);
    const bool odd_matches =
        matches_reference(device_odd_result, reference.odd, bundle.plan.odd_active_buffer().length);

    std::cout << "Scheme: " << scheme_path << '\n';
    std::cout << "Steps executed: " << bundle.scheme.steps.size() << '\n';
    std::cout << "Even logical length: " << bundle.plan.even_active_buffer().length << '\n';
    std::cout << "Odd logical length: " << bundle.plan.odd_active_buffer().length << '\n';

    if (even_matches && odd_matches) {
        std::cout << "SUCCESS: lifting preprocess and forward LWT match host reference.\n";
        return EXIT_SUCCESS;
    }

    return EXIT_FAILURE;
}
