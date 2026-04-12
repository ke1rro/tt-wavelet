#include <bit>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/tilize_utils.hpp"
#include "tt_wavelet/include/pad_split.hpp"
#include "tt_wavelet/include/pad_split_device.hpp"
#include "tt_wavelet/include/device_step_desc.hpp"

inline tt::tt_metal::CBHandle create_circular_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreCoord& core,
    tt::CBIndex cb_index,
    const uint32_t num_tiles_in_cb,
    const uint32_t tile_size_bytes,
    tt::DataFormat data_format = tt::DataFormat::Float32) {
    const tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(num_tiles_in_cb * tile_size_bytes, {{cb_index, data_format}})
            .set_page_size(cb_index, tile_size_bytes);

    return tt::tt_metal::CreateCircularBuffer(program, core, cb_config);
}

template <typename T>
void print_array(const std::vector<T>& arr) {
    std::cout << "[";
    for (size_t i = 0; i < arr.size(); ++i) {
        std::cout << arr[i] << ", ";
    }
    std::cout << "]" << std::endl;
}

template <typename T>
void print_prefix(const std::vector<T>& arr, const size_t count, const std::string& label) {
    std::cout << label << " [";
    for (size_t i = 0; i < std::min(count, arr.size()); ++i) {
        std::cout << arr[i] << ", ";
    }
    std::cout << "]" << std::endl;
}

uint8_t K(const uint8_t k) { return 17 - k; }

uint8_t halo_pad(const uint8_t k) { return K(k); }

ttwv::device::DeviceStepDesc
make_device_step_desc(const ttwv::device::DeviceStepType type, const std::vector<float>& coeffs) {
    assert(coeffs.size() <= ttwv::device::step_coeff_capacity);

    ttwv::device::DeviceStepDesc desc{};
    desc.type = static_cast<uint32_t>(type);
    desc.k = static_cast<uint32_t>(coeffs.size());
    for (size_t i = 0; i < coeffs.size(); ++i) {
        desc.coeffs_packed[i] = std::bit_cast<uint32_t>(coeffs[i]);
    }
    return desc;
}

std::vector<uint32_t> device_step_desc_to_runtime_args(const ttwv::device::DeviceStepDesc& desc) {
    std::vector<uint32_t> args(ttwv::device::step_desc_word_count, 0);
    static_assert(sizeof(ttwv::device::DeviceStepDesc) == ttwv::device::step_desc_word_count * sizeof(uint32_t));
    std::memcpy(args.data(), &desc, sizeof(desc));
    return args;
}

std::vector<float> cpu_stencil_reference(
    const std::vector<float>& halo,
    const std::vector<float>& input,
    const std::vector<float>& coeffs) {
    const size_t coeff_count = coeffs.size();

    std::vector<float> padded_signal(64, 0.0f);
    for (size_t i = 0; i < 32; ++i) {
        padded_signal[i] = halo[i];
        padded_signal[32 + i] = input[i];
    }

    std::vector<float> stencil_output(64, 0.0f);
    for (size_t n = 0; n < stencil_output.size(); ++n) {
        float sum = 0.0f;
        for (size_t j = 0; j < coeff_count; ++j) {
            if (n >= j) {
                sum += coeffs[j] * padded_signal[n - j];
            }
        }
        stencil_output[n] = sum;
    }

    std::vector<float> expected(32, 0.0f);
    for (size_t i = 0; i < 16; ++i) {
        expected[i] = stencil_output[16 + i];
        expected[16 + i] = stencil_output[32 + i];
    }
    return expected;
}

int main() {
    constexpr uint32_t num_tiles_input{2};
    constexpr uint32_t num_tiles_output{1};
    constexpr uint32_t elems_per_tile{tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH};
    constexpr tt::DataFormat cb_data_format = tt::DataFormat::Float32;
    const uint32_t size_tile_b = tt::tile_size(cb_data_format);
    constexpr int device_id{0};
    std::string reader{"tt-wavelet/kernels/dataflow/stencil_read.cpp"};
    std::string writer{"tt-wavelet/kernels/dataflow/stencil_write.cpp"};
    std::string compute{"tt-wavelet/kernels/stencil_compute.cpp"};
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);

    tt::tt_metal::distributed::MeshCommandQueue& command_queue = mesh_device->mesh_command_queue();
    auto program = tt::tt_metal::CreateProgram();

    constexpr tt::tt_metal::CoreCoord core{1, 0};

    tt::tt_metal::distributed::DeviceLocalBufferConfig buffer_config{
        .page_size = size_tile_b, .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt::tt_metal::distributed::ReplicatedBufferConfig input_dram_config{
        .size = static_cast<uint64_t>(size_tile_b * num_tiles_input)};
    tt::tt_metal::distributed::ReplicatedBufferConfig output_dram_config{
        .size = static_cast<uint64_t>(size_tile_b * num_tiles_output)};

    auto input_dram_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(input_dram_config, buffer_config, mesh_device.get());
    auto output_dram_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(output_dram_config, buffer_config, mesh_device.get());

    std::vector<float> halo(32, 0.0f);
    std::vector<float> input(32, 0.0f);
    std::vector<float> coeffs = {1.0f, 1.0f, 1.0f};
    const auto step_desc = make_device_step_desc(ttwv::device::DeviceStepType::Predict, coeffs);
    const uint8_t k = coeffs.size();
    const uint8_t halo_idx = halo_pad(k);

    size_t sig = 1;
    const size_t halo_signal_count = halo.size() - halo_idx;
    for (size_t j = 0; j < halo_signal_count; ++j) {
        halo[halo_idx + j] = static_cast<float>(sig++);
    }

    const size_t input_signal_count = std::min(input.size(), static_cast<size_t>(halo_idx));
    for (size_t j = 0; j < input_signal_count; ++j) {
        input[j] = static_cast<float>(sig++);
    }

    std::cout << "Halo:";
    print_array(halo);
    std::cout << "\n";
    std::cout << "Input:";
    print_array(input);
    std::cout << "\n";

    std::vector<float> halo_tile(elems_per_tile, 0.0f);
    std::vector<float> input_tile(elems_per_tile, 0.0f);
    std::copy(halo.begin(), halo.end(), halo_tile.begin());
    std::copy(input.begin(), input.end(), input_tile.begin());

    auto tiled_halo = tilize_nfaces(halo_tile, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH);
    auto tiled_input = tilize_nfaces(input_tile, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH);

    std::vector<float> combined((tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH) * 2);
    std::cout << "Tiled Halo:" << std::endl;
    print_array(tiled_halo);
    std::cout << "Tiled Input:" << std::endl;
    print_array(tiled_input);

    std::copy(tiled_halo.begin(), tiled_halo.end(), combined.begin());
    std::copy(
        tiled_input.begin(),
        tiled_input.end(),
        combined.begin() + tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH);
    std::cout << "Combined Tiles (halo followed by input):" << std::endl;
    print_array(combined);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, input_dram_buffer, combined, false);

    constexpr tt::CBIndex halo_cb = tt::CBIndex::c_0;
    constexpr tt::CBIndex input_cb = tt::CBIndex::c_1;
    constexpr tt::CBIndex output_cb = tt::CBIndex::c_16;

    create_circular_buffer(program, core, halo_cb, 1, size_tile_b, cb_data_format);
    create_circular_buffer(program, core, input_cb, 1, size_tile_b, cb_data_format);
    create_circular_buffer(program, core, output_cb, 1, size_tile_b, cb_data_format);

    std::vector<uint32_t> reader_compile_args;
    std::vector<uint32_t> writer_compile_args;
    reader_compile_args.push_back(static_cast<uint32_t>(halo_cb));
    reader_compile_args.push_back(static_cast<uint32_t>(input_cb));
    writer_compile_args.push_back(static_cast<uint32_t>(output_cb));
    std::vector<uint32_t> compute_compile_args;
    compute_compile_args.push_back(static_cast<uint32_t>(halo_cb));
    compute_compile_args.push_back(static_cast<uint32_t>(input_cb));
    compute_compile_args.push_back(static_cast<uint32_t>(output_cb));

    tt::tt_metal::TensorAccessorArgs(input_dram_buffer->get_backing_buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_dram_buffer->get_backing_buffer()).append_to(writer_compile_args);
    auto reader_ker = tt::tt_metal::CreateKernel(
        program,
        reader,
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::RISCV_0_default,
            .compile_args = reader_compile_args,
        });

    auto wtiter_ker = tt::tt_metal::CreateKernel(
        program,
        writer,
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::RISCV_1_default,
            .compile_args = writer_compile_args,
        });

    auto compute_ker = tt::tt_metal::CreateKernel(
        program,
        compute,
        core,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .compile_args = compute_compile_args});

    tt::tt_metal::SetRuntimeArgs(program, reader_ker, core, {static_cast<uint32_t>(input_dram_buffer->address())});
    tt::tt_metal::SetRuntimeArgs(program, compute_ker, core, device_step_desc_to_runtime_args(step_desc));

    tt::tt_metal::SetRuntimeArgs(program, wtiter_ker, core, {static_cast<uint32_t>(output_dram_buffer->address())});
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange device_range =
        tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
    tt::tt_metal::distributed::Finish(command_queue);

    std::vector<float> result_of_stencil;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, result_of_stencil, output_dram_buffer);
    std::cout << "Result of stencil computation (face-based tile):" << std::endl;
    print_array(result_of_stencil);

    auto result_row_major = untilize_nfaces(result_of_stencil, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH);
    std::cout << "Result row-major row 0:" << std::endl;
    print_prefix(result_row_major, 32, "hw_output:");

    auto expected = cpu_stencil_reference(halo, input, coeffs);
    std::cout << "CPU reference row 0:" << std::endl;
    print_prefix(expected, 32, "expected:");

    bool pass = true;
    constexpr float tolerance = 1e-3f;
    for (size_t i = 0; i < 32; ++i) {
        float hw = result_row_major[i];
        float ref = expected[i];
        if (std::abs(hw - ref) > tolerance) {
            std::cerr << "MISMATCH at col " << i << ": hw=" << hw << " ref=" << ref
                      << " diff=" << std::abs(hw - ref) << std::endl;
            pass = false;
        }
    }

    std::cout << (pass ? "PASS: All 32 output values match CPU reference."
                       : "FAIL: Output mismatch detected.")
              << std::endl;

    return pass ? 0 : 1;
}
