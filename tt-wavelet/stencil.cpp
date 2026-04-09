#include <bit>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "../tt-metal/tt_metal/api/tt-metalium/base_types.hpp"
#include "../tt-metal/tt_metal/api/tt-metalium/distributed.hpp"
#include "../tt-metal/tt_metal/api/tt-metalium/host_api.hpp"
#include "../tt-metal/tt_metal/api/tt-metalium/kernel_types.hpp"
#include "../tt-metal/tt_metal/api/tt-metalium/mesh_coord.hpp"
#include "../tt-metal/tt_metal/api/tt-metalium/mesh_trace_id.hpp"
#include "../tt-metal/tt_metal/api/tt-metalium/mesh_workload.hpp"
#include "../tt-metal/tt_metal/api/tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/tilize_utils.hpp"
#include "tt_wavelet/include/pad_split.hpp"
#include "tt_wavelet/include/pad_split_device.hpp"

inline tt::tt_metal::CBHandle create_circular_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreCoord& core,
    tt::CBIndex cb_index,
    const uint32_t num_tiles_in_cb,
    const uint32_t tile_size_bytes,
    tt::DataFormat data_format = tt::DataFormat::Float16_b) {
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

uint8_t K(const uint8_t k) { return 17 - k; }

uint8_t halo_pad(const uint8_t k) { return 32 - K(k); }

void push_step_coeffs_to_compile_args(std::vector<uint32_t>& args, std::vector<float>& coeffs) {
    for (size_t i{0}; i < coeffs.size(); ++i) {
        args.push_back(std::bit_cast<uint32_t>(coeffs[i]));
    }
}

int main() {
    constexpr uint32_t num_tiles_input{2};
    constexpr uint32_t num_tiles_output{1};
    constexpr uint32_t elems_per_tile{tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH};
    uint32_t size_tile_b = sizeof(float) * elems_per_tile;
    constexpr int device_id{0};
    std::string reader{"tt-wavelet/kernels/dataflow/stencil_read.cpp"};
    std::string writer{"tt-wavelet/kernels/dataflow/stencil_write.cpp"};
    std::string compute{"tt-wavelet/kernels/stencil_compute.cpp"};
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);

    tt::tt_metal::distributed::MeshCommandQueue& command_queue = mesh_device->mesh_command_queue();
    auto program = tt::tt_metal::CreateProgram();

    constexpr tt::tt_metal::CoreCoord core{0, 0};

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
    std::vector<float> coeffs = {-0.49999999999999994, -0.49999999999999994};
    const uint8_t k = coeffs.size();
    const uint8_t halo_idx = halo_pad(k);

    size_t sig = 1.0f;
    for (size_t j = 0; j < halo_idx; ++j) {
        halo[halo_idx + j] = sig++;
    }

    for (size_t j = 0; j < 32 - halo_idx; ++j) {
        input[j] = sig++;
    }
    print_array(halo);
    print_array(input);

    std::vector<float> halo_tile(elems_per_tile, 0.0f);
    std::vector<float> input_tile(elems_per_tile, 0.0f);
    std::copy(halo.begin(), halo.end(), halo_tile.begin());
    std::copy(input.begin(), input.end(), input_tile.begin());

    const std::vector<uint32_t> tile_shape = {1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};

    auto tiled_halo = convert_layout(
        tt::stl::make_const_span(halo_tile),
        tile_shape,
        TensorLayoutType::LIN_ROW_MAJOR,
        TensorLayoutType::TILED_NFACES);

    auto tiled_input = convert_layout(
        tt::stl::make_const_span(input_tile),
        tile_shape,
        TensorLayoutType::LIN_ROW_MAJOR,
        TensorLayoutType::TILED_NFACES);

    std::vector<float> combined((tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH) * 2);

    std::copy(tiled_halo.begin(), tiled_halo.end(), combined.begin());
    std::copy(
        tiled_input.begin(),
        tiled_input.end(),
        combined.begin() + tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH);

    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, input_dram_buffer, combined, false);

    constexpr tt::CBIndex halo_cb = tt::CBIndex::c_0;
    constexpr tt::CBIndex input_cb = tt::CBIndex::c_1;
    constexpr tt::CBIndex output_cb = tt::CBIndex::c_16;

    create_circular_buffer(program, core, halo_cb, 1, size_tile_b);
    create_circular_buffer(program, core, input_cb, 1, size_tile_b);
    create_circular_buffer(program, core, output_cb, 1, size_tile_b);

    std::vector<uint32_t> reader_compile_args;
    std::vector<uint32_t> writer_compile_args;
    reader_compile_args.push_back(static_cast<uint32_t>(halo_cb));
    reader_compile_args.push_back(static_cast<uint32_t>(input_cb));
    writer_compile_args.push_back(static_cast<uint32_t>(output_cb));
    std::vector<uint32_t> compute_compile_args;
    compute_compile_args.push_back(static_cast<uint32_t>(k));
    compute_compile_args.push_back(static_cast<uint32_t>(halo_cb));
    compute_compile_args.push_back(static_cast<uint32_t>(input_cb));
    compute_compile_args.push_back(static_cast<uint32_t>(output_cb));
    // Took bior 2.8  [{"type": "update", "shift": -1, "coefficients": [-0.49999999999999994, -0.49999999999999994]}

    push_step_coeffs_to_compile_args(compute_compile_args, coeffs);

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

    tt::tt_metal::SetRuntimeArgs(program, wtiter_ker, core, {static_cast<uint32_t>(output_dram_buffer->address())});
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange device_range =
        tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
    tt::tt_metal::distributed::Finish(command_queue);

    std::vector<float> result_of_stencil;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, result_of_stencil, output_dram_buffer);

    print_array(result_of_stencil);
}
