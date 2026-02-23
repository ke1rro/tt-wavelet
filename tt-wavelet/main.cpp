#include <cmath>
#include <iostream>
#include <random>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"

static constexpr uint32_t num_tiles{64};
static constexpr uint32_t elems_per_tile{tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH};
static constexpr uint32_t size_tile_bytes{sizeof(bfloat16) * elems_per_tile};

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

int main() {
    bool pass = true;
    std::string reader_kernel_path = "tt-wavelet/kernels/reader_tiles.cpp";
    std::string writer_kernel_path = "tt-wavelet/kernels/writer_tiles.cpp";
    std::string compute_kernel_path = "tt-wavelet/kernels/compute_tiles.cpp";
    constexpr int device_id{0};
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);

    tt::tt_metal::distributed::MeshCommandQueue& command_queue = mesh_device->mesh_command_queue();
    tt::tt_metal::Program program{tt::tt_metal::CreateProgram()};
    constexpr tt::tt_metal::CoreCoord core{0, 0};

    // Buffer conf
    tt::tt_metal::distributed::DeviceLocalBufferConfig buffer_config{
        .page_size = size_tile_bytes, .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt::tt_metal::distributed::ReplicatedBufferConfig dram_config{
        .size = static_cast<tt::tt_metal::DeviceAddr>(size_tile_bytes * num_tiles)};

    auto src0_dram_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(dram_config, buffer_config, mesh_device.get());
    auto src1_dram_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(dram_config, buffer_config, mesh_device.get());
    auto dst_dram_buffer = tt::tt_metal::distributed::MeshBuffer::create(dram_config, buffer_config, mesh_device.get());

    constexpr float val_to_add{-0.1f};
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    std::vector<bfloat16> src0_data(static_cast<size_t>(elems_per_tile * num_tiles));
    std::vector<bfloat16> src1_data(static_cast<size_t>(elems_per_tile * num_tiles), static_cast<bfloat16>(val_to_add));

    for (auto& val : src0_data) {
        val = static_cast<bfloat16>(distribution(rng));
    }
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, src0_dram_buffer, src0_data, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, src1_dram_buffer, src1_data, false);

    constexpr uint8_t tiles_per_buffer{2};

    tt::CBIndex src_cb0_index = tt::CBIndex::c_0;

    tt::tt_metal::CBHandle cb_src0 =
        create_circular_buffer(program, core, tt::CBIndex::c_0, tiles_per_buffer, size_tile_bytes);

    tt::tt_metal::CBHandle cb_src1 =
        create_circular_buffer(program, core, tt::CBIndex::c_1, tiles_per_buffer, size_tile_bytes);

    tt::tt_metal::CBHandle cb_dst =
        create_circular_buffer(program, core, tt::CBIndex::c_16, tiles_per_buffer, size_tile_bytes);

    std::vector<uint32_t> reader_args;
    std::vector<uint32_t> writer_args;
    tt::tt_metal::TensorAccessorArgs(src0_dram_buffer->get_backing_buffer()).append_to(reader_args);
    tt::tt_metal::TensorAccessorArgs(src1_dram_buffer->get_backing_buffer()).append_to(reader_args);
    tt::tt_metal::TensorAccessorArgs(dst_dram_buffer->get_backing_buffer()).append_to(writer_args);
    auto reader = tt::tt_metal::CreateKernel(
        program,
        reader_kernel_path,
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::RISCV_0_default,
            .compile_args = reader_args});
    auto writer = tt::tt_metal::CreateKernel(
        program,
        writer_kernel_path,
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::RISCV_1_default,
            .compile_args = writer_args});

    auto compute = tt::tt_metal::CreateKernel(
        program, compute_kernel_path, core, tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4});
    tt::tt_metal::SetRuntimeArgs(
        program,
        reader,
        core,
        {static_cast<uint32_t>(src0_dram_buffer->address()),
         static_cast<uint32_t>(src1_dram_buffer->address()),
         num_tiles});

    tt::tt_metal::SetRuntimeArgs(program, writer, core, {static_cast<uint32_t>(dst_dram_buffer->address()), num_tiles});

    tt::tt_metal::SetRuntimeArgs(program, compute, core, {num_tiles});

    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange device_range =
        tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
    tt::tt_metal::distributed::Finish(command_queue);

    std::vector<bfloat16> result_vec;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, result_vec, dst_dram_buffer, true);

    constexpr float eps = 1e-2f;  // loose tolerance because of the nature of bfloat16
    for (size_t i = 0; i < result_vec.size(); ++i) {
        const float expected = static_cast<float>(src0_data[i]) + val_to_add;
        const float actual = static_cast<float>(result_vec[i]);

        if (std::abs(expected - actual) > eps) {
            pass = false;
            std::cout << "Wrong " << actual << " vs " << expected << "\n";
        }
    }
    if (pass) {
        std::cout << "Test passed!\n";
    } else {
        std::cout << "Test failed!\n";
    }
    return 0;
}