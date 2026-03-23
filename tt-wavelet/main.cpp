#include <cmath>
#include <concepts>
#include <filesystem>
#include <iostream>
#include <new>
#include <random>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt_stl/assert.hpp>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"

inline tt::tt_metal::CBHandle create_cir_buff(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreCoord& core,
    tt::CBIndex cb_idx,
    const uint32_t num_tiles_in_cb,
    const uint32_t tile_size_bytes,
    tt::DataFormat data_format = tt::DataFormat::Float32) {
    const tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(num_tiles_in_cb * tile_size_bytes, {{cb_idx, data_format}})
            .set_page_size(cb_idx, tile_size_bytes);
    return tt::tt_metal::CreateCircularBuffer(program, core, cb_config);
};

int main() {
    using DataT = float;
    constexpr uint32_t elems_per_tile{tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH};
    constexpr uint32_t tile_size_bytes{sizeof(DataT) * elems_per_tile};
    constexpr int device_id{0};
    const auto mesh_device{tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id)};

    std::filesystem::path reader_kernel = "tt-wavelet/kernels/dataflow/read.cpp";
    std::filesystem::path writer_kernel = "tt-wavelet/kernels/dataflow/write.cpp";
    std::filesystem::path compute_kernel = "tt-wavelet/kernels/compute/compute.cpp";

    tt::tt_metal::CoreCoord core{0, 0};
    tt::tt_metal::Program program{tt::tt_metal::CreateProgram()};

    tt::tt_metal::distributed::MeshCommandQueue& command_queue = mesh_device->mesh_command_queue();

    tt::tt_metal::distributed::DeviceLocalBufferConfig buffer_conf{
        .page_size = tile_size_bytes, .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt::tt_metal::distributed::ReplicatedBufferConfig dram_config{.size = static_cast<uint64_t>(tile_size_bytes)};

    auto base{tt::tt_metal::distributed::MeshBuffer::create(dram_config, buffer_conf, mesh_device.get())};
    auto in0{tt::tt_metal::distributed::MeshBuffer::create(dram_config, buffer_conf, mesh_device.get())};
    auto in1{tt::tt_metal::distributed::MeshBuffer::create(dram_config, buffer_conf, mesh_device.get())};
    auto out{tt::tt_metal::distributed::MeshBuffer::create(dram_config, buffer_conf, mesh_device.get())};
    std::vector<float> base_signal(1024);
    std::vector<float> in0_signal(1024);
    std::vector<float> in1_signal(1024);

    for (auto& val : base_signal) {
        val = 10.0f;
    }

    for (auto& val : in0_signal) {
        val = 2.0f;
    }
    for (auto& val : in1_signal) {
        val = 3.0f;
    }

    base_signal = tilize_nfaces(base_signal, 32, 32);
    in0_signal = tilize_nfaces(in0_signal, 32, 32);
    in1_signal = tilize_nfaces(in1_signal, 32, 32);

    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, base, base_signal, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, in0, in0_signal, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, in1, in1_signal, false);

    tt::tt_metal::CBHandle base_cb = create_cir_buff(program, core, tt::CBIndex::c_0, 1, tile_size_bytes);
    tt::tt_metal::CBHandle in0_cb = create_cir_buff(program, core, tt::CBIndex::c_1, 1, tile_size_bytes);
    tt::tt_metal::CBHandle in1_cb = create_cir_buff(program, core, tt::CBIndex::c_2, 1, tile_size_bytes);
    tt::tt_metal::CBHandle out_cb = create_cir_buff(program, core, tt::CBIndex::c_16, 1, tile_size_bytes);

    std::vector<uint32_t> reader_args;
    std::vector<uint32_t> writer_args;
    tt::tt_metal::TensorAccessorArgs(base->get_backing_buffer()).append_to(reader_args);
    tt::tt_metal::TensorAccessorArgs(in0->get_backing_buffer()).append_to(reader_args);
    tt::tt_metal::TensorAccessorArgs(in1->get_backing_buffer()).append_to(reader_args);
    tt::tt_metal::TensorAccessorArgs(out->get_backing_buffer()).append_to(writer_args);

    auto reader = tt::tt_metal::CreateKernel(
        program,
        reader_kernel,
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::RISCV_0_default,
            .compile_args = reader_args});

    auto writer = tt::tt_metal::CreateKernel(
        program,
        writer_kernel,
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::RISCV_1_default,
            .compile_args = writer_args});

    auto compute = tt::tt_metal::CreateKernel(
        program, compute_kernel, core, tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

    tt::tt_metal::SetRuntimeArgs(
        program,
        reader,
        core,
        {static_cast<uint32_t>(base->address()),
         static_cast<uint32_t>(in0->address()),
         static_cast<uint32_t>(in1->address())});
    tt::tt_metal::SetRuntimeArgs(program, writer, core, {static_cast<uint32_t>(out->address())});

    tt::tt_metal::distributed::MeshWorkload workload;
    auto dev_range{tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape())};
    workload.add_program(dev_range, std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
    tt::tt_metal::distributed::Finish(command_queue);

    std::vector<float> res;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, res, out, true);
    res = untilize_nfaces(res, 32, 32);
    constexpr float expected = 10.25f;
    constexpr uint32_t preview_count = 16;
    std::cout << "preview(" << preview_count << "): ";
    for (uint32_t i = 0; i < preview_count && i < res.size(); ++i) {
        std::cout << res[i] << " ";
    }
    std::cout << "\nexpected each value ~= " << expected << std::endl;
    return 0;
}
