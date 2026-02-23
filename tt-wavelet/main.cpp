#include <iostream>
#include <random>
#include <vector>

#include "tracy/Tracy.hpp"

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"

static constexpr uint32_t num_tiles{50};
static constexpr uint32_t elements_per_tile{tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT};
static constexpr uint32_t tile_size_bytes{sizeof(bfloat16) * elements_per_tile};
static constexpr uint32_t dram_buffer_size{tile_size_bytes * num_tiles};

int main() {
    ZoneScoped;
    std::cin.get();  // Wait for Tracy GUI to connect

     // Initialize MeshDevice and CommandQueue
    bool pass = true;
    constexpr int device_id{0};
    
    const auto mesh_device = [device_id]() {
        ZoneScopedN("Device Init");
        return tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    }();
    tt::tt_metal::distributed::MeshCommandQueue& command_queue = mesh_device->mesh_command_queue();
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    std::cout << "program created\n";
    TracyMessageC("Program created", 15, 0x00FF00);

    const tt::tt_metal::distributed::DeviceLocalBufferConfig l1_config{
        .page_size = tile_size_bytes,
        .buffer_type = tt::tt_metal::BufferType::L1,
    };
    tt::tt_metal::distributed::ReplicatedBufferConfig l1_buffer_config{.size = tile_size_bytes};

    const tt::tt_metal::distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_size_bytes,
        .buffer_type = tt::tt_metal::BufferType::DRAM,
    };

    tt::tt_metal::distributed::ReplicatedBufferConfig dram_buffer_config{.size = dram_buffer_size};

    const auto l1_buffer = [&]() {
        ZoneScopedN("Buffer Creation");
        auto buf = tt::tt_metal::distributed::MeshBuffer::create(l1_buffer_config, l1_config, mesh_device.get());
        TracyPlot("Buffer Size (bytes)", static_cast<int64_t>(dram_buffer_size));
        return buf;
    }();
    auto input_dram_buffer = tt::tt_metal::distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
    auto output_dram_buffer = tt::tt_metal::distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());

    std::vector<bfloat16> input_vec = [&]() {
        ZoneScopedN("Data Preparation");
        std::vector<bfloat16> vec(static_cast<std::size_t>(elements_per_tile) * num_tiles);
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0f, 100.0f);
        for (auto& elem : vec) {
            elem = static_cast<bfloat16>(dist(rng));
        }
        TracyPlot("Input Elements", static_cast<int64_t>(vec.size()));
        return vec;
    }();
    
    {
        ZoneScopedN("Write Input Buffer");
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, input_dram_buffer, input_vec, false);
        TracyMessageC("Input data written to DRAM", 27, 0x0080FF);
    }
    constexpr tt::tt_metal::CoreCoord core{0, 0};

    std::vector<uint32_t> dram_copy_compile_time_arg;
    tt::tt_metal::TensorAccessorArgs(*input_dram_buffer->get_backing_buffer()).append_to(dram_copy_compile_time_arg);
    tt::tt_metal::TensorAccessorArgs(*output_dram_buffer->get_backing_buffer()).append_to(dram_copy_compile_time_arg);

    tt::tt_metal::KernelHandle dram_copy_kernel_id;
    {
        ZoneScopedN("Kernel Setup");
        dram_copy_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "tt-wavelet/kernel/loopback.cpp",
            core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::RISCV_0_default,
                .compile_args = dram_copy_compile_time_arg});
        TracyPlot("Num Tiles", static_cast<int64_t>(num_tiles));
        TracyMessageC("Kernel configured", 17, 0xFFAA00);
    }

    const std::vector<uint32_t> runtime_args = {
        static_cast<uint32_t>(l1_buffer->address()),
        static_cast<uint32_t>(input_dram_buffer->address()),
        static_cast<uint32_t>(output_dram_buffer->address()),
        static_cast<uint32_t>(num_tiles),
    };
    SetRuntimeArgs(program, dram_copy_kernel_id, core, runtime_args);

    {
        ZoneScopedN("Execute Workload");
        tt::tt_metal::distributed::MeshWorkload workload;
        tt::tt_metal::distributed::MeshCoordinateRange device_range =
            tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());
        workload.add_program(device_range, std::move(program));
        tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, /*blocking=*/false);
        TracyMessageC("Workload enqueued", 17, 0xFF8000);
        tt::tt_metal::distributed::Finish(command_queue);
        TracyMessageC("Workload finished", 17, 0x00FF00);
    }

    std::vector<bfloat16> result_vec;
    {
        ZoneScopedN("Read Output Buffer");
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, result_vec, output_dram_buffer, true);
        TracyMessageC("Output data read from DRAM", 27, 0x0080FF);
    }

    {
        ZoneScopedN("Validate Results");
        for (int i = 0; i < input_vec.size(); i++) {
            if (input_vec[i] != result_vec[i]) {
                pass = false;
                TracyMessageC("Validation failed!", 18, 0xFF0000);
                break;
            }
        }
        if (pass) {
            TracyMessageC("Validation passed!", 18, 0x00FF00);
        }
    }

    if (!mesh_device->close()) {
        pass = false;
    }
    for (size_t i{0}; i < input_vec.size(); i++) {
        std::cout << input_vec[i] << " ";
    }

    fmt::print("Test {}.\n", pass ? "passed" : "failed");

    return 0;
}