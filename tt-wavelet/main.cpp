#include <cmath>
#include <concepts>
#include <iostream>
#include <new>
#include <random>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"

template <typename T>
concept SupportedType = std::same_as<T, bfloat16> || std::same_as<T, float> || std::same_as<T, uint32_t> ||
                        std::same_as<T, int32_t> || std::same_as<T, uint16_t> || std::same_as<T, uint8_t>;

template <SupportedType T>
struct Config {
public:
    explicit Config(const uint32_t num_tiles) :
        num_tiles(num_tiles),
        num_output_tiles(num_tiles * 2),
        elems_per_tile(tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH),
        size_tile_bytes(sizeof(T) * elems_per_tile) {}

    uint32_t num_tiles{};
    uint32_t num_output_tiles{};
    uint32_t elems_per_tile{};
    uint32_t size_tile_bytes{};
};

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
    constexpr int device_id{0};
    constexpr uint32_t num_tiles{64};
    const Config<bfloat16> cfg{num_tiles};
    constexpr tt::tt_metal::CoreCoord core{
        0,
        0,
    };
    constexpr uint32_t src0_cb_index{tt::CBIndex::c_0}, output_cb_index{tt::CBIndex::c_16};
    constexpr uint32_t num_input_tiles{2};
    const auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    tt::tt_metal::distributed::MeshCommandQueue& command_queue = mesh_device->mesh_command_queue();
    tt::tt_metal::Program program{tt::tt_metal::CreateProgram()};

    tt::tt_metal::distributed::DeviceLocalBufferConfig dram_buff_conf{
        .page_size = cfg.size_tile_bytes,
        .buffer_type = tt::tt_metal::BufferType::DRAM,
    };

    tt::tt_metal::distributed::ReplicatedBufferConfig src_dram_config{
        .size = static_cast<uint64_t>(cfg.size_tile_bytes * num_tiles),
    };
    tt::tt_metal::distributed::ReplicatedBufferConfig dst_dram_config{
        .size = static_cast<uint64_t>(cfg.size_tile_bytes * cfg.num_output_tiles),
    };

    auto src_dram_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(src_dram_config, dram_buff_conf, mesh_device.get());
    auto dst_dram_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(dst_dram_config, dram_buff_conf, mesh_device.get());

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.f, 1.0f);
    std::vector<bfloat16> src0_vec(
        static_cast<std::size_t>(cfg.num_tiles) * static_cast<std::size_t>(cfg.elems_per_tile));
    for (bfloat16& v : src0_vec) {
        v = bfloat16(dist(rng));
    }
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * cfg.size_tile_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, cfg.size_tile_bytes);

    tt::tt_metal::CircularBufferConfig cb_out_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * cfg.size_tile_bytes, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, cfg.size_tile_bytes);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, src_dram_buffer, src0_vec, false);

    std::vector<uint32_t> reader_args;
    tt::tt_metal::TensorAccessorArgs(src_dram_buffer->get_backing_buffer()).append_to(reader_args);
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt-wavelet/kernels/reader.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::RISCV_0_default,
            .compile_args = reader_args});

    std::vector<uint32_t> writer_args_comp;
    tt::tt_metal::TensorAccessorArgs(dst_dram_buffer->get_backing_buffer()).append_to(writer_args_comp);
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt-wavelet/kernels/writer.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::RISCV_1_default,
            .compile_args = writer_args_comp});

    tt::tt_metal::KernelHandle eltwise_sfpu_kernel_id = CreateKernel(
        program,
        "eltwise_sfpu/kernels/compute/eltwise_sfpu.cpp",
        core,
        tt::tt_metal::ComputeConfig{
            .math_approx_mode = false,
        });

    tt::tt_metal::SetRuntimeArgs(program, eltwise_sfpu_kernel_id, core, {cfg.num_tiles});
    tt::tt_metal::SetRuntimeArgs(
        program, reader_kernel_id, core, {static_cast<uint32_t>(src_dram_buffer->address()), cfg.num_tiles});
    tt::tt_metal::SetRuntimeArgs(
        program, writer_kernel_id, core, {static_cast<uint32_t>(dst_dram_buffer->address()), cfg.num_tiles});

    tt::tt_metal::distributed::MeshWorkload workload;
    auto device_range = tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
    tt::tt_metal::distributed::Finish(command_queue);

    std::vector<bfloat16> result_vec;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, result_vec, dst_dram_buffer, true);

    constexpr float eps = 2e-2f;
    for (uint32_t i = 0; i < result_vec.size(); ++i) {
        const float expected = static_cast<float>(bfloat16(std::exp(static_cast<float>(src0_vec[i]))));
        const float result = static_cast<float>(result_vec[i]);
        if (std::abs(expected - result) > eps) {
            pass = false;
            log_error(tt::LogTest, "Result mismatch at index {}: {} != {}", i, expected, result);
        }
    }
    pass &= mesh_device->close();
    return pass ? 0 : 1;
}
