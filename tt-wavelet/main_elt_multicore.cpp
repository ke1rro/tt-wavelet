#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/work_split.hpp"

static constexpr uint32_t num_tile_pairs{64};
static constexpr uint32_t num_output_tiles{2 * num_tile_pairs};
static constexpr uint32_t elems_per_tile{tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH};
static constexpr uint32_t size_tile_bytes{sizeof(bfloat16) * elems_per_tile};

int main() {
    bool pass = true;
    std::string reader_kernel_path = "tt-wavelet/kernels/reader_tiles_multicore.cpp";
    std::string writer_kernel_path = "tt-wavelet/kernels/writer_tiles_multicore.cpp";
    std::string compute_kernel_path = "tt-wavelet/kernels/compute_tiles_multicore.cpp";
    constexpr int device_id{0};

    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    auto& command_queue = mesh_device->mesh_command_queue();
    tt::tt_metal::Program program{tt::tt_metal::CreateProgram()};

    auto core_grid = mesh_device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
        tt::tt_metal::split_work_to_cores(core_grid, num_tile_pairs, true);

    TT_ASSERT(num_cores > 0, "No cores available for multicore wavelet transform");

    tt::tt_metal::distributed::DeviceLocalBufferConfig buffer_config{
        .page_size = size_tile_bytes, .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt::tt_metal::distributed::ReplicatedBufferConfig src_dram_config{
        .size = static_cast<uint64_t>(size_tile_bytes * num_tile_pairs)};
    tt::tt_metal::distributed::ReplicatedBufferConfig dst_dram_config{
        .size = static_cast<uint64_t>(size_tile_bytes * num_output_tiles)};

    auto src0_dram_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(src_dram_config, buffer_config, mesh_device.get());
    auto src1_dram_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(src_dram_config, buffer_config, mesh_device.get());
    auto dst_dram_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(dst_dram_config, buffer_config, mesh_device.get());

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    std::vector<bfloat16> src0_data(static_cast<size_t>(elems_per_tile * num_tile_pairs));
    std::vector<bfloat16> src1_data(static_cast<size_t>(elems_per_tile * num_tile_pairs));

    for (auto& val : src0_data) {
        val = static_cast<bfloat16>(distribution(rng));
    }
    for (auto& val : src1_data) {
        val = static_cast<bfloat16>(distribution(rng));
    }

    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, src0_dram_buffer, src0_data, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, src1_dram_buffer, src1_data, false);

    constexpr uint32_t input_cb_tiles{2};
    constexpr uint32_t output_cb_tiles{4};
    const auto cb_data_format = tt::DataFormat::Float16_b;

    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(
            input_cb_tiles * size_tile_bytes, {{tt::CBIndex::c_0, cb_data_format}})
            .set_page_size(tt::CBIndex::c_0, size_tile_bytes));

    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(
            input_cb_tiles * size_tile_bytes, {{tt::CBIndex::c_1, cb_data_format}})
            .set_page_size(tt::CBIndex::c_1, size_tile_bytes));

    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(
            output_cb_tiles * size_tile_bytes, {{tt::CBIndex::c_16, cb_data_format}})
            .set_page_size(tt::CBIndex::c_16, size_tile_bytes));

    std::vector<uint32_t> reader_compile_args;
    tt::tt_metal::TensorAccessorArgs(src0_dram_buffer->get_backing_buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(src1_dram_buffer->get_backing_buffer()).append_to(reader_compile_args);

    std::vector<uint32_t> writer_compile_args;
    tt::tt_metal::TensorAccessorArgs(dst_dram_buffer->get_backing_buffer()).append_to(writer_compile_args);

    auto reader = tt::tt_metal::CreateKernel(
        program,
        reader_kernel_path,
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::RISCV_1_default,
            .compile_args = reader_compile_args});

    auto writer = tt::tt_metal::CreateKernel(
        program,
        writer_kernel_path,
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::RISCV_0_default,
            .compile_args = writer_compile_args});

    auto compute = tt::tt_metal::CreateKernel(
        program,
        compute_kernel_path,
        all_cores,
        tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

    uint32_t assigned_tile_pairs = 0;
    auto work_groups = {
        std::make_pair(core_group_1, work_per_core1),
        std::make_pair(core_group_2, work_per_core2),
    };

    for (const auto& [core_group, work_per_core] : work_groups) {
        if (work_per_core == 0) {
            continue;
        }

        for (const auto& range : core_group.ranges()) {
            for (const auto& core : range) {
                tt::tt_metal::SetRuntimeArgs(
                    program,
                    reader,
                    core,
                    {static_cast<uint32_t>(src0_dram_buffer->address()),
                     static_cast<uint32_t>(src1_dram_buffer->address()),
                     work_per_core,
                     assigned_tile_pairs});

                tt::tt_metal::SetRuntimeArgs(
                    program,
                    writer,
                    core,
                    {static_cast<uint32_t>(dst_dram_buffer->address()), 2 * work_per_core, 2 * assigned_tile_pairs});

                tt::tt_metal::SetRuntimeArgs(program, compute, core, {work_per_core});
                assigned_tile_pairs += work_per_core;
            }
        }
    }

    TT_ASSERT(assigned_tile_pairs == num_tile_pairs, "Work split mismatch for multicore wavelet transform");

    tt::tt_metal::distributed::MeshWorkload workload;
    auto device_range = tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
    tt::tt_metal::distributed::Finish(command_queue);

    std::vector<bfloat16> result_vec;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, result_vec, dst_dram_buffer, true);

    constexpr float inv_sqrt2 = 0.707106781186548f;
    constexpr float eps = 2e-2f;
    if (result_vec.size() != static_cast<size_t>(elems_per_tile) * num_output_tiles) {
        pass = false;
        std::cout << "Unexpected output size: " << result_vec.size() << "\n";
    }

    if (pass) {
        for (size_t tile_idx = 0; tile_idx < num_tile_pairs; ++tile_idx) {
            const size_t src_tile_offset = tile_idx * elems_per_tile;
            const size_t low_tile_offset = (2 * tile_idx) * elems_per_tile;
            const size_t high_tile_offset = (2 * tile_idx + 1) * elems_per_tile;

            for (size_t e = 0; e < elems_per_tile; ++e) {
                const float even = static_cast<float>(src0_data[src_tile_offset + e]);
                const float odd = static_cast<float>(src1_data[src_tile_offset + e]);

                const float expected_low = (even + odd) * inv_sqrt2;
                const float expected_high = (odd - even) * inv_sqrt2;

                const float actual_low = static_cast<float>(result_vec[low_tile_offset + e]);
                const float actual_high = static_cast<float>(result_vec[high_tile_offset + e]);

                if (std::abs(expected_low - actual_low) > eps) {
                    pass = false;
                    std::cout << "Low mismatch at tile " << tile_idx << ", elem " << e << ": " << actual_low << " vs "
                              << expected_low << "\n";
                    break;
                }

                if (std::abs(expected_high - actual_high) > eps) {
                    pass = false;
                    std::cout << "High mismatch at tile " << tile_idx << ", elem " << e << ": " << actual_high
                              << " vs " << expected_high << "\n";
                    break;
                }
            }

            if (!pass) {
                break;
            }
        }
    }

    if (pass) {
        std::cout << "Multicore test passed!\n";
    } else {
        std::cout << "Multicore test failed!\n";
    }

    pass &= mesh_device->close();
    return pass ? 0 : 1;
}
