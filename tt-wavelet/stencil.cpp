#include <bit>
#include <cassert>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/tilize_utils.hpp"

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
void print_tile(const std::vector<T>& arr, const size_t rows = 32, const size_t cols = 32) {
    std::cout << "[";
    for (size_t i = 0; i < rows; ++i) {
        if(i > 0) {
            std::cout << std::endl << " ";
        }

        for (size_t j = 0; j < cols; ++j) {
            if(j > 0) {
                std::cout << ", ";
            }
            std::cout << arr[i * cols + j];
        }
    }
    std::cout << "]" << std::endl;
}

void push_step_coeffs_to_compile_args(std::vector<uint32_t>& args, std::vector<float>& coeffs) {
    for (size_t i{0}; i < coeffs.size(); ++i) {
        args.push_back(std::bit_cast<uint32_t>(coeffs[i]));
    }
}

std::vector<float> read_floats_from_file(const std::string& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    std::vector<float> values;
    float value = 0.0f;
    while (input >> value) {
        values.push_back(value);
    }

    if (!input.eof() && input.fail()) {
        throw std::runtime_error("Failed while reading floating-point values from: " + path);
    }

    return values;
}

struct KernelSpec {
    uint8_t k;
    std::vector<float> coeffs;
};

KernelSpec read_kernel_from_file(const std::string& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open kernel file: " + path);
    }

    size_t k_value = 0;
    if (!(input >> k_value)) {
        throw std::runtime_error("Kernel file is missing the kernel length prefix: " + path);
    }

    if (k_value < 2 || k_value > 13) {
        throw std::runtime_error("Kernel length must be in the range [2, 13], got: " + std::to_string(k_value));
    }

    std::vector<float> coeffs;
    coeffs.reserve(k_value);

    float coeff = 0.0f;
    while (input >> coeff) {
        coeffs.push_back(coeff);
    }

    if (!input.eof() && input.fail()) {
        throw std::runtime_error("Failed while reading kernel coefficients from: " + path);
    }

    if (coeffs.size() != k_value) {
        throw std::runtime_error(
            "Kernel file coefficient count mismatch: expected " + std::to_string(k_value) + " but got " +
            std::to_string(coeffs.size()));
    }

    return KernelSpec{static_cast<uint8_t>(k_value), std::move(coeffs)};
}

std::vector<float> split_tile(const std::vector<float>& values, size_t row_begin, size_t rows, size_t cols) {
    std::vector<float> tile;
    tile.reserve(rows * cols);

    for (size_t row = 0; row < rows; ++row) {
        const size_t offset = (row_begin + row) * cols;
        tile.insert(tile.end(), values.begin() + offset, values.begin() + offset + cols);
    }

    return tile;
}

void print_matrix_row_major(const std::vector<float>& values, size_t rows, size_t cols) {
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            if (col > 0) {
                std::cout << ' ';
            }
            std::cout << values[row * cols + col];
        }
        std::cout << '\n';
    }
}

int main(int argc, char** argv) {
    try {
        constexpr uint32_t num_tiles_input{2};
        constexpr uint32_t num_tiles_output{1};
        constexpr tt::DataFormat cb_data_format = tt::DataFormat::Float32;
        const uint32_t size_tile_b = tt::tile_size(cb_data_format);
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

        std::vector<float> halo;
        std::vector<float> input;
        std::vector<float> coeffs;
        uint8_t k = 0;

        if (argc == 3) {
            const std::string tensor_path = argv[1];
            const std::string kernel_path = argv[2];

            const std::vector<float> tensor_values = read_floats_from_file(tensor_path);
            if (tensor_values.size() != 64 * 32) {
                throw std::runtime_error(
                    "Input tensor must contain exactly 64x32 = 2048 floats, got " +
                    std::to_string(tensor_values.size()));
            }

            const KernelSpec kernel_spec = read_kernel_from_file(kernel_path);
            k = kernel_spec.k;
            coeffs = kernel_spec.coeffs;

            halo = split_tile(tensor_values, 0, 32, 32);
            input = split_tile(tensor_values, 32, 32, 32);
        } else if (argc == 1) {
            halo.assign(32 * 32, 1.0f);
            input.assign(32 * 32, 2.0f);
            coeffs = {1.0f, 1.0f, 1.0f};
            k = static_cast<uint8_t>(coeffs.size());
        } else {
            throw std::runtime_error(
                "Usage: tt_wavelet_test <input_tensor_file> <kernel_file>\n"
                "Both files should contain whitespace-separated floats; the kernel file must begin with K.");
        }

        // User-requested debug layout: keep first-row-contiguous row-major tile payload.
        // This bypasses host tilize so flattened tile content matches the custom 32x32 printout.
        auto tiled_halo = tilize_nfaces(halo, 32, 32);
        auto tiled_input = tilize_nfaces(input, 32, 32);

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

        create_circular_buffer(program, core, halo_cb, 1, size_tile_b, cb_data_format);
        create_circular_buffer(program, core, input_cb, 1, size_tile_b, cb_data_format);
        create_circular_buffer(program, core, output_cb, 1, size_tile_b, cb_data_format);

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

        push_step_coeffs_to_compile_args(compute_compile_args, coeffs);

        std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
        unpack_to_dest_mode[kLwtSrcTile0Cb] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[kLwtSrcTile1Cb] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[kLwtBaseTileCb] = UnpackToDestMode::UnpackToDestFp32;
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
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .compile_args = compute_compile_args});

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
        std::vector<float> result_row_major = untilize_nfaces(result_of_stencil, 32, 32);
        print_matrix_row_major(result_row_major, 32, 32);

        //     std::vector<float> debug_tiles;
        // tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, debug_tiles, output_dram_buffer);

        // std::vector<float> result_of_halo(debug_tiles.begin(), debug_tiles.begin() + elems_per_tile);
        // std::vector<float> result_of_input(debug_tiles.begin() + elems_per_tile, debug_tiles.end());

        // print_array(result_of_halo);
        // print_array(result_of_input);

        // auto result_row_halo = convert_layout(
        //     tt::stl::make_const_span(result_of_halo),
        //     tile_shape,
        //     TensorLayoutType::TILED_NFACES,
        //     TensorLayoutType::LIN_ROW_MAJOR);

        // print_prefix(result_row_halo, 64, "row-major prefix:");

        // auto result_row_input = convert_layout(
        //     tt::stl::make_const_span(result_of_input),
        //     tile_shape,
        //     TensorLayoutType::TILED_NFACES,
        //     TensorLayoutType::LIN_ROW_MAJOR);

        // print_prefix(result_row_input, 64, "row-major prefix:");

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
}