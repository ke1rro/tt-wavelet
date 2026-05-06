#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"

namespace {

constexpr uint32_t kTileWidth = 32;
constexpr uint32_t kTileHeight = 32;
constexpr uint32_t kTileScalars = kTileWidth * kTileHeight;
constexpr uint32_t kTilesPerCase = 3;
constexpr uint32_t kCaseCount = 1;
constexpr uint32_t kOpShift = 0;
constexpr uint32_t kOpRecover1 = 1;
constexpr uint32_t kOpRecover2 = 2;
constexpr tt::DataFormat kDataFormat = tt::DataFormat::Float32;

[[nodiscard]] constexpr uint32_t tile_offset(const uint32_t row, const uint32_t col) {
    const uint32_t face_row = row / 16;
    const uint32_t face_col = col / 16;
    const uint32_t face = face_row * 2U + face_col;
    return face * 16U * 16U + (row % 16U) * 16U + (col % 16U);
}

[[nodiscard]] float splice_value(const std::vector<float>& data, const uint32_t row, const uint32_t col) {
    const uint32_t tile = col < 32 ? 0 : 1;
    const uint32_t tile_col = col < 32 ? col : col - 32;
    return data[tile * kTileScalars + tile_offset(row, tile_col)];
}

void set_splice_value(std::vector<float>& data, const uint32_t row, const uint32_t col, const float value) {
    const uint32_t tile = col < 32 ? 0 : 1;
    const uint32_t tile_col = col < 32 ? col : col - 32;
    data[tile * kTileScalars + tile_offset(row, tile_col)] = value;
}

[[nodiscard]] float tmp_value(const std::vector<float>& data, const uint32_t row, const uint32_t col) {
    return data[2 * kTileScalars + tile_offset(row, col)];
}

void set_tmp_value(std::vector<float>& data, const uint32_t row, const uint32_t col, const float value) {
    data[2 * kTileScalars + tile_offset(row, col)] = value;
}

[[nodiscard]] std::vector<float> make_input() {
    std::vector<float> data(kTilesPerCase * kTileScalars, 0.0F);

    for (uint32_t row = 0; row < 32; ++row) {
        for (uint32_t col = 0; col < 64; ++col) {
            set_splice_value(data, row, col, static_cast<float>(1000 + row * 64 + col));
        }
    }

    for (uint32_t row = 0; row < 32; ++row) {
        for (uint32_t col = 0; col < 32; ++col) {
            set_tmp_value(data, row, col, static_cast<float>(100000 + row * 32 + col));
        }
    }

    return data;
}

[[nodiscard]] std::vector<float> expected_shift(const std::vector<float>& input, const uint32_t k) {
    std::vector<float> expected = input;
    for (uint32_t row = 0; row < 32; ++row) {
        for (uint32_t col = 63; col >= k && col < 64; --col) {
            set_splice_value(expected, row, col, splice_value(input, row, col - k));
        }
    }
    return expected;
}

[[nodiscard]] std::vector<float> expected_recover1(const std::vector<float>& input) {
    std::vector<float> expected = input;
    for (uint32_t col = 0; col < 16; ++col) {
        set_splice_value(expected, 0, col, tmp_value(input, 0, col));
        for (uint32_t row = 1; row < 32; ++row) {
            set_splice_value(expected, row, col, splice_value(input, row - 1, 48 + col));
        }
        set_tmp_value(expected, 0, col, splice_value(input, 31, 48 + col));
    }
    return expected;
}

[[nodiscard]] std::vector<float> expected_recover2(const std::vector<float>& input) {
    std::vector<float> expected = input;
    for (uint32_t col = 0; col < 16; ++col) {
        for (uint32_t row = 0; row < 31; ++row) {
            set_splice_value(expected, row, 48 + col, splice_value(input, row + 1, col));
        }
        set_splice_value(expected, 31, 48 + col, tmp_value(input, 3, col));
        set_tmp_value(expected, 3, col, splice_value(input, 0, col));
    }
    return expected;
}

[[nodiscard]] bool compare_case(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    const std::string_view name,
    const uint32_t garbage_columns) {
    bool pass = true;
    uint32_t mismatch_count = 0;
    auto check_value = [&](const uint32_t tile, const uint32_t row, const uint32_t col) {
        const size_t index = tile * kTileScalars + tile_offset(row, col);
        if (actual[index] != expected[index]) {
            if (mismatch_count < 20) {
                std::cerr << name << " mismatch tile=" << tile << " row=" << row << " col=" << col
                          << " expected=" << expected[index] << " actual=" << actual[index] << '\n';
            }
            ++mismatch_count;
            pass = false;
        }
    };

    for (uint32_t row = 0; row < 32; ++row) {
        for (uint32_t col = garbage_columns; col < 64; ++col) {
            check_value(col < 32 ? 0 : 1, row, col < 32 ? col : col - 32);
        }
    }

    for (uint32_t row = 0; row < 32; ++row) {
        for (uint32_t col = 0; col < 32; ++col) {
            check_value(2, row, col);
        }
    }

    if (mismatch_count > 20) {
        std::cerr << name << " had " << mismatch_count << " total mismatches\n";
    }

    return pass;
}

void create_circular_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreCoord& core,
    const uint32_t cb_index,
    const uint32_t entry_count,
    const uint32_t page_size) {
    const auto config = tt::tt_metal::CircularBufferConfig(entry_count * page_size, {{cb_index, kDataFormat}})
                            .set_page_size(cb_index, page_size);
    static_cast<void>(tt::tt_metal::CreateCircularBuffer(program, core, config));
}

void run_program(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    tt::tt_metal::Program&& program) {
    tt::tt_metal::distributed::MeshWorkload workload;
    const auto device_range = tt::tt_metal::distributed::MeshCoordinateRange(mesh_device.shape());
    workload.add_program(device_range, std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
    tt::tt_metal::distributed::Finish(command_queue);
}

[[nodiscard]] bool run_case(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const std::filesystem::path& kernel_root,
    const std::string_view name,
    const uint32_t op,
    const uint32_t shift_k) {
    constexpr tt::tt_metal::CoreCoord core{0, 0};
    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_output = tt::CBIndex::c_16;

    const uint32_t tile_nbytes = tt::tile_size(kDataFormat);
    const uint32_t case_nbytes = kTilesPerCase * tile_nbytes;
    const auto input = make_input();
    std::vector<float> expected;
    if (op == kOpShift) {
        expected = expected_shift(input, shift_k);
    } else if (op == kOpRecover1) {
        expected = expected_recover1(input);
    } else {
        expected = expected_recover2(input);
    }

    const tt::tt_metal::distributed::DeviceLocalBufferConfig local_config{
        .page_size = tile_nbytes,
        .buffer_type = tt::tt_metal::BufferType::DRAM,
    };
    const tt::tt_metal::distributed::ReplicatedBufferConfig replicated_config{
        .size = static_cast<uint64_t>(case_nbytes * kCaseCount),
    };
    auto input_buffer = tt::tt_metal::distributed::MeshBuffer::create(replicated_config, local_config, &mesh_device);
    auto output_buffer = tt::tt_metal::distributed::MeshBuffer::create(replicated_config, local_config, &mesh_device);

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    create_circular_buffer(program, core, cb_input, kTilesPerCase, tile_nbytes);
    create_circular_buffer(program, core, cb_output, kTilesPerCase, tile_nbytes);

    std::vector<uint32_t> reader_compile_args = {cb_input};
    tt::tt_metal::TensorAccessorArgs(*input_buffer->get_backing_buffer()).append_to(reader_compile_args);
    std::vector<uint32_t> writer_compile_args = {cb_output};
    tt::tt_metal::TensorAccessorArgs(*output_buffer->get_backing_buffer()).append_to(writer_compile_args);
    const std::vector<uint32_t> compute_compile_args = {cb_input, cb_output, op, shift_k};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[cb_input] = UnpackToDestMode::UnpackToDestFp32;

    const auto reader = tt::tt_metal::CreateKernel(
        program,
        kernel_root / "kernels/test/splice_ops_reader.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));
    const auto writer = tt::tt_metal::CreateKernel(
        program,
        kernel_root / "kernels/test/splice_ops_writer.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));
    const auto compute = tt::tt_metal::CreateKernel(
        program,
        kernel_root / "kernels/test/splice_ops_compute.cpp",
        core,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = compute_compile_args,
        });

    tt::tt_metal::SetRuntimeArgs(
        program, reader, core, std::array<uint32_t, 2>{static_cast<uint32_t>(input_buffer->address()), kCaseCount});
    tt::tt_metal::SetRuntimeArgs(program, compute, core, std::array<uint32_t, 1>{kCaseCount});
    tt::tt_metal::SetRuntimeArgs(
        program, writer, core, std::array<uint32_t, 2>{static_cast<uint32_t>(output_buffer->address()), kCaseCount});

    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, input_buffer, input, false);
    run_program(mesh_device, command_queue, std::move(program));

    std::vector<float> actual;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, actual, output_buffer, true);
    return compare_case(actual, expected, name, op == kOpShift ? shift_k : 0);
}

}  // namespace

int main() {
    try {
        setenv("TT_LOGGER_LEVEL", "error", 0);

        auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
        auto& command_queue = mesh_device->mesh_command_queue();
        const std::filesystem::path kernel_root = TT_WAVELET_SOURCE_DIR;

        bool pass = true;
        for (const uint32_t k : {0U, 1U, 2U, 7U, 15U}) {
            pass &= run_case(*mesh_device, command_queue, kernel_root, "SHIFT<" + std::to_string(k) + ">", kOpShift, k);
        }
        pass &= run_case(*mesh_device, command_queue, kernel_root, "RECOVER1", kOpRecover1, 0);
        pass &= run_case(*mesh_device, command_queue, kernel_root, "RECOVER2", kOpRecover2, 0);

        if (!mesh_device->close()) {
            pass = false;
        }

        if (!pass) {
            throw std::runtime_error("splice operation regression failed");
        }

        std::cout << "splice operation regression passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return EXIT_FAILURE;
    }
}
