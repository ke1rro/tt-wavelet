// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/tilize_utils.hpp"
#include "tt_wavelet/include/common/fp32_arithmetic.hpp"

namespace {

constexpr uint32_t kRows = 32;
constexpr uint32_t kOutputColumns = 32;
constexpr uint32_t kPackedColumns = 64;
constexpr tt::tt_metal::CoreCoord kCore{0, 0};
constexpr tt::DataFormat kDataFormat = tt::DataFormat::Float32;
constexpr const char* kReaderPath = "tt-wavelet/tests/kernels/horizontal_dense_stencil_reader.cpp";
constexpr const char* kWriterPath = "tt-wavelet/tests/kernels/vertical_stencil_writer.cpp";
constexpr const char* kComputePath = "tt-wavelet/tests/kernels/horizontal_dense_stencil_compute.cpp";

[[nodiscard]] uint64_t ordered_float_bits(const float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t magnitude = bits & 0x7FFFFFFFU;
    return (bits & 0x80000000U) != 0 ? 0x80000000ULL - magnitude : 0x80000000ULL + magnitude;
}

[[nodiscard]] uint64_t ulp_distance(const float lhs, const float rhs) {
    const uint64_t lhs_ordered = ordered_float_bits(lhs);
    const uint64_t rhs_ordered = ordered_float_bits(rhs);
    return lhs_ordered > rhs_ordered ? lhs_ordered - rhs_ordered : rhs_ordered - lhs_ordered;
}

void create_circular_buffer(tt::tt_metal::Program& program, const tt::CBIndex cb, const uint32_t tile_bytes) {
    const auto config =
        tt::tt_metal::CircularBufferConfig(tile_bytes, {{cb, kDataFormat}}).set_page_size(cb, tile_bytes);
    static_cast<void>(tt::tt_metal::CreateCircularBuffer(program, kCore, config));
}

[[nodiscard]] std::vector<float> make_coefficients(const uint32_t k) {
    if (k == 2) {
        // db11's first two-tap production route.
        return {
            std::bit_cast<float>(0xBE02ACD2U),
            std::bit_cast<float>(0xBBDC3C52U),
        };
    }
    std::vector<float> coefficients(k);
    for (uint32_t index = 0; index < k; ++index) {
        const float magnitude = static_cast<float>(index + 1) / static_cast<float>(5 * k);
        coefficients[index] = (index & 1U) == 0 ? magnitude : -0.75F * magnitude;
    }
    return coefficients;
}

[[nodiscard]] std::vector<float> make_source(const uint32_t k) {
    const uint32_t source_columns = kOutputColumns + k - 1;
    std::vector<float> source(kRows * source_columns);
    uint32_t state = 0x12345678U;
    for (uint32_t row = 0; row < kRows; ++row) {
        for (uint32_t column = 0; column < source_columns; ++column) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            const uint32_t bits = 0x3F000000U | (state & 0x007FFFFFU);
            const float magnitude = std::bit_cast<float>(bits);
            source[row * source_columns + column] = (state & 0x80000000U) != 0 ? -magnitude : magnitude;
        }
    }
    return source;
}

[[nodiscard]] std::vector<float> make_base() {
    std::vector<float> base(kRows * kOutputColumns);
    uint32_t state = 0x9E3779B9U;
    for (uint32_t row = 0; row < kRows; ++row) {
        for (uint32_t column = 0; column < kOutputColumns; ++column) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            const uint32_t bits = 0x3E800000U | (state & 0x007FFFFFU);
            const float magnitude = std::bit_cast<float>(bits);
            base[row * kOutputColumns + column] = (state & 0x40000000U) != 0 ? -magnitude : magnitude;
        }
    }
    return base;
}

[[nodiscard]] std::vector<float> pack_source(const std::vector<float>& source, const uint32_t k) {
    const uint32_t source_columns = kOutputColumns + k - 1;
    const uint32_t alignment = 17 - k;
    std::vector<float> packed(kRows * kPackedColumns, 0.0F);
    for (uint32_t row = 0; row < kRows; ++row) {
        std::copy_n(
            source.begin() + static_cast<std::ptrdiff_t>(row * source_columns),
            source_columns,
            packed.begin() + static_cast<std::ptrdiff_t>(row * kPackedColumns + alignment));
    }
    return packed;
}

[[nodiscard]] std::vector<float> tilize_source_and_base(
    const std::vector<float>& packed_source, const std::vector<float>& base) {
    std::vector<float> input;
    input.reserve(3 * kRows * kOutputColumns);
    for (uint32_t tile_column = 0; tile_column < 2; ++tile_column) {
        std::vector<float> row_major(kRows * kOutputColumns);
        for (uint32_t row = 0; row < kRows; ++row) {
            std::copy_n(
                packed_source.begin() +
                    static_cast<std::ptrdiff_t>(row * kPackedColumns + tile_column * kOutputColumns),
                kOutputColumns,
                row_major.begin() + static_cast<std::ptrdiff_t>(row * kOutputColumns));
        }
        std::vector<float> tile = tilize_nfaces(row_major, 32, 32);
        input.insert(input.end(), tile.begin(), tile.end());
    }
    std::vector<float> base_tile = tilize_nfaces(base, 32, 32);
    input.insert(input.end(), base_tile.begin(), base_tile.end());
    return input;
}

[[nodiscard]] std::vector<float> reference(
    const std::vector<float>& source,
    const std::vector<float>& base,
    const std::vector<float>& coefficients,
    const ttwv::Fp32Arithmetic arithmetic) {
    const uint32_t source_columns = kOutputColumns + static_cast<uint32_t>(coefficients.size()) - 1;
    std::vector<float> output(kRows * kOutputColumns);
    for (uint32_t row = 0; row < kRows; ++row) {
        for (uint32_t column = 0; column < kOutputColumns; ++column) {
            float accumulator = base[row * kOutputColumns + column];
            for (uint32_t coefficient = 0; coefficient < coefficients.size(); ++coefficient) {
                accumulator = ttwv::fp32_fma(
                    coefficients[coefficient],
                    source[row * source_columns + column + coefficients.size() - 1 - coefficient],
                    accumulator,
                    arithmetic);
            }
            output[row * kOutputColumns + column] = accumulator;
        }
    }
    return output;
}

void run_case(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const uint32_t k) {
    const uint32_t tile_bytes = tt::tile_size(kDataFormat);
    const std::vector<float> coefficients = make_coefficients(k);
    const std::vector<float> source = make_source(k);
    const std::vector<float> base = make_base();
    const ttwv::Fp32Arithmetic arithmetic = mesh_device.arch() == tt::ARCH::BLACKHOLE
                                                ? ttwv::Fp32Arithmetic::kBlackholeSfpu
                                                : ttwv::Fp32Arithmetic::kWormholeSfpu;
    const std::vector<float> expected = reference(source, base, coefficients, arithmetic);
    const std::vector<float> input = tilize_source_and_base(pack_source(source, k), base);

    const tt::tt_metal::distributed::DeviceLocalBufferConfig local_config{
        .page_size = tile_bytes,
        .buffer_type = tt::tt_metal::BufferType::DRAM,
    };
    auto input_buffer = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{
            .size = static_cast<uint64_t>(3 * tile_bytes),
        },
        local_config,
        &mesh_device);
    auto output_buffer = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{
            .size = static_cast<uint64_t>(tile_bytes),
        },
        local_config,
        &mesh_device);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, input_buffer, input, false);

    constexpr tt::CBIndex cb_source0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_source1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_base = tt::CBIndex::c_2;
    constexpr tt::CBIndex cb_output = tt::CBIndex::c_16;
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    create_circular_buffer(program, cb_source0, tile_bytes);
    create_circular_buffer(program, cb_source1, tile_bytes);
    create_circular_buffer(program, cb_base, tile_bytes);
    create_circular_buffer(program, cb_output, tile_bytes);

    std::vector<uint32_t> reader_compile_args = {
        static_cast<uint32_t>(cb_source0),
        static_cast<uint32_t>(cb_source1),
        static_cast<uint32_t>(cb_base),
    };
    tt::tt_metal::TensorAccessorArgs(*input_buffer->get_backing_buffer()).append_to(reader_compile_args);
    std::vector<uint32_t> writer_compile_args = {
        static_cast<uint32_t>(cb_output),
    };
    tt::tt_metal::TensorAccessorArgs(*output_buffer->get_backing_buffer()).append_to(writer_compile_args);
    std::vector<uint32_t> compute_compile_args = {
        k,
        static_cast<uint32_t>(cb_source0),
        static_cast<uint32_t>(cb_source1),
        static_cast<uint32_t>(cb_base),
        static_cast<uint32_t>(cb_output),
    };
    std::vector<uint32_t> compute_runtime_args;
    compute_runtime_args.reserve(coefficients.size());
    for (const float coefficient : coefficients) {
        compute_runtime_args.push_back(std::bit_cast<uint32_t>(coefficient));
    }

    const auto reader = tt::tt_metal::CreateKernel(
        program, kReaderPath, kCore, tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));
    const auto writer = tt::tt_metal::CreateKernel(
        program, kWriterPath, kCore, tt::tt_metal::WriterDataMovementConfig(writer_compile_args));
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[cb_source0] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[cb_source1] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[cb_base] = UnpackToDestMode::UnpackToDestFp32;
    const auto compute = tt::tt_metal::CreateKernel(
        program,
        kComputePath,
        kCore,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = compute_compile_args,
        });
    tt::tt_metal::SetRuntimeArgs(program, compute, kCore, compute_runtime_args);

    tt::tt_metal::SetRuntimeArgs(
        program, reader, kCore, {static_cast<uint32_t>(input_buffer->get_backing_buffer()->address())});
    tt::tt_metal::SetRuntimeArgs(
        program, writer, kCore, {static_cast<uint32_t>(output_buffer->get_backing_buffer()->address())});
    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
    tt::tt_metal::distributed::Finish(command_queue);

    std::vector<float> tilized_output;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, tilized_output, output_buffer, true);
    const std::vector<float> actual = untilize_nfaces(tilized_output, 32, 32);
    float max_absolute_error = 0.0F;
    uint64_t max_ulp_distance = 0;
    size_t max_error_index = 0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const float error = std::abs(actual[index] - expected[index]);
        max_ulp_distance = std::max(max_ulp_distance, ulp_distance(actual[index], expected[index]));
        if (error > max_absolute_error) {
            max_absolute_error = error;
            max_error_index = index;
        }
    }
    if (max_absolute_error > 1.0e-4F) {
        throw std::runtime_error(
            "K=" + std::to_string(k) + " dense horizontal stencil max error " + std::to_string(max_absolute_error) +
            " at row " + std::to_string(max_error_index / kOutputColumns) + ", column " +
            std::to_string(max_error_index % kOutputColumns));
    }
    std::cout << "PASS dense horizontal K=" << k << " max_abs=" << max_absolute_error << " max_ulp=" << max_ulp_distance
              << " target_1e-5=" << (max_absolute_error <= 1.0e-5F ? "yes" : "no") << '\n';
}

}  // namespace

int main() {
    try {
        auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
        auto& command_queue = mesh_device->mesh_command_queue();
        for (const uint32_t k : std::array<uint32_t, 4>{1, 2, 9, 17}) {
            run_case(*mesh_device, command_queue, k);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
