#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_workload.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"

using json = nlohmann::json;

// ---------------------------------------------------------------------------------------------------------

struct LiftingStep {
    std::string type;
    int shift;
    std::vector<float> coefficients;
};

struct LiftingScheme {
    int tap_size;
    int delay_even;
    int delay_odd;
    std::vector<LiftingStep> steps;
};

std::vector<float> read_signal(const std::string& filepath) {
    std::vector<float> signal;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open signal file: " + filepath);
    }
    float val;
    while (file >> val) {
        signal.push_back(val);
    }
    return signal;
}

// --------------------------------------------------------------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_signal.txt> <path_to_scheme.json>" << std::endl;
        return 1;
    }

    std::string signal_path = argv[1];
    std::string json_path = argv[2];

    std::vector<float> even_vec;
    std::vector<float> odd_vec;

    std::vector<float> raw_signal;
    // Генеруємо тестовий сигнал [0, 1, 2, ... 4095]
    for (int i = 0; i < 4096; ++i) {
        raw_signal.push_back(static_cast<float>(i));
    }

    for (size_t i = 0; i < raw_signal.size(); ++i) {
        if (i % 2 == 0) {
            even_vec.push_back(raw_signal[i]);
        } else {
            odd_vec.push_back(raw_signal[i]);
        }
    }

    static constexpr uint32_t elements_per_tile{tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT};

    uint32_t needed_elements = ((even_vec.size() + elements_per_tile - 1) / elements_per_tile) * elements_per_tile;

    // zero padding for now
    even_vec.resize(needed_elements, 0.0f);
    odd_vec.resize(needed_elements, 0.0f);

    // parse json
    // ----------------------------------------------------------------------------------------------------------
    std::ifstream f(json_path);
    if (!f.is_open()) {
        std::cerr << "Could not open JSON file: " << json_path << std::endl;
        return 1;
    }
    json data = json::parse(f);

    LiftingScheme scheme;
    scheme.tap_size = data["tap_size"];
    scheme.delay_even = data["delay"]["even"];
    scheme.delay_odd = data["delay"]["odd"];

    for (const auto& step_json : data["steps"]) {
        LiftingStep step;
        step.type = step_json["type"];
        step.shift = step_json["shift"];
        for (float coef : step_json["coefficients"]) {
            step.coefficients.push_back(coef);
        }
        scheme.steps.push_back(step);
    }

    // init core
    // -------------------------------------------------------------------------------------------------------------
    constexpr int device_id{0};
    const auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    tt::tt_metal::distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // tile config
    // -----------------------------------------------------------------------------------------------------------
    uint32_t num_tiles = odd_vec.size() / elements_per_tile;
    static constexpr uint32_t tile_size_bytes{sizeof(float) * elements_per_tile};
    uint32_t dram_buffer_size{tile_size_bytes * num_tiles};

    // buff init
    // --------------------------------------------------------------------------------------------------------------
    const tt::tt_metal::distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_size_bytes,
        .buffer_type = tt::tt_metal::BufferType::DRAM,
    };

    tt::tt_metal::distributed::ReplicatedBufferConfig dram_buffer_config_in{.size = dram_buffer_size};

    // Збільшуємо вихідні буфери, щоб туди влізли всі "сирі" тапи для тестового дампу
    uint32_t dram_debug_buffer_size = tile_size_bytes * 100;  // на 100 тайлів
    tt::tt_metal::distributed::ReplicatedBufferConfig dram_buffer_config_out{.size = dram_debug_buffer_size};

    const tt::tt_metal::distributed::DeviceLocalBufferConfig l1_config{
        .page_size = tile_size_bytes,
        .buffer_type = tt::tt_metal::BufferType::L1,
    };

    tt::tt_metal::distributed::ReplicatedBufferConfig l1_buffer_config{.size = tile_size_bytes};

    auto dram_in_odd{
        tt::tt_metal::distributed::MeshBuffer::create(dram_buffer_config_in, dram_config, mesh_device.get())};
    auto dram_in_even{
        tt::tt_metal::distributed::MeshBuffer::create(dram_buffer_config_in, dram_config, mesh_device.get())};
    auto dram_out_odd{
        tt::tt_metal::distributed::MeshBuffer::create(dram_buffer_config_out, dram_config, mesh_device.get())};
    auto dram_out_even{
        tt::tt_metal::distributed::MeshBuffer::create(dram_buffer_config_out, dram_config, mesh_device.get())};

    // cb init
    // -------------------------------------------------------------------------------------------------------------

    uint32_t cb_size_tiles = 16;
    uint32_t cb_size_bytes = cb_size_tiles * tile_size_bytes;

    auto core = tt::tt_metal::CoreCoord(0, 0);

    constexpr uint32_t cb_id_in_odd = tt::CBIndex::c_0;
    auto cb_in_odd_config = tt::tt_metal::CircularBufferConfig(cb_size_bytes, {{cb_id_in_odd, tt::DataFormat::Float32}})
                                .set_page_size(cb_id_in_odd, tile_size_bytes);
    auto cb_in_odd = tt::tt_metal::CreateCircularBuffer(program, core, cb_in_odd_config);

    constexpr uint32_t cb_id_in_even = tt::CBIndex::c_1;
    auto cb_in_even_config =
        tt::tt_metal::CircularBufferConfig(cb_size_bytes, {{cb_id_in_even, tt::DataFormat::Float32}})
            .set_page_size(cb_id_in_even, tile_size_bytes);
    auto cb_in_even = tt::tt_metal::CreateCircularBuffer(program, core, cb_in_even_config);

    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;
    auto cb_out_config = tt::tt_metal::CircularBufferConfig(cb_size_bytes, {{cb_id_out, tt::DataFormat::Float32}})
                             .set_page_size(cb_id_out, tile_size_bytes);
    auto cb_out = tt::tt_metal::CreateCircularBuffer(program, core, cb_out_config);

    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, dram_in_odd, odd_vec, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, dram_in_even, even_vec, true);

    // kernel init
    // ---------------------------------------------------------------------------------------------------------
    bool is_fp32_dest_acc_en = true;

    std::vector<uint32_t> dram_copy_compile_time_arg;

    tt::tt_metal::TensorAccessorArgs(*dram_in_odd->get_backing_buffer()).append_to(dram_copy_compile_time_arg);
    tt::tt_metal::TensorAccessorArgs(*dram_in_even->get_backing_buffer()).append_to(dram_copy_compile_time_arg);

    tt::tt_metal::TensorAccessorArgs(*dram_out_odd->get_backing_buffer()).append_to(dram_copy_compile_time_arg);
    tt::tt_metal::TensorAccessorArgs(*dram_out_even->get_backing_buffer()).append_to(dram_copy_compile_time_arg);

    auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "tt-wavelet/kernels/reader.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = dram_copy_compile_time_arg});

    auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "tt-wavelet/kernels/writer.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = dram_copy_compile_time_arg});

    /*
    std::vector<uint32_t> compute_kernel_args = {
        num_tiles,
        cb_id_in_odd,
        cb_id_in_even,
        cb_id_out
    };

    auto compute_kernel = tt::tt_metal::CreateKernel(
        program,
        "kernels/compute/lifting_sfpu.cpp",
        core,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = tt::MathFidelity::HiFi4,
            .fp32_dest_acc_en = is_fp32_dest_acc_en,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args
        }
    );
    */

    // RUNTIME ARGS FOR READER -----------------------------------------------------
    std::vector<uint32_t> reader_rt_args;
    reader_rt_args.push_back(static_cast<uint32_t>(dram_in_odd->address()));
    reader_rt_args.push_back(static_cast<uint32_t>(dram_in_even->address()));
    reader_rt_args.push_back(num_tiles);

    constexpr uint32_t POLICY_LEN = 5;
    reader_rt_args.push_back(POLICY_LEN);

    reader_rt_args.push_back(static_cast<uint32_t>(scheme.steps.size()));

    for (const auto& step : scheme.steps) {
        uint32_t op_type = 0;
        if (step.type == "predict") {
            op_type = 0;
        } else if (step.type == "update") {
            op_type = 1;
        } else if (step.type == "scale-even") {
            op_type = 2;
        } else if (step.type == "scale-odd") {
            op_type = 3;
        }

        reader_rt_args.push_back(op_type);
        reader_rt_args.push_back(static_cast<uint32_t>(step.coefficients.size()));
        reader_rt_args.push_back(static_cast<uint32_t>(step.shift));
    }

    tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core, reader_rt_args);
    // --------------------------------------------------------------------------------------------

    // RUNTIME ARGS FOR WRITER (TEST DUMP SINK) ------------------------------------
    std::vector<uint32_t> writer_rt_args;
    writer_rt_args.push_back(static_cast<uint32_t>(dram_out_odd->address()));   // Пишемо базу сюди
    writer_rt_args.push_back(static_cast<uint32_t>(dram_out_even->address()));  // Пишемо тапи сюди
    writer_rt_args.push_back(num_tiles);
    writer_rt_args.push_back(POLICY_LEN);
    writer_rt_args.push_back(static_cast<uint32_t>(scheme.steps.size()));

    for (const auto& step : scheme.steps) {
        uint32_t op_type = 0;
        if (step.type == "predict") {
            op_type = 0;
        } else if (step.type == "update") {
            op_type = 1;
        } else if (step.type == "scale-even") {
            op_type = 2;
        } else if (step.type == "scale-odd") {
            op_type = 3;
        }

        writer_rt_args.push_back(op_type);
        writer_rt_args.push_back(static_cast<uint32_t>(step.coefficients.size()));
        writer_rt_args.push_back(static_cast<uint32_t>(step.shift));
    }

    tt::tt_metal::SetRuntimeArgs(program, writer_kernel, core, writer_rt_args);

    /*
    // runtime args for compute
    std::vector<uint32_t> math_rt_args;

    math_rt_args.push_back(scheme.steps.size());

    for (const auto& step : scheme.steps) {
        uint32_t op_type = 0;
        if (step.type == "predict") op_type = 0;
        else if (step.type == "update") op_type = 1;
        else if (step.type == "scale-even") op_type = 2;
        else if (step.type == "scale-odd") op_type = 3;

        math_rt_args.push_back(op_type);

        math_rt_args.push_back(static_cast<uint32_t>(step.shift));

        math_rt_args.push_back(step.coefficients.size());

        for (float coef : step.coefficients) {
            math_rt_args.push_back(*reinterpret_cast<const uint32_t*>(&coef));
        }
    }

    tt::tt_metal::SetRuntimeArgs(
        program,
        compute_kernel,
        core,
        math_rt_args
    );
    */

    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange device_range =
        tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());

    workload.add_program(device_range, std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);

    tt::tt_metal::distributed::Finish(cq);

    std::vector<float> odd_out_vec(100 * 1024 / sizeof(float), 0.0f);   // Буфер під дамп БАЗИ
    std::vector<float> even_out_vec(100 * 1024 / sizeof(float), 0.0f);  // Буфер під дамп ТАПІВ

    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, odd_out_vec, dram_out_odd, true);
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, even_out_vec, dram_out_even, true);
    tt::tt_metal::distributed::Finish(cq);

    std::cout << "Read from device complete!" << std::endl;

    std::cout << "\n===== DUMP BASE (TILE 0, Step 0) =====" << std::endl;
    std::cout << "Start: ";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << odd_out_vec[i] << " ";
    }
    std::cout << " ... ";
    for (size_t i = 1019; i < 1024; ++i) {
        std::cout << odd_out_vec[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "\n===== DUMP TAPS (TILE 0, Step 0) =====" << std::endl;
    // Знаючи, що у bior3.9 9 тапів, їх буде записано в even_out_vec послідовно (5 тайлів першого пасу в початок і тд)
    for (int t = 0; t < 5; ++t) {
        std::cout << "Tap " << t << " Start: ";
        for (size_t i = t * 1024; i < t * 1024 + 5; ++i) {
            std::cout << even_out_vec[i] << " ";
        }
        std::cout << " ... " << std::endl;
    }

    return 0;
}