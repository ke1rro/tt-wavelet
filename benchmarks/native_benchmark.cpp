// SPDX-License-Identifier: Apache-2.0
#include "ttwt/wavelet.hpp"
#include "ttwt/runtime_resources.hpp"
#include "ttwt/device/execution_observation.hpp"
#include "configuration.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/profiler.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <tt-metalium/cluster.hpp>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <unordered_map>

namespace {
using Json = nlohmann::json;
using Mesh = tt::tt_metal::distributed::MeshDevice;
using Tensor = ttnn::Tensor;
void emit(const Json &value) { std::cout << "TTWT_BENCH " << value.dump() << std::endl; }

Tensor upload(const std::vector<float> &values, uint32_t h, uint32_t w, bool one_d, Mesh *mesh) {
  const tt::tt_metal::Shape shape = one_d ? tt::tt_metal::Shape({w}) : tt::tt_metal::Shape({h, w});
  const tt::tt_metal::TensorSpec spec(
      shape, tt::tt_metal::TensorLayout(
                 tt::tt_metal::DataType::FLOAT32,
                 tt::tt_metal::PageConfig(one_d ? tt::tt_metal::Layout::ROW_MAJOR
                                                : tt::tt_metal::Layout::TILE),
                 tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
                                            tt::tt_metal::BufferType::DRAM)));
  return Tensor::from_vector(values, spec, mesh);
}

std::filesystem::path profiler_log() {
  const char *root = std::getenv("TT_METAL_PROFILER_DIR");
  return std::filesystem::path(root ? root : "generated/profiler") / ".logs/profile_log_device.csv";
}
uint64_t log_size(const std::filesystem::path &path) {
  return std::filesystem::exists(path) ? std::filesystem::file_size(path) : 0;
}
Json board_info(uint32_t chip) {
  const auto descriptor = YAML::LoadFile(tt::tt_metal::SerializeClusterDescriptor());
  for (const auto &entry : descriptor["boards"]) {
    for (const auto &member : entry[2]["chips"]) {
      if (member.as<uint32_t>() != chip) {
        continue;
      }
      auto model = entry[1]["board_type"].as<std::string>();
      std::transform(model.begin(), model.end(), model.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      return {{"device_model", model}, {"board_id", entry[0]["board_id"].as<uint64_t>()}};
    }
  }
  throw std::runtime_error("Runtime descriptor contains no board for the selected chip");
}
} // namespace

int main(int argc, char **argv) {
  std::shared_ptr<Mesh> mesh;
  try {
    std::filesystem::path config_path, manifest_path;
    bool dry_run = false, probe_device = false;
    std::string timing = "wall", dispatch = "normal_fast_dispatch";
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        std::cout << "ttwt_benchmark --config benchmark.toml --manifest benchmark_manifest.json "
                     "[--dry-run | --probe-device] --timing-mode wall|device_profile "
                     "--dispatch-mode normal_fast_dispatch|mesh_trace_replay\n"
                     "Persistent stdin protocol: CASE_ID BYTES\\n + exact FP32 payload\n";
        return 0;
      }
      if (arg == "--dry-run" || arg == "--probe-device") {
        dry_run = arg == "--dry-run";
        probe_device = arg == "--probe-device";
        continue;
      }
      if (i + 1 >= argc) {
        throw std::invalid_argument("Missing option value");
      }
      const std::string value = argv[++i];
      if (arg == "--config") {
        config_path = value;
      } else if (arg == "--manifest") {
        manifest_path = value;
      } else if (arg == "--timing-mode") {
        timing = value;
      } else if (arg == "--dispatch-mode") {
        dispatch = value;
      } else {
        throw std::invalid_argument("Unknown option " + arg);
      }
    }
    if (config_path.empty() || manifest_path.empty()) {
      throw std::invalid_argument("--config and generated --manifest are required");
    }
    const auto plan = ttwt::benchmark::load_configuration(config_path, manifest_path);
    if (dry_run) {
      std::cout << plan.dump(2) << std::endl;
      return 0;
    }
    const auto &benchmark = plan.at("config").at("benchmark");
    const auto &hardware_config = plan.at("config").at("tenstorrent");
    const int device_id = hardware_config.at("device_id").get<int>();
    const size_t trace_bytes = hardware_config.at("trace_region_bytes").get<size_t>();
    const int warmups = benchmark.at("warmups").get<int>();
    const int repetitions = benchmark.at("repetitions").get<int>();
    const std::string boundary_mode = benchmark.at("boundary_mode").get<std::string>();
    if (hardware_config.at("num_command_queues") != 1 || warmups < 1 || repetitions < 1 ||
        benchmark.at("input").at("dtype") != "float32") {
      throw std::invalid_argument("Unsupported queue count/dtype or invalid repetitions");
    }
    std::unordered_map<std::string, Json> configured_cases;
    for (const auto &case_info : plan.at("workloads")) {
      if (!configured_cases.emplace(case_info.at("case_id").get<std::string>(), case_info).second) {
        throw std::invalid_argument("Duplicate case ID in manifest");
      }
    }
    if ((timing != "wall" && timing != "device_profile") ||
        (dispatch != "normal_fast_dispatch" && dispatch != "mesh_trace_replay")) {
      throw std::invalid_argument("Invalid measurement mode");
    }
    for (const char *var :
         {"TT_METAL_SLOW_DISPATCH_MODE", "TT_METAL_WATCHER", "TT_METAL_DPRINT_CORES"}) {
      if (std::getenv(var)) {
        throw std::invalid_argument(std::string("Unset ") + var);
      }
    }
    if (timing == "wall" && std::getenv("TT_METAL_DEVICE_PROFILER")) {
      throw std::invalid_argument("Wall mode requires profiler unset");
    }
    if (timing == "device_profile" && !std::getenv("TT_METAL_DEVICE_PROFILER")) {
      throw std::invalid_argument("Device mode requires TT_METAL_DEVICE_PROFILER=1");
    }
    // Production API: one CQ, default architecture-supported dispatch-core placement.
    mesh = Mesh::create_unit_mesh(device_id, DEFAULT_L1_SMALL_SIZE, trace_bytes, 1);
    mesh->enable_program_cache();
    const auto architecture = mesh->arch();
    if (architecture != tt::ARCH::WORMHOLE_B0 && architecture != tt::ARCH::BLACKHOLE) {
      throw std::invalid_argument("Only Wormhole and Blackhole supported");
    }
    const auto chip = mesh->get_devices().front()->id();
    const auto board = board_info(chip);
    const auto grid = mesh->compute_with_storage_grid_size();
    emit({{"event", "hardware"},
          {"architecture", architecture == tt::ARCH::BLACKHOLE ? "blackhole" : "wormhole"},
          {"device_model", board["device_model"]},
          {"board_id", board["board_id"]},
          {"chip_id", chip},
          {"core_grid_x", grid.x},
          {"core_grid_y", grid.y},
          {"l1_bytes_per_core", mesh->l1_size_per_core()},
          {"clock_mhz", mesh->get_devices().front()->get_clock_rate_mhz()},
          {"num_command_queues", 1},
          {"config_sha256", plan.at("config_sha256")},
          {"matrix_sha256", plan.at("matrix_sha256")},
          {"resource_root", ttwt::detail::runtime_resources::get_root()}});
    auto &queue = mesh->mesh_command_queue();
    if (probe_device) {
      mesh->close();
      return 0;
    }
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) {
        continue;
      }
      std::optional<tt::tt_metal::distributed::MeshTraceId> trace;
      try {
        std::istringstream args(line);
        std::string case_id;
        uint64_t bytes;
        args >> case_id >> bytes;
        if (!args || !configured_cases.contains(case_id)) {
          throw std::invalid_argument("Invalid case header");
        }
        const auto &case_info = configured_cases.at(case_id);
        const auto wavelet = case_info.at("wavelet").get<std::string>();
        const auto dimension = case_info.at("dimension").get<std::string>();
        const auto direction = case_info.at("direction").get<std::string>();
        const auto size = case_info.at("logical_width").get<uint32_t>();
        const bool one_d = dimension == "1d";
        const bool forward = direction == "forward";
        const uint32_t coeff = ttwt::dwt_coeff_len(size, wavelet);
        const uint64_t band_elements = one_d ? coeff : uint64_t{coeff} * coeff;
        uint64_t elements = one_d ? size : uint64_t{size} * size;
        if (!forward) {
          elements = band_elements * (one_d ? 2 : 4);
        }
        if (bytes != elements * sizeof(float) || bytes > 512ULL * 1024 * 1024) {
          throw std::invalid_argument("Wrong/oversized input payload");
        }
        std::vector<float> payload(elements);
        std::cin.read(reinterpret_cast<char *>(payload.data()),
                      static_cast<std::streamsize>(bytes));
        if (!std::cin) {
          throw std::runtime_error("Incomplete binary input");
        }
        mesh->disable_and_clear_program_cache();
        mesh->enable_program_cache();
        ttwt::detail::execution_observation = {};
        std::vector<Tensor> inputs, outputs;
        if (forward) {
          inputs.push_back(upload(payload, size, size, one_d, mesh.get()));
        } else {
          for (int b = 0; b < (one_d ? 2 : 4); ++b) {
            const auto begin = payload.begin() + static_cast<ptrdiff_t>(b * band_elements);
            inputs.push_back(
                upload(std::vector<float>(begin, begin + static_cast<ptrdiff_t>(band_elements)),
                       coeff, coeff, one_d, mesh.get()));
          }
        }
        // All H2D, layout conversion, and initial device output allocation are outside timings.
        const auto invoke = [&] {
          if (one_d && forward) {
            const auto preallocated = outputs.empty()
                                          ? std::nullopt
                                          : std::optional(std::make_tuple(outputs[0], outputs[1]));
            auto [a, d] = ttwt::dwt(inputs[0], wavelet, boundary_mode, std::nullopt, preallocated);
            outputs = {a, d};
          } else if (one_d) {
            const auto out = outputs.empty() ? std::nullopt : std::optional(outputs[0]);
            outputs = {
                ttwt::idwt(inputs[0], inputs[1], wavelet, size, boundary_mode, std::nullopt, out)};
          } else if (forward) {
            const auto out = outputs.empty() ? std::nullopt
                                             : std::optional(std::array<Tensor, 4>{
                                                   outputs[0], outputs[1], outputs[2], outputs[3]});
            auto [ll, lh, hl, hh] =
                ttwt::dwt_2d(inputs[0], wavelet, boundary_mode, std::nullopt, out);
            outputs = {ll, lh, hl, hh};
          } else {
            const auto out = outputs.empty() ? std::nullopt : std::optional(outputs[0]);
            outputs = {ttwt::idwt_2d(inputs[0], inputs[1], inputs[2], inputs[3], wavelet,
                                     {size, size}, boundary_mode, std::nullopt, out)};
          }
        };
        for (int i = 0; i < warmups; ++i) {
          invoke();
          tt::tt_metal::distributed::Finish(queue);
        }
        const auto chosen = ttwt::detail::execution_observation;
        if (!chosen.cores || chosen.programs != 1) {
          throw std::runtime_error(
              "Expected one wavelet program; observed unsupported execution structure");
        }
        const auto entries = mesh->num_program_cache_entries();
        if (!entries) {
          throw std::runtime_error("Warmup did not populate the program cache");
        }
        if (dispatch == "mesh_trace_replay") {
          trace = mesh->begin_mesh_trace(queue);
          invoke();
          mesh->end_mesh_trace(queue, *trace);
          tt::tt_metal::distributed::Finish(queue);
          for (int i = 0; i < warmups; ++i) {
            mesh->replay_mesh_trace(queue, *trace, false);
            tt::tt_metal::distributed::Finish(queue);
          }
        }
        std::set<tt::tt_metal::experimental::ProgramExecutionUID> profiled_executions;
        if (timing == "device_profile") {
          tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh);
          for (const auto &[id, analyses] :
               tt::tt_metal::experimental::GetLatestProgramsPerfData()) {
            if (id == chip) {
              for (const auto &analysis : analyses) {
                profiled_executions.insert(analysis.program_execution_uid);
              }
            }
          }
        }
        for (int repeat = 0; repeat < repetitions; ++repeat) {
          tt::tt_metal::distributed::Finish(queue);
          const auto offset = log_size(profiler_log());
          const auto start = std::chrono::steady_clock::now();
          if (trace) {
            mesh->replay_mesh_trace(queue, *trace, false);
          } else {
            invoke();
          }
          tt::tt_metal::distributed::Finish(queue);
          const auto end = std::chrono::steady_clock::now();
          const auto physical = inputs[0].tensor_spec().physical_shape();
          Json record = {{"event", "sample"},
                         {"repeat", repeat},
                         {"status", "ok"},
                         {"tensix_cores", chosen.cores},
                         {"core_grid_x", grid.x},
                         {"core_grid_y", grid.y},
                         {"physical_height", physical.height()},
                         {"physical_width", physical.width()},
                         {"input_memory_layout",
                          one_d ? "row_major_dram_interleaved" : "tile32_dram_interleaved"},
                         {"output_memory_layout",
                          one_d ? "stick32_dram_interleaved" : "tile32_dram_interleaved"},
                         {"cache_entries", entries},
                         {"cache_reused", entries == mesh->num_program_cache_entries()}};
          if (entries != mesh->num_program_cache_entries()) {
            throw std::runtime_error(
                "Measured execution unexpectedly created a program-cache entry");
          }
          if (timing == "wall") {
            record["host_wall_ns"] = std::chrono::duration<double, std::nano>(end - start).count();
            record["profiling_method"] = "none";
          } else {
            // Device read/postprocessing is outside the wall region; no host-time substitution.
            tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh);
            const auto programs = tt::tt_metal::experimental::GetLatestProgramsPerfData();
            size_t count = 0;
            for (const auto &[id, analyses] : programs) {
              if (id != chip) {
                continue;
              }
              count += analyses.size();
              for (const auto &analysis : analyses) {
                const auto found =
                    analysis.program_analyses_results.find("DEVICE KERNEL DURATION [ns]");
                const bool fresh =
                    profiled_executions.insert(analysis.program_execution_uid).second;
                if (fresh && found != analysis.program_analyses_results.end() &&
                    found->second.duration > 0) {
                  record["device_execution_ns"] = found->second.duration;
                }
                record["device_program_runtime_id"] = analysis.program_execution_uid.runtime_id;
                record["device_trace_id"] = analysis.program_execution_uid.trace_id;
                record["device_trace_counter"] = analysis.program_execution_uid.trace_id_counter;
              }
            }
            record["device_time_method"] = "tt_metal_program_first_to_last_kernel_marker";
            record["profiling_method"] = "tt_metal_device_profiler";
            if (count != 1 || !record.contains("device_execution_ns")) {
              record.erase("device_execution_ns");
              record["status"] = "unavailable";
              record["error_message"] =
                  "Profiler did not return exactly one valid complete wavelet-program interval";
            }
            const auto path = profiler_log();
            if (std::filesystem::exists(path)) {
              record["profiler_log"] = std::filesystem::absolute(path).string();
              record["profiler_start_offset"] = offset;
              record["profiler_end_offset"] = log_size(path);
            }
          }
          emit(record);
        }
        if (trace) {
          mesh->release_mesh_trace(*trace);
          trace.reset();
        }
        emit({{"event", "done"}});
      } catch (const std::exception &error) {
        if (trace) {
          mesh->release_mesh_trace(*trace);
        }
        emit({{"event", "error"}, {"error_message", error.what()}});
        // An invalid binary header cannot be safely resynchronized; start a fresh process.
        mesh->close();
        return 2;
      }
    }
    mesh->close();
  } catch (const std::exception &error) {
    emit({{"event", "error"}, {"error_message", error.what()}});
    if (mesh) {
      mesh->close();
    }
    return 2;
  }
}
