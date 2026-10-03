#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tt-metalium/experimental/profiler.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_device.hpp"

namespace ttwv::benchmark {

inline constexpr std::string_view kDeviceKernelDuration = "DEVICE KERNEL DURATION [ns]";

struct HostTiming {
    double enqueue_or_dispatch_ms{0.0};
    double sync_wait_ms{0.0};
    double host_api_total_ms{0.0};
};

[[nodiscard]] inline bool device_profiler_enabled() noexcept {
    const char* value = std::getenv("TT_METAL_DEVICE_PROFILER");
    return value != nullptr && (std::string_view{value} == "1" || std::string_view{value} == "true" ||
                                std::string_view{value} == "True" || std::string_view{value} == "TRUE");
}

inline void configure_device_profiler_environment() {
    setenv("TT_METAL_DEVICE_PROFILER", "1", 1);
    setenv("TT_METAL_PROFILER_MID_RUN_DUMP", "1", 1);
    setenv("TT_METAL_PROFILER_CPP_POST_PROCESS", "1", 1);
    setenv("TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY", "1", 1);
    setenv("TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES", "1", 1);
}

template <typename Enqueue>
[[nodiscard]] HostTiming measure_host_timing(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue, Enqueue&& enqueue) {
    const auto start = std::chrono::steady_clock::now();
    std::forward<Enqueue>(enqueue)();
    const auto enqueued = std::chrono::steady_clock::now();
    tt::tt_metal::distributed::Finish(command_queue);
    const auto synchronized = std::chrono::steady_clock::now();
    return HostTiming{
        .enqueue_or_dispatch_ms = std::chrono::duration<double, std::milli>(enqueued - start).count(),
        .sync_wait_ms = std::chrono::duration<double, std::milli>(synchronized - enqueued).count(),
        .host_api_total_ms = std::chrono::duration<double, std::milli>(synchronized - start).count(),
    };
}

inline void discard_device_profiler_samples(tt::tt_metal::distributed::MeshDevice& mesh_device) {
    if (device_profiler_enabled()) {
        tt::tt_metal::ReadMeshDeviceProfilerResults(mesh_device);
        static_cast<void>(tt::tt_metal::experimental::GetLatestProgramsPerfData());
    }
}

[[nodiscard]] inline std::vector<double> read_device_kernel_times_ms(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const size_t expected_count) {
    if (!device_profiler_enabled()) {
        return {};
    }

    tt::tt_metal::ReadMeshDeviceProfilerResults(mesh_device);
    const auto by_device = tt::tt_metal::experimental::GetLatestProgramsPerfData();
    std::vector<std::pair<tt::tt_metal::experimental::ProgramExecutionUID, double>> ordered;
    for (const auto& [_, programs] : by_device) {
        for (const auto& program : programs) {
            const auto duration = program.program_analyses_results.find(std::string{kDeviceKernelDuration});
            if (duration != program.program_analyses_results.end()) {
                ordered.emplace_back(
                    program.program_execution_uid, static_cast<double>(duration->second.duration) / 1e6);
            }
        }
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    if (ordered.size() != expected_count) {
        throw std::runtime_error(
            "Device profiler returned " + std::to_string(ordered.size()) + " program durations for " +
            std::to_string(expected_count) + " measured executions");
    }

    std::vector<double> durations;
    durations.reserve(ordered.size());
    for (const auto& [_, duration] : ordered) {
        durations.push_back(duration);
    }
    return durations;
}

}  // namespace ttwv::benchmark
