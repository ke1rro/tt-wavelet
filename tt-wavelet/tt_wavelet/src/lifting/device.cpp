#include "tt_wavelet/include/lifting/device.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tt_stl/assert.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tt-metalium/buffer.hpp"
#include "tt-metalium/circular_buffer_constants.h"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt_wavelet/include/common/padding.hpp"
#include "tt_wavelet/include/common/signal.hpp"
#include "tt_wavelet/include/pad_split/device.hpp"

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#endif

namespace ttwv {

namespace {

constexpr uint32_t kStickWidth = 32;
constexpr uint32_t kNocAlignmentBytes = 32;
constexpr uint32_t kSpliceAdvanceElements = 32 * 48;
constexpr uint32_t kSpliceTailElements = 16;
constexpr uint32_t kTilePagesPerSplice = 2;
constexpr uint32_t kL1ReserveBytes = 128 * 1024;
constexpr tt::DataFormat kDataFormat = tt::DataFormat::Float32;

constexpr uint32_t kCbEven = tt::CBIndex::c_0;
constexpr uint32_t kCbOdd = tt::CBIndex::c_1;
constexpr uint32_t kCbEvenCache = tt::CBIndex::c_2;
constexpr uint32_t kCbOddCache = tt::CBIndex::c_3;
constexpr uint32_t kCbDone = tt::CBIndex::c_16;
constexpr uint32_t kCbWriterScratch = tt::CBIndex::c_17;
constexpr uint32_t kCbEvenWork = tt::CBIndex::c_24;
constexpr uint32_t kCbOddWork = tt::CBIndex::c_25;
constexpr uint32_t kCbFinalEven = tt::CBIndex::c_26;
constexpr uint32_t kCbFinalOdd = tt::CBIndex::c_27;
constexpr uint32_t kCbPrediction = tt::CBIndex::c_28;
constexpr uint32_t kCbBaseRecoverState = tt::CBIndex::c_29;
constexpr uint32_t kCbPredictionRecoverState = tt::CBIndex::c_30;

constexpr const char* kLwtReaderKernel = "kernels/dataflow/lwt_reader.cpp";
constexpr const char* kLwtComputeKernel = "kernels/compute/lwt_compute.cpp";
constexpr const char* kLwtWriterKernel = "kernels/dataflow/lwt_writer.cpp";

struct StreamState {
    int32_t shift{0};
    size_t length{0};
};

struct TransformGeometry {
    PadSplit1DLayout pad_split{};
    StreamState final_even{};
    StreamState final_odd{};
    uint32_t even_crop{0};
    uint32_t odd_crop{0};
    uint32_t output_length{0};
    uint32_t splice_count{0};
    bool swapped{false};
};

struct LogicalInterval {
    int64_t begin{0};
    int64_t end{-1};

    [[nodiscard]] bool empty() const noexcept { return end < begin; }
};

struct CorePiece {
    uint32_t source_start{0};
    uint32_t source_length{0};
    uint32_t source_prefix{0};
    uint32_t splice_count{0};
    uint32_t approximation_offset{0};
    uint32_t detail_offset{0};
    uint32_t output_stick_start{0};
    uint32_t output_length{0};
};

[[nodiscard]] uint32_t checked_u32(const size_t value, const char* label) {
    TT_FATAL(
        value <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()), "{} {} exceeds uint32_t", label, value);
    return static_cast<uint32_t>(value);
}

[[nodiscard]] std::filesystem::path kernel_path(const std::filesystem::path& root, const char* relative) {
    return root / relative;
}

[[nodiscard]] std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> make_signal_buffer(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const SignalBuffer& signal) {
    const uint32_t page_size = signal.aligned_stick_bytes(kNocAlignmentBytes);
    const tt::tt_metal::distributed::DeviceLocalBufferConfig local{
        .page_size = page_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM,
    };
    const uint64_t alloc_size =
        static_cast<uint64_t>(std::max(signal.physical_nbytes(), static_cast<size_t>(page_size)));
    const tt::tt_metal::distributed::ReplicatedBufferConfig replicated{.size = alloc_size};
    auto buf = tt::tt_metal::distributed::MeshBuffer::create(replicated, local, &mesh_device);
#ifdef TRACY_ENABLE
    // Use the backing buffer's device address as a unique token for Tracy.
    // Tracy treats it as an opaque key — it never dereferences it.
    void* tracy_ptr = reinterpret_cast<void*>(buf->get_backing_buffer()->address());
    TracyAllocN(tracy_ptr, static_cast<size_t>(alloc_size), "DRAM");
#endif
    return buf;
}

void include_interval(LogicalInterval& target, const LogicalInterval source) {
    if (source.empty()) {
        return;
    }
    if (target.empty()) {
        target = source;
        return;
    }
    target.begin = std::min(target.begin, source.begin);
    target.end = std::max(target.end, source.end);
}

[[nodiscard]] LogicalInterval predict_source_requirement(
    const LogicalInterval& output, const LiftingStep& step) noexcept {
    if (output.empty()) {
        return output;
    }
    const int64_t width = static_cast<int64_t>(step.coeffs().size());
    return LogicalInterval{
        .begin = output.begin - static_cast<int64_t>(step.shift()) - (width - 1),
        .end = output.end - static_cast<int64_t>(step.shift()),
    };
}

[[nodiscard]] std::pair<LogicalInterval, LogicalInterval> reverse_initial_requirements(
    const LiftingScheme& scheme, const LogicalInterval final_interval) {
    LogicalInterval even = final_interval;
    LogicalInterval odd = final_interval;
    for (auto step = scheme.steps().rbegin(); step != scheme.steps().rend(); ++step) {
        switch (step->type()) {
            case LiftingStepType::kPredict: include_interval(even, predict_source_requirement(odd, *step)); break;
            case LiftingStepType::kUpdate: include_interval(odd, predict_source_requirement(even, *step)); break;
            case LiftingStepType::kSwap: std::swap(even, odd); break;
            case LiftingStepType::kScaleEven:
            case LiftingStepType::kScaleOdd: break;
        }
    }
    return {even, odd};
}

[[nodiscard]] uint32_t predict_update_count(const LiftingScheme& scheme) noexcept {
    uint32_t count = 0;
    for (const auto& step : scheme.steps()) {
        if (step.type() == LiftingStepType::kPredict || step.type() == LiftingStepType::kUpdate) {
            ++count;
        }
    }
    return count;
}

[[nodiscard]] std::vector<CorePiece> make_core_pieces(
    const TransformGeometry& geometry, const LiftingScheme& scheme, const uint32_t core_count) {
    TT_FATAL(core_count > 0, "WaveletProgram requires at least one output core");
    const uint32_t output_sticks =
        checked_u32(ceil_div(geometry.output_length, static_cast<size_t>(kStickWidth)), "output sticks");
    TT_FATAL(core_count <= output_sticks, "cannot assign empty output ranges to LWT cores");
    const uint32_t quotient = output_sticks / core_count;
    const uint32_t remainder = output_sticks % core_count;
    const int64_t direct_shift = static_cast<int64_t>(scheme.tap_size() / 2);
    const int64_t even_length = static_cast<int64_t>(geometry.pad_split.output.even.length);
    const int64_t odd_length = static_cast<int64_t>(geometry.pad_split.output.odd.length);
    const int64_t maximum_length = std::max(even_length, odd_length);
    const int64_t guard =
        static_cast<int64_t>(predict_update_count(scheme)) * static_cast<int64_t>(kSpliceAdvanceElements);
    const int64_t terminal_tail =
        static_cast<int64_t>(predict_update_count(scheme)) * static_cast<int64_t>(kSpliceTailElements);

    std::vector<CorePiece> pieces;
    pieces.reserve(core_count);
    uint32_t output_stick_start = 0;
    for (uint32_t index = 0; index < core_count; ++index) {
        const uint32_t stick_count = quotient + (index < remainder ? 1U : 0U);
        const uint32_t output_begin = output_stick_start * kStickWidth;
        const uint32_t output_length = std::min(stick_count * kStickWidth, geometry.output_length - output_begin);
        const LogicalInterval final_interval{
            .begin = direct_shift + output_begin,
            .end = direct_shift + output_begin + output_length - 1,
        };
        const auto [required_even, required_odd] = reverse_initial_requirements(scheme, final_interval);
        TT_FATAL(!required_even.empty() && !required_odd.empty(), "empty LWT reverse-propagated source interval");

        const int64_t initial_begin = std::min(
            required_even.begin - static_cast<int64_t>(scheme.delay_even()),
            required_odd.begin - static_cast<int64_t>(scheme.delay_odd()));
        const int64_t initial_end = std::max(
            required_even.end - static_cast<int64_t>(scheme.delay_even()),
            required_odd.end - static_cast<int64_t>(scheme.delay_odd()));
        const int64_t output_source_begin =
            std::min<int64_t>(geometry.even_crop + output_begin, geometry.odd_crop + output_begin);
        int64_t source_start = 0;
        int64_t source_end = maximum_length - 1;
        int64_t virtual_prefix = terminal_tail;
        int64_t virtual_suffix = terminal_tail;
        if (core_count > 1) {
            const int64_t requested_start = std::min(initial_begin - guard, output_source_begin);
            source_start = std::max<int64_t>(0, requested_start);
            virtual_prefix = std::min<int64_t>(guard, source_start - requested_start);
            const int64_t guarded_end = initial_end + guard;
            source_end = std::min<int64_t>(maximum_length - 1, guarded_end);
            virtual_suffix = std::min<int64_t>(guard, std::max<int64_t>(0, guarded_end - source_end));
        }
        TT_FATAL(source_end >= source_start, "empty guarded LWT input piece");
        const uint32_t source_length = checked_u32(
            static_cast<size_t>(virtual_prefix + source_end - source_start + 1 + virtual_suffix), "piece length");
        const uint32_t splice_count = checked_u32(
            ceil_div(source_length + kSpliceTailElements, static_cast<size_t>(kSpliceAdvanceElements)),
            "piece splices");

        const int64_t approximation_forward_offset = virtual_prefix + geometry.even_crop + output_begin - source_start;
        const int64_t detail_forward_offset = virtual_prefix + geometry.odd_crop + output_begin - source_start;
        TT_FATAL(
            approximation_forward_offset >= 0 &&
                approximation_forward_offset + output_length <= static_cast<int64_t>(source_length) &&
                detail_forward_offset >= 0 &&
                detail_forward_offset + output_length <= static_cast<int64_t>(source_length),
            "owned output interval is outside its guarded LWT input piece");
        pieces.push_back(
            CorePiece{
                .source_start = checked_u32(static_cast<size_t>(source_start), "piece start"),
                .source_length = source_length,
                .source_prefix = checked_u32(static_cast<size_t>(virtual_prefix), "piece prefix"),
                .splice_count = splice_count,
                .approximation_offset =
                    checked_u32(static_cast<size_t>(approximation_forward_offset), "approximation offset"),
                .detail_offset = checked_u32(static_cast<size_t>(detail_forward_offset), "detail offset"),
                .output_stick_start = output_stick_start,
                .output_length = output_length,
            });
        output_stick_start += stick_count;
    }
    return pieces;
}

[[nodiscard]] uint64_t piece_working_bytes(
    const uint32_t maximum_splices, const uint32_t tile_bytes, const uint32_t stick_bytes) {
    const uint32_t chain_pages = maximum_splices * kTilePagesPerSplice;
    return static_cast<uint64_t>(chain_pages) * 10U * tile_bytes + static_cast<uint64_t>(tile_bytes) * 3U +
           static_cast<uint64_t>(stick_bytes) * 3U;
}

void create_cb(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& cores,
    const uint32_t index,
    const uint32_t pages,
    const uint32_t page_bytes) {
    auto config =
        tt::tt_metal::CircularBufferConfig(pages * page_bytes, {{index, kDataFormat}}).set_page_size(index, page_bytes);
    static_cast<void>(tt::tt_metal::CreateCircularBuffer(program, cores, config));
#ifdef TRACY_ENABLE
    // CBs have no persistent device address on the host. Use (cores ptr XOR index) as a
    // unique token — stable for the lifetime of this program object, which is all Tracy needs.
    void* tracy_ptr = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(&cores) ^ (static_cast<uintptr_t>(index) << 16));
    TracyAllocN(tracy_ptr, static_cast<size_t>(pages) * page_bytes, "L1/SRAM");
#endif
}

[[nodiscard]] StreamState apply_predict_update(
    const StreamState source, const StreamState base, const LiftingStep& step) {
    const size_t width = step.coeffs().size();
    TT_FATAL(width > 0 && source.length >= width, "lifting stencil is wider than its source stream");

    const int32_t convolution_shift = source.shift + step.shift() + static_cast<int32_t>(width) - 1;
    const size_t convolution_length = source.length - width + 1;
    const int64_t output_start = std::max<int64_t>(base.shift, convolution_shift);
    const int64_t output_end = std::min<int64_t>(
        static_cast<int64_t>(base.shift) + static_cast<int64_t>(base.length),
        static_cast<int64_t>(convolution_shift) + static_cast<int64_t>(convolution_length));
    TT_FATAL(output_end >= output_start, "lifting step has an empty valid support");
    return StreamState{
        .shift = static_cast<int32_t>(output_start),
        .length = static_cast<size_t>(output_end - output_start),
    };
}

[[nodiscard]] TransformGeometry make_geometry(const SignalBuffer& input, const LiftingScheme& scheme) {
    const uint32_t pad = scheme.tap_size() - 1;
    TransformGeometry geometry{
        .pad_split = make_pad_split_1d_layout(
            input, 0, 0, Pad1DConfig{.mode = BoundaryMode::kSymmetric, .left = pad, .right = pad}),
    };
    StreamState even{.shift = scheme.delay_even(), .length = geometry.pad_split.output.even.length};
    StreamState odd{.shift = scheme.delay_odd(), .length = geometry.pad_split.output.odd.length};
    int32_t device_even_shift = scheme.delay_even();
    int32_t device_odd_shift = scheme.delay_odd();

    for (const auto& step : scheme.steps()) {
        switch (step.type()) {
            case LiftingStepType::kPredict:
                odd = apply_predict_update(even, odd, step);
                device_odd_shift = std::min(
                    device_odd_shift,
                    device_even_shift + step.shift() + static_cast<int32_t>(step.coeffs().size()) - 1);
                break;
            case LiftingStepType::kUpdate:
                even = apply_predict_update(odd, even, step);
                device_even_shift = std::min(
                    device_even_shift,
                    device_odd_shift + step.shift() + static_cast<int32_t>(step.coeffs().size()) - 1);
                break;
            case LiftingStepType::kScaleEven:
            case LiftingStepType::kScaleOdd: break;
            case LiftingStepType::kSwap:
                std::swap(even, odd);
                std::swap(device_even_shift, device_odd_shift);
                geometry.swapped = !geometry.swapped;
                break;
        }
    }

    const size_t output_length = canonical_output_length(input.length, scheme);
    const int32_t direct_shift = static_cast<int32_t>(scheme.tap_size() / 2);
    TT_FATAL(direct_shift >= even.shift && direct_shift >= odd.shift, "canonical crop begins outside device support");
    const size_t even_valid_crop = static_cast<size_t>(direct_shift - even.shift);
    const size_t odd_valid_crop = static_cast<size_t>(direct_shift - odd.shift);
    TT_FATAL(
        even_valid_crop + output_length <= even.length && odd_valid_crop + output_length <= odd.length,
        "canonical coefficient window is outside computed support");
    TT_FATAL(
        direct_shift >= device_even_shift && direct_shift >= device_odd_shift,
        "canonical crop precedes computed device-chain origins");

    const size_t chain_length =
        std::max(geometry.pad_split.output.even.length, geometry.pad_split.output.odd.length) + kSpliceTailElements;
    geometry.final_even = even;
    geometry.final_odd = odd;
    geometry.even_crop = checked_u32(static_cast<size_t>(direct_shift - device_even_shift), "even crop");
    geometry.odd_crop = checked_u32(static_cast<size_t>(direct_shift - device_odd_shift), "odd crop");
    geometry.output_length = checked_u32(output_length, "coefficient length");
    geometry.splice_count = checked_u32(ceil_div(chain_length, static_cast<size_t>(kSpliceAdvanceElements)), "splices");
    return geometry;
}

void run_workload(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue, tt::tt_metal::distributed::MeshWorkload& workload) {
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
    tt::tt_metal::distributed::Finish(command_queue);
}

}  // namespace

bool validate_lifting_scheme(const LiftingScheme& scheme) {
    if (scheme.tap_size() == 0 || scheme.steps().empty()) {
        return false;
    }
    for (const auto& step : scheme.steps()) {
        const auto width = step.coeffs().size();
        switch (step.type()) {
            case LiftingStepType::kPredict:
            case LiftingStepType::kUpdate:
                if (width == 0 || width > 17 || step.shift() < -16 || step.shift() > 16) {
                    return false;
                }
                break;
            case LiftingStepType::kScaleEven:
            case LiftingStepType::kScaleOdd:
                if (width != 1) {
                    return false;
                }
                break;
            case LiftingStepType::kSwap:
                if (width != 0) {
                    return false;
                }
                break;
        }
    }
    return true;
}

size_t canonical_output_length(const size_t input_length, const LiftingScheme& scheme) noexcept {
    return (input_length + static_cast<size_t>(scheme.tap_size()) - 1) / 2;
}

std::vector<uint32_t> pack_compile_time_steps(const LiftingScheme& scheme) {
    TT_FATAL(validate_lifting_scheme(scheme), "invalid generated lifting scheme");
    std::vector<uint32_t> packed;
    for (const auto& step : scheme.steps()) {
        const uint32_t type = static_cast<uint32_t>(step.type());
        const uint32_t width = static_cast<uint32_t>(step.coeffs().size());
        packed.push_back((width << 3) | type);
        if (step.type() == LiftingStepType::kPredict || step.type() == LiftingStepType::kUpdate) {
            packed.push_back(static_cast<uint32_t>(step.shift()));
        }
        for (const float coefficient : step.coeffs()) {
            packed.push_back(std::bit_cast<uint32_t>(coefficient));
        }
    }
    return packed;
}

std::string packed_program_cases(const std::vector<uint32_t>& packed) {
    std::string cases;
    for (size_t index = 0; index < packed.size(); ++index) {
        cases += "case " + std::to_string(index) + "u:return " + std::to_string(packed[index]) + "u;";
    }
    return cases;
}

uint32_t predict_update_width_mask(const LiftingScheme& scheme) {
    uint32_t mask = 0;
    for (const auto& step : scheme.steps()) {
        if (step.type() == LiftingStepType::kPredict || step.type() == LiftingStepType::kUpdate) {
            mask |= 1U << (static_cast<uint32_t>(step.coeffs().size()) - 1U);
        }
    }
    return mask;
}

class WaveletProgram::Impl {
public:
    Impl(
        const std::filesystem::path& kernel_root,
        tt::tt_metal::distributed::MeshDevice& mesh_device,
        tt::tt_metal::distributed::MeshCommandQueue& command_queue,
        const tt::tt_metal::CoreCoord& core,
        const LiftingScheme& scheme,
        const size_t input_length) :
        mesh_device_(mesh_device), command_queue_(command_queue), input_length_(input_length) {
        TT_FATAL(input_length > 0, "WaveletProgram requires a non-empty signal");
        TT_FATAL(validate_lifting_scheme(scheme), "invalid generated lifting scheme");

        SignalBuffer input_desc{
            .length = input_length, .stick_width = kStickWidth, .element_size_bytes = sizeof(float)};
        geometry_ = make_geometry(input_desc, scheme);
        input_ = make_signal_buffer(mesh_device_, input_desc);
        even_ = make_signal_buffer(mesh_device_, geometry_.pad_split.output.even);
        odd_ = make_signal_buffer(mesh_device_, geometry_.pad_split.output.odd);
        const SignalBuffer output_desc{
            .length = geometry_.output_length, .stick_width = kStickWidth, .element_size_bytes = sizeof(float)};
        approximation_ = make_signal_buffer(mesh_device_, output_desc);
        detail_ = make_signal_buffer(mesh_device_, output_desc);

        input_desc.dram_address = input_->get_backing_buffer()->address();
        geometry_.pad_split = make_pad_split_1d_layout(
            input_desc,
            even_->get_backing_buffer()->address(),
            odd_->get_backing_buffer()->address(),
            geometry_.pad_split.pad_config);

        auto pad_program = create_pad_split_1d_program(
            kernel_root,
            core,
            *input_->get_backing_buffer(),
            *even_->get_backing_buffer(),
            *odd_->get_backing_buffer(),
            geometry_.pad_split);
        const auto device_range = tt::tt_metal::distributed::MeshCoordinateRange(mesh_device_.shape());
        pad_workload_.add_program(device_range, std::move(pad_program.program));

        tt::tt_metal::Program transform = tt::tt_metal::CreateProgram();
        const uint32_t tile_bytes = tt::tile_size(kDataFormat);
        const uint32_t stick_bytes = input_desc.aligned_stick_bytes(kNocAlignmentBytes);
        const auto& sub_device_ids = mesh_device_.get_sub_device_ids();
        TT_FATAL(!sub_device_ids.empty(), "mesh device provides no worker sub-devices");
        const auto available_set =
            mesh_device_.worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_ids.front());
        auto available_cores = tt::tt_metal::corerange_to_cores(available_set);
        TT_FATAL(!available_cores.empty(), "mesh device provides no Tensix cores");
        const auto preferred = std::find(available_cores.begin(), available_cores.end(), core);
        TT_FATAL(preferred != available_cores.end(), "requested pad/split core is not a Tensix worker");
        std::iter_swap(available_cores.begin(), preferred);

        const uint32_t output_sticks =
            checked_u32(ceil_div(geometry_.output_length, static_cast<size_t>(kStickWidth)), "output sticks");
        const size_t physical_stream_length =
            std::max(geometry_.pad_split.output.even.length, geometry_.pad_split.output.odd.length);
        const size_t dependency_guard = predict_update_count(scheme) * static_cast<size_t>(kSpliceAdvanceElements);
        uint32_t selected_core_count =
            physical_stream_length <= dependency_guard
                ? 1U
                : std::min<uint32_t>(checked_u32(available_cores.size(), "available cores"), output_sticks);
        uint32_t maximum_splices = 0;
        std::vector<CorePiece> pieces;
        uint64_t working_bytes = 0;
        while (selected_core_count > 0) {
            auto candidate = make_core_pieces(geometry_, scheme, selected_core_count);
            uint32_t candidate_maximum_splices = 0;
            for (const auto& piece : candidate) {
                candidate_maximum_splices = std::max(candidate_maximum_splices, piece.splice_count);
            }
            const uint64_t candidate_bytes = piece_working_bytes(candidate_maximum_splices, tile_bytes, stick_bytes);
            working_bytes = candidate_bytes;
            if (candidate_bytes + kL1ReserveBytes <= mesh_device_.l1_size_per_core()) {
                pieces = std::move(candidate);
                maximum_splices = candidate_maximum_splices;
                working_bytes = candidate_bytes;
                break;
            }
            --selected_core_count;
        }
        TT_FATAL(
            selected_core_count > 0,
            "signal needs {} L1 bytes for one guarded LWT piece, but one Tensix core provides {} bytes",
            working_bytes,
            mesh_device_.l1_size_per_core());
        std::vector<tt::tt_metal::CoreRange> active_ranges;
        active_ranges.reserve(selected_core_count);
        for (uint32_t index = 0; index < selected_core_count; ++index) {
            active_ranges.emplace_back(available_cores[index]);
        }
        const tt::tt_metal::CoreRangeSet active_cores(std::move(active_ranges));
        const uint32_t chain_pages = maximum_splices * kTilePagesPerSplice;

        create_cb(transform, active_cores, kCbEven, chain_pages, tile_bytes);
        create_cb(transform, active_cores, kCbOdd, chain_pages, tile_bytes);
        create_cb(transform, active_cores, kCbEvenCache, 1, stick_bytes);
        create_cb(transform, active_cores, kCbOddCache, 1, stick_bytes);
        create_cb(transform, active_cores, kCbDone, 1, tile_bytes);
        create_cb(transform, active_cores, kCbWriterScratch, 1, stick_bytes);
        create_cb(transform, active_cores, kCbEvenWork, chain_pages * 2, tile_bytes);
        create_cb(transform, active_cores, kCbOddWork, chain_pages * 2, tile_bytes);
        create_cb(transform, active_cores, kCbFinalEven, chain_pages, tile_bytes);
        create_cb(transform, active_cores, kCbFinalOdd, chain_pages, tile_bytes);
        create_cb(transform, active_cores, kCbPrediction, chain_pages * 2, tile_bytes);
        create_cb(transform, active_cores, kCbBaseRecoverState, 1, tile_bytes);
        create_cb(transform, active_cores, kCbPredictionRecoverState, 1, tile_bytes);

        std::vector<uint32_t> reader_args;
        tt::tt_metal::TensorAccessorArgs(*even_->get_backing_buffer()).append_to(reader_args);
        tt::tt_metal::TensorAccessorArgs(*odd_->get_backing_buffer()).append_to(reader_args);
        const std::unordered_map<std::string, uint32_t> reader_named{
            {"cb_even", kCbEven},
            {"cb_odd", kCbOdd},
            {"cb_even_cache", kCbEvenCache},
            {"cb_odd_cache", kCbOddCache},
            {"cache_bytes", stick_bytes},
        };
        const auto reader = tt::tt_metal::CreateKernel(
            transform,
            kernel_path(kernel_root, kLwtReaderKernel),
            active_cores,
            tt::tt_metal::ReaderDataMovementConfig(reader_args, {}, reader_named));

        const std::vector<uint32_t> compute_args = pack_compile_time_steps(scheme);
        const std::string compute_program_cases = packed_program_cases(compute_args);
        const uint32_t width_mask = predict_update_width_mask(scheme);
        std::vector<UnpackToDestMode> unpack_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
        unpack_mode[kCbEven] = UnpackToDestMode::UnpackToDestFp32;
        unpack_mode[kCbOdd] = UnpackToDestMode::UnpackToDestFp32;
        unpack_mode[kCbEvenWork] = UnpackToDestMode::UnpackToDestFp32;
        unpack_mode[kCbOddWork] = UnpackToDestMode::UnpackToDestFp32;
        unpack_mode[kCbFinalEven] = UnpackToDestMode::UnpackToDestFp32;
        unpack_mode[kCbFinalOdd] = UnpackToDestMode::UnpackToDestFp32;
        unpack_mode[kCbPrediction] = UnpackToDestMode::UnpackToDestFp32;
        unpack_mode[kCbBaseRecoverState] = UnpackToDestMode::UnpackToDestFp32;
        unpack_mode[kCbPredictionRecoverState] = UnpackToDestMode::UnpackToDestFp32;
        const std::unordered_map<std::string, uint32_t> compute_named{
            {"cb_even", kCbEven},
            {"cb_odd", kCbOdd},
            {"cb_even_work", kCbEvenWork},
            {"cb_odd_work", kCbOddWork},
            {"cb_final_even", kCbFinalEven},
            {"cb_final_odd", kCbFinalOdd},
            {"cb_prediction", kCbPrediction},
            {"cb_base_recover_state", kCbBaseRecoverState},
            {"cb_prediction_recover_state", kCbPredictionRecoverState},
            {"cb_out", kCbDone},
            {"even_delay", static_cast<uint32_t>(scheme.delay_even())},
            {"odd_delay", static_cast<uint32_t>(scheme.delay_odd())},
            {"num_steps", static_cast<uint32_t>(scheme.steps().size())},
            {"width_mask", width_mask},
        };
        const auto compute = tt::tt_metal::CreateKernel(
            transform,
            kernel_path(kernel_root, kLwtComputeKernel),
            active_cores,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .dst_full_sync_en = true,
                .unpack_to_dest_mode = unpack_mode,
                .defines = {{"LWT_PROGRAM_CASES", compute_program_cases}},
                .named_compile_args = compute_named,
            });

        std::vector<uint32_t> writer_args;
        tt::tt_metal::TensorAccessorArgs(*approximation_->get_backing_buffer()).append_to(writer_args);
        tt::tt_metal::TensorAccessorArgs(*detail_->get_backing_buffer()).append_to(writer_args);
        const std::unordered_map<std::string, uint32_t> writer_named{
            {"cb_even", kCbFinalEven},
            {"cb_odd", kCbFinalOdd},
            {"cb_done", kCbDone},
            {"cb_scratch", kCbWriterScratch},
            {"final_swapped", 0U},
            {"stick_width", kStickWidth},
        };
        const auto writer = tt::tt_metal::CreateKernel(
            transform,
            kernel_path(kernel_root, kLwtWriterKernel),
            active_cores,
            tt::tt_metal::WriterDataMovementConfig(writer_args, {}, writer_named));

        for (uint32_t index = 0; index < selected_core_count; ++index) {
            const auto& piece = pieces[index];
            const auto& transform_core = available_cores[index];
            tt::tt_metal::SetRuntimeArgs(
                transform,
                reader,
                transform_core,
                std::array<uint32_t, 8>{
                    piece.splice_count,
                    static_cast<uint32_t>(even_->get_backing_buffer()->address()),
                    static_cast<uint32_t>(odd_->get_backing_buffer()->address()),
                    checked_u32(geometry_.pad_split.output.even.length, "even length"),
                    checked_u32(geometry_.pad_split.output.odd.length, "odd length"),
                    piece.source_start,
                    piece.source_length,
                    piece.source_prefix,
                });
            tt::tt_metal::SetRuntimeArgs(
                transform, compute, transform_core, std::array<uint32_t, 1>{piece.splice_count});
            tt::tt_metal::SetRuntimeArgs(
                transform,
                writer,
                transform_core,
                std::array<uint32_t, 7>{
                    piece.splice_count,
                    static_cast<uint32_t>(approximation_->get_backing_buffer()->address()),
                    static_cast<uint32_t>(detail_->get_backing_buffer()->address()),
                    piece.approximation_offset,
                    piece.detail_offset,
                    piece.output_stick_start,
                    piece.output_length,
                });
        }
        transform_workload_.add_program(device_range, std::move(transform));
    }

    [[nodiscard]] WaveletCoefficients execute(const std::span<const float> signal) {
        TT_FATAL(!executed_, "WaveletProgram instances execute exactly once");
        TT_FATAL(signal.size() == input_length_, "signal length differs from the compiled WaveletProgram length");
        executed_ = true;

        std::vector<float> input(signal.begin(), signal.end());
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue_, input_, input, false);
        run_workload(command_queue_, pad_workload_);
        run_workload(command_queue_, transform_workload_);

        WaveletCoefficients result;
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue_, result.approximation, approximation_, true);
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue_, result.detail, detail_, true);
        result.approximation.resize(geometry_.output_length);
        result.detail.resize(geometry_.output_length);
        return result;
    }

    ~Impl() {
#ifdef TRACY_ENABLE
        // Emit TracyFreeN for each DRAM buffer to close the alloc/free pairs.
        // Must be called before the shared_ptrs are reset so addresses are still valid.
        auto tracy_free_dram = [](const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& buf) {
            if (buf) {
                void* tracy_ptr = reinterpret_cast<void*>(buf->get_backing_buffer()->address());
                TracyFreeN(tracy_ptr, "DRAM");
            }
        };
        tracy_free_dram(input_);
        tracy_free_dram(even_);
        tracy_free_dram(odd_);
        tracy_free_dram(approximation_);
        tracy_free_dram(detail_);
#endif
    }

private:
    tt::tt_metal::distributed::MeshDevice& mesh_device_;
    tt::tt_metal::distributed::MeshCommandQueue& command_queue_;
    size_t input_length_;
    bool executed_{false};
    TransformGeometry geometry_{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> input_;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> even_;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> odd_;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> approximation_;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> detail_;
    tt::tt_metal::distributed::MeshWorkload pad_workload_;
    tt::tt_metal::distributed::MeshWorkload transform_workload_;
};

WaveletProgram::WaveletProgram(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const tt::tt_metal::CoreCoord& core,
    const LiftingScheme& scheme,
    const size_t input_length) :
    impl_(std::make_unique<Impl>(kernel_root, mesh_device, command_queue, core, scheme, input_length)) {}

WaveletProgram::~WaveletProgram() = default;
WaveletProgram::WaveletProgram(WaveletProgram&&) noexcept = default;
WaveletProgram& WaveletProgram::operator=(WaveletProgram&&) noexcept = default;

WaveletCoefficients WaveletProgram::execute(const std::span<const float> signal) { return impl_->execute(signal); }

}  // namespace ttwv
