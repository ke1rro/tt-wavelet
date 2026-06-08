#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "lwt_readers_utils.hpp"

namespace {

constexpr uint32_t kStepPredict = static_cast<uint32_t>(ttwv::StepType::kPredict);
constexpr uint32_t kStepUpdate = static_cast<uint32_t>(ttwv::StepType::kUpdate);
constexpr uint32_t kStepScaleEven = static_cast<uint32_t>(ttwv::StepType::kScaleEven);
constexpr uint32_t kStepScaleOdd = static_cast<uint32_t>(ttwv::StepType::kScaleOdd);

constexpr uint32_t kBlockElements = ttwv::device_protocol::kLwtHalfStickElements;
constexpr uint32_t kRowsPerGroup = ttwv::device_protocol::kLwtRowsPerGroup;
constexpr uint32_t kInputTilesPerGroup = 2;
constexpr uint32_t kOutputBlocksPerRow = ttwv::device_protocol::kLwtOutputBlocksPerRow;
constexpr uint32_t kGroupOutputElements = ttwv::device_protocol::kLwtGroupOutputElements;

template <typename SrcAccessor>
ALWI float read_dense_or_zero(
    const SrcAccessor& src,
    ttwv::kernels::primitives::StickReadCache& cache,
    const uint32_t logical_length,
    const uint32_t logical_index) {
    return logical_index < logical_length ? ttwv::kernels::primitives::read_source_value(src, cache, logical_index)
                                          : 0.0F;
}

template <typename ConfigAccessor>
ALWI const uint32_t* load_route_config(
    const ConfigAccessor& config, const uint32_t cb_config, const uint32_t route_index) {
    cb_reserve_back(cb_config, 1);
    noc_async_read(
        config.get_noc_addr(route_index), get_write_ptr(cb_config), ttwv::device_protocol::kRouteConfigPageBytes);
    noc_async_read_barrier();
    cb_push_back(cb_config, 1);
    cb_wait_front(cb_config, 1);
    return reinterpret_cast<const uint32_t*>(get_read_ptr(cb_config));
}

ALWI uint32_t get_splicized_idx(const uint32_t row, const uint32_t col) {
    return (col / 32) * 1024 + get_tilized_idx(row, col);
}

void splicize_full(const ttwv::kernels::primitives::InputStream& stream, float* splice, float* buffer) {
    for (uint32_t row = 0; row < 32; row++) {
        for (uint32_t col = 16; col < 64; col++) {
            if (stream.empty()) {
                return;
            }

            const float value = stream.pop();

            splice[get_splicized_idx(row, col)] = value
            if (col >= 48 && row < 31) {
                splice[get_splicized_idx(row + 1, col - 48)] = value;
            } else if (col >= 48 && row == 31) {
                buffer[col - 48] = value;
            }
        }
    }
}

void splicize_partial(const ttwv::kernels::primitives::InputStream& stream, float* splice) {
    for (uint32_t row = 0; row < 32; row++) {
        for (uint32_t col = 0; col < 48; col++) {
            if (stream.empty()) {
                return;
            }

            splice[get_splicized_idx(row, col)] = stream.pop();
        }
    }
}

void step_predict_update(
    const InputStream& src,
    const InputStream& base,
    const uint32_t src_pad,
    const uint32_t num_splices,
    const uint32_t cb_src,
    const uint32_t cb_base) {
    float src_buffer[16];
    for (uint32_t splice = 0; splice < num_splices; ++splice) {
        cb_reserve_back(cb_src, 2);
        float* src_ptr = reinterpret_cast<float*>(get_write_ptr(cb_src));

        if (splice == 0) {
            for (uint32_t i = src_pad; i < 16; i++) {
                src_ptr[i] = src.pop();
            }
        } else {
            // Copy buffer: TODO may be replaced with memcpy
            for (uint32_t i = 0; i < 16; i++) {
                src_ptr[i] = src_buffer[i];
            }
        }
        splicize_full(src_ptr, src, src_buffer);

        cb_push_back(cb_src, 2);

        cb_reserve_back(cb_base, 2);
        float* base_ptr = reinterpret_cast<float*>(get_write_ptr(cb_base));
        splicize_partial(base_ptr, base);
        cb_push_back(cb_base, 2);
    }
}

void step_scale(
    const InputStream& src,
    const uint32_t num_splices,
    const uint32_t cb_src) {
    for (uint32_t splice = 0; splice < num_splices; ++splice) {
        cb_reserve_back(cb_src, 2);
        float* src_ptr = reinterpret_cast<float*>(get_write_ptr(cb_src));
        splicize_partial(src_ptr, src);
        cb_push_back(cb_src, 2);
    }
}

template <uint32_t arg_base, uint32_t num_steps>
inline void unroll_step(
    const InputStream& even_stream,
    const InputStream& odd_stream,
    const uint32_t cb_input,
    const uint32_t cb_base
) {

    if constexpr (num_steps > 0) {
        constexpr uint32_t kMeta = get_compile_time_arg_val(arg_base);
        constexpr uint8_t kType = kMeta & 0x7;

        constexpr uint32_t splice_number = get_compile_time_arg_val(arg_base + 1);

        even_stream.step(kType);
        odd_stream.step(kType);

        if constexpr(kType == kStepTypeScaleEven) {
            step_scale(even_stream, splice_number, cb_input);
            unroll_step<arg_base + 2, num_steps - 1>(even_stream, odd_stream, cb_input, cb_base, cb_sync);
        } else if constexpr(kType == kStepTypeScaleOdd) {
            step_scale(odd_stream, splice_number, cb_input);
            unroll_step<arg_base + 2, num_steps - 1>(even_stream, odd_stream, cb_input, cb_base, cb_sync);
        } else if constexpr(kType == kStepTypeSwap) {
            unroll_step<arg_base + 2, num_steps - 1>(odd_stream, even_stream, cb_input, cb_base, cb_sync);
        } else if constexpr(kType == kStepTypePredict) {
            constexpr uint32_t kPad = (kMeta >> 3);

            step_predict_update<K>(cb_input, cb_base, cb_output, h_coeffs, splice_number);
            unroll_step<arg_base + 2 + K, num_steps - 1>(cb_input, cb_base, cb_output);
        } else {
            static_assert(false, "Unsupported step type");
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_even = get_named_compile_time_arg_val("cb_even");
    constexpr uint32_t cb_odd = get_named_compile_time_arg_val("cb_odd");
    constexpr uint32_t cb_input = get_named_compile_time_arg_val("cb_input");
    constexpr uint32_t cb_base = get_named_compile_time_arg_val("cb_base");
    constexpr uint32_t num_steps = get_named_compile_time_arg_val("num_steps");
    constexpr auto tensor_accessor_args = TensorAccessorArgs<0>();

    const uint32_t even_addr = get_arg_val<uint32_t>(0);
    const uint32_t even_length = get_arg_val<uint32_t>(1);
    const uint32_t odd_addr = get_arg_val<uint32_t>(2);
    const uint32_t odd_length = get_arg_val<uint32_t>(3);

    const auto config = TensorAccessor(config_args, route_config_addr, ttwv::device_protocol::kRouteConfigPageBytes);

    const auto even_accessor = TensorAccessor(tensor_accessor_args, even_addr, ttwv::device_protocol::kBlobBytes);
    const auto odd_accessor = TensorAccessor(tensor_accessor_args, odd_addr, ttwv::device_protocol::kBlobBytes);

    const InputStream even_stream(cb_input, even_accessor, even_length);
    const InputStream odd_stream(cb_input, odd_accessor, odd_length);

    unroll_step<0, num_steps>(even_stream, odd_stream, cb_input, cb_base);
}
