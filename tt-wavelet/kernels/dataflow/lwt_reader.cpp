#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "lwt_readers_utils.hpp"

namespace {

constexpr uint32_t kStepPredict = static_cast<uint32_t>(ttwv::StepType::kPredict);
constexpr uint32_t kStepUpdate = static_cast<uint32_t>(ttwv::StepType::kUpdate);
constexpr uint32_t kStepScaleEven = static_cast<uint32_t>(ttwv::StepType::kScaleEven);
constexpr uint32_t kStepScaleOdd = static_cast<uint32_t>(ttwv::StepType::kScaleOdd);


// FROM tt-metal SOURCE CODE
ALWI uint32_t get_tilized_idx(uint32_t h, uint32_t w) {
    using namespace tt::constants;
    // Get local coordinates within the tile
    uint32_t local_row = h % TILE_HEIGHT;
    uint32_t local_col = w % TILE_WIDTH;
    // Determine the index offset based on which quadrant we're in
    uint32_t offset = 0;
    // If we're in the right half (columns beyond FACE_WIDTH)
    if (local_col >= FACE_WIDTH) {
        local_col -= FACE_WIDTH;
        offset += FACE_HEIGHT * FACE_WIDTH;  // Right face offset
    }
    // If we're in the bottom half (rows beyond FACE_WIDTH)
    if (local_row >= FACE_WIDTH) {
        local_row -= FACE_WIDTH;
        offset += FACE_HEIGHT * TILE_WIDTH;  // Bottom face offset
    }
    // Final index within the tile
    uint32_t index = offset + local_row * FACE_WIDTH + local_col;
    return index;
}

ALWI uint32_t get_splicized_idx(const uint32_t row, const uint32_t col) {
    return (col / 32) * 1024 + get_tilized_idx(row, col);
}

ALWI void splicize_full(InputStream& stream, float* splice, float* buffer) {
    for (uint32_t row = 0; row < 32; row++) {
        for (uint32_t col = 16; col < 64; col++) {
            if (stream.empty()) {
                return;
            }

            const float value = stream.pop();

            splice[get_splicized_idx(row, col)] = value;
            if (col >= 48 && row < 31) {
                splice[get_splicized_idx(row + 1, col - 48)] = value;
            } else if (col >= 48 && row == 31) {
                buffer[col - 48] = value;
            }
        }
    }
}

ALWI void splicize_partial(InputStream& stream, float* splice) {
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
    InputStream& src,
    InputStream& base,
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
        splicize_full(src, src_ptr, src_buffer);

        cb_push_back(cb_src, 2);

        cb_reserve_back(cb_base, 2);
        float* base_ptr = reinterpret_cast<float*>(get_write_ptr(cb_base));
        splicize_partial(base, base_ptr);
        cb_push_back(cb_base, 2);
    }
}

void step_scale(
    InputStream& src,
    const uint32_t num_splices,
    const uint32_t cb_src) {
    for (uint32_t splice = 0; splice < num_splices; ++splice) {
        cb_reserve_back(cb_src, 2);
        float* src_ptr = reinterpret_cast<float*>(get_write_ptr(cb_src));
        splicize_partial(src, src_ptr);
        cb_push_back(cb_src, 2);
    }
}

template <uint32_t arg_base, uint32_t num_steps>
inline void unroll_step(
    InputStream& even_stream,
    InputStream& odd_stream,
    const uint32_t cb_input,
    const uint32_t cb_base
) {

    if constexpr (num_steps > 0) {
        constexpr uint32_t kMeta = get_compile_time_arg_val(arg_base);
        constexpr uint8_t kType = kMeta & 0x7;
        constexpr uint32_t kEvenMode = get_compile_time_arg_val(arg_base + 1);
        constexpr uint32_t kOddMode = get_compile_time_arg_val(arg_base + 2);
        constexpr uint32_t splice_number = get_compile_time_arg_val(arg_base + 3);

        even_stream.step(kEvenMode);
        odd_stream.step(kOddMode);

        constexpr uint32_t kEvenSkip = get_compile_time_arg_val(arg_base + 4);
        constexpr uint32_t kOddSkip = get_compile_time_arg_val(arg_base + 5);
        even_stream.skip(kEvenSkip);
        odd_stream.skip(kOddSkip);

        constexpr uint32_t new_arg_base = arg_base + 6;

        if constexpr(kType == kStepTypeScaleEven) {
            step_scale(even_stream, splice_number, cb_input);
            unroll_step<new_arg_base, num_steps - 1>(even_stream, odd_stream, cb_input, cb_base);
        } else if constexpr(kType == kStepTypeScaleOdd) {
            step_scale(odd_stream, splice_number, cb_input);
            unroll_step<new_arg_base, num_steps - 1>(even_stream, odd_stream, cb_input, cb_base);
        } else if constexpr(kType == kStepTypeSwap) {
            unroll_step<new_arg_base, num_steps - 1>(odd_stream, even_stream, cb_input, cb_base);
        } else if constexpr(kType == kStepTypePredict) {
            constexpr uint32_t pad = (kMeta >> 3);

            step_predict_update(even_stream, odd_stream, pad, splice_number, cb_input, cb_base);
            unroll_step<new_arg_base, num_steps - 1>(even_stream, odd_stream, cb_input, cb_base);
        } else if constexpr(kType == kStepTypeUpdate) {
            constexpr uint32_t pad = (kMeta >> 3);

            step_predict_update(odd_stream, even_stream, pad, splice_number, cb_input, cb_base);
            unroll_step<new_arg_base, num_steps - 1>(even_stream, odd_stream, cb_input, cb_base);
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

    const auto even_accessor = TensorAccessor(tensor_accessor_args, even_addr, ttwv::device_protocol::kBlobBytes);
    const auto odd_accessor = TensorAccessor(tensor_accessor_args, odd_addr, ttwv::device_protocol::kBlobBytes);

    const InputStream even_stream(cb_even, even_accessor, even_length);
    const InputStream odd_stream(cb_odd, odd_accessor, odd_length);

    unroll_step<0, num_steps>(even_stream, odd_stream, cb_input, cb_base);
}
