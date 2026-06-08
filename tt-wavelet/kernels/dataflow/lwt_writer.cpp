#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "lwt_tile_row_major_utils.hpp"

namespace {

using get_splicized_idx = ttwv::kernels::primitives::get_splicized_idx;

ALWI void unsplicize(float* splice, OutputStream& stream) {
    for (uint32_t row = 0; row < 32; row++) {
        for (uint32_t col = 0; col < 48; col++) {
            if (stream.full()) {
                return;
            }

            stream.push(splice[get_splicized_idx(row, col)]);
        }
    }
}

void step(OutputStream& stream, const uint32_t num_splices, const uint32_t cb) {
    for (uint32_t splice = 0; splice < num_splices; splice++) {
        cb_wait_front(cb, 2);
        float* ptr = reinterpret_cast<float*>(get_read_ptr(cb));
        unsplicize(splice, stream);
        cb_pop_front(cb_input, 2);
    }
}

template <uint32_t arg_base, uint32_t num_steps>
inline void unroll_step(
    InputStream& even_stream,
    InputStream& odd_stream,
    const uint32_t cb_output
) {

    if constexpr (num_steps > 0) {
        constexpr uint32_t kMeta = get_compile_time_arg_val(arg_base);
        constexpr uint8_t kType = kMeta & 0x7;
        constexpr uint32_t kEvenMode = get_compile_time_arg_val(arg_base + 1);
        constexpr uint32_t kOddMode = get_compile_time_arg_val(arg_base + 2);
        constexpr uint32_t splice_number = get_compile_time_arg_val(arg_base + 3);

        even_stream.step(kEvenMode);
        odd_stream.step(kOddMode);

        constexpr uint32_t new_arg_base = arg_base + 4;

        if constexpr(kType == kStepTypeScaleEven || kStepTypeUpdate) {
            step(even_stream, splice_number, cb_output);
            unroll_step<new_arg_base, num_steps - 1>(even_stream, odd_stream, cb_output);
        } else if constexpr(kType == kStepTypeScaleOdd || kStepTypePredict) {
            step(odd_stream, splice_number, cb_output);
            unroll_step<new_arg_base, num_steps - 1>(even_stream, odd_stream, cb_output);
        } else if constexpr(kType == kStepTypeSwap) {
            unroll_step<new_arg_base, num_steps - 1>(odd_stream, even_stream, cb_output);
        }
    } else {
        even_stream.flush();
        odd_stream.flush();
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_even = get_named_compile_time_arg_val("cb_even");
    constexpr uint32_t cb_odd = get_named_compile_time_arg_val("cb_odd");
    constexpr uint32_t cb_output = get_named_compile_time_arg_val("cb_output");
    constexpr uint32_t num_steps = get_named_compile_time_arg_val("num_steps");
    constexpr auto tensor_accessor_args = TensorAccessorArgs<0>();

    const uint32_t even_addr = get_arg_val<uint32_t>(0);
    const uint32_t even_length = get_arg_val<uint32_t>(1);
    const uint32_t odd_addr = get_arg_val<uint32_t>(2);
    const uint32_t odd_length = get_arg_val<uint32_t>(3);

    const auto even_accessor = TensorAccessor(tensor_accessor_args, even_addr, ttwv::device_protocol::kBlobBytes);
    const auto odd_accessor = TensorAccessor(tensor_accessor_args, odd_addr, ttwv::device_protocol::kBlobBytes);

    const OutputStream even_stream(cb_even, even_accessor, even_length);
    const OutputStream odd_stream(cb_odd, odd_accessor, odd_length);

    unroll_step<0, num_steps>(even_stream, odd_stream, cb_output);
}
