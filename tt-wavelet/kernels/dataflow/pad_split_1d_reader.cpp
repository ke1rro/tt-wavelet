#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#define ALWI inline __attribute__((always_inline))

namespace {

constexpr uint32_t kInvalidStick = 0xFFFFFFFFU;

struct StickReadCache {
    uint32_t cb_id;
    uint32_t stick_nbytes;
    uint32_t stick_width;
    uint32_t cached_stick_id;
    bool valid;
};

struct OutputStickWriter {
    uint32_t cb_id;
    uint32_t stick_width;
    uint32_t total_stick_count;
    uint32_t element_count;
    uint32_t sticks_pushed;
    float* ptr;
};

ALWI uint32_t positive_mod(const int32_t value, const uint32_t modulus) {
    if (modulus == 0) {
        return 0;
    }

    const int32_t signed_modulus = static_cast<int32_t>(modulus);
    const int32_t remainder = value % signed_modulus;
    return static_cast<uint32_t>(remainder < 0 ? remainder + signed_modulus : remainder);
}

ALWI uint32_t symmetric_index(const int32_t index, const uint32_t length) {
    if (length <= 1) {
        return 0;
    }

    const uint64_t period = static_cast<uint64_t>(length) * 2;
    const uint32_t reflected = positive_mod(index, static_cast<uint32_t>(period > 0xFFFFFFFFU ? 0xFFFFFFFFU : period));
    return reflected < length ? reflected : static_cast<uint32_t>(period) - 1U - reflected;
}

ALWI uint32_t even_count(const uint32_t padded_length, const uint32_t stick_width) {
    return ((padded_length + 1) / 2 + stick_width - 1) / stick_width;
}
ALWI uint32_t odd_count(const uint32_t padded_length, const uint32_t stick_width) {
    return (padded_length / 2 + stick_width - 1) / stick_width;
}

template <typename SrcAccessor>
ALWI void cache_source_stick(const SrcAccessor& src, StickReadCache& cache, const uint32_t source_stick) {
    if (cache.valid) {
        cb_pop_front(cache.cb_id, 1);
    }

    cb_reserve_back(cache.cb_id, 1);
    const uint32_t cache_l1_addr = get_write_ptr(cache.cb_id);
    const uint64_t src_noc_addr = src.get_noc_addr(source_stick);
    noc_async_read(src_noc_addr, cache_l1_addr, cache.stick_nbytes);
    noc_async_read_barrier();
    cb_push_back(cache.cb_id, 1);
    cb_wait_front(cache.cb_id, 1);

    cache.cached_stick_id = source_stick;
    cache.valid = true;
}

template <typename SrcAccessor>
ALWI float read_padded_symmetric_value(
    const SrcAccessor& src,
    StickReadCache& cache,
    const uint32_t input_length,
    const uint32_t left_pad,
    const uint32_t out_idx) {
    if (input_length == 0) {
        return 0.0F;
    }

    const int32_t logical = static_cast<int32_t>(out_idx) - static_cast<int32_t>(left_pad);
    const uint32_t source_index = symmetric_index(logical, input_length);
    const uint32_t source_stick = source_index / cache.stick_width;
    const uint32_t source_lane = source_index % cache.stick_width;

    if (!cache.valid || source_stick != cache.cached_stick_id) {
        cache_source_stick(src, cache, source_stick);
    }

    const auto* cached_values = reinterpret_cast<const float*>(get_read_ptr(cache.cb_id));
    return cached_values[source_lane];
}

ALWI void release_cache(StickReadCache& cache) {
    if (!cache.valid) {
        return;
    }

    cb_pop_front(cache.cb_id, 1);
    cache.cached_stick_id = kInvalidStick;
    cache.valid = false;
}

ALWI OutputStickWriter
make_output_stick_writer(const uint32_t cb_id, const uint32_t stick_width, const uint32_t total_stick_count) {
    OutputStickWriter writer{cb_id, stick_width, total_stick_count, 0, 0, nullptr};
    if (total_stick_count == 0) {
        return writer;
    }

    cb_reserve_back(cb_id, 1);
    writer.ptr = reinterpret_cast<float*>(get_write_ptr(cb_id));
    return writer;
}

ALWI void push_output_value(OutputStickWriter& writer, const float value) {
    if (writer.ptr == nullptr) {
        return;
    }

    writer.ptr[writer.element_count % writer.stick_width] = value;
    writer.element_count++;

    if (writer.element_count % writer.stick_width != 0) {
        return;
    }

    cb_push_back(writer.cb_id, 1);
    writer.sticks_pushed++;

    // After push_back the current page is published to the consumer, so we must
    // reserve a new page before writing the next stick.
    if (writer.sticks_pushed < writer.total_stick_count) {
        cb_reserve_back(writer.cb_id, 1);
        writer.ptr = reinterpret_cast<float*>(get_write_ptr(writer.cb_id));
    } else {
        writer.ptr = nullptr;
    }
}

ALWI void flush_partial_output_stick(OutputStickWriter& writer) {
    if (writer.ptr == nullptr) {
        return;
    }

    const uint32_t remaining = writer.element_count % writer.stick_width;
    if (remaining == 0) {
        return;
    }

    for (uint32_t i = remaining; i < writer.stick_width; ++i) {
        writer.ptr[i] = 0.0F;
    }
    cb_push_back(writer.cb_id, 1);
    writer.sticks_pushed++;
    writer.ptr = nullptr;
}

}  // namespace

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_length = get_arg_val<uint32_t>(1);
    const uint32_t padded_length = get_arg_val<uint32_t>(2);
    const uint32_t left_pad = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_even = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_odd = get_compile_time_arg_val(1);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_cache = get_compile_time_arg_val(3);
    constexpr uint32_t stick_width = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();
    const auto src = TensorAccessor(src_args, src_addr, stick_nbytes);

    StickReadCache read_cache{cb_id_cache, stick_nbytes, stick_width, kInvalidStick, false};

    const uint32_t even_stick_count = even_count(padded_length, stick_width);
    const uint32_t odd_stick_count = odd_count(padded_length, stick_width);
    const uint32_t pair_count = padded_length / 2;

    auto even_writer = make_output_stick_writer(cb_id_even, stick_width, even_stick_count);
    auto odd_writer = make_output_stick_writer(cb_id_odd, stick_width, odd_stick_count);

    // Stride-2: each iteration produces exactly one even + one odd element
    for (uint32_t pair = 0; pair < pair_count; ++pair) {
        push_output_value(even_writer, read_padded_symmetric_value(src, read_cache, input_length, left_pad, pair * 2));
        push_output_value(
            odd_writer, read_padded_symmetric_value(src, read_cache, input_length, left_pad, pair * 2 + 1));
    }

    // If padded_length is odd, one trailing even element
    if (padded_length & 1U) {
        push_output_value(
            even_writer, read_padded_symmetric_value(src, read_cache, input_length, left_pad, padded_length - 1));
    }

    flush_partial_output_stick(even_writer);
    flush_partial_output_stick(odd_writer);

    release_cache(read_cache);
}
