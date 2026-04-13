#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#define ALWI inline __attribute__((always_inline))

namespace ttwv::kernels::primitives {

struct OutputStickWriter {
    uint32_t cb_id;
    uint32_t stick_width;
    uint32_t total_stick_count;
    uint32_t element_count;
    uint32_t sticks_pushed;
    float* ptr;
};

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

}  // namespace ttwv::kernels::primitives
