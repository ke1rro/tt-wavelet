#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#define ALWI inline __attribute__((always_inline))

namespace ttwv::kernels::primitives {

ALWI void fill_zeros(float* ptr, const uint32_t n) { __builtin_memset(ptr, 0, n * sizeof(float)); }

ALWI void push_zero_stick(const uint32_t cb_id, const uint32_t stick_nbytes) {
    cb_reserve_back(cb_id, 1);

    uint32_t write_addr = get_write_ptr(cb_id);
    uint32_t bytes_left = stick_nbytes;
    const uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);

    while (bytes_left > 0) {
        const uint32_t read_size = bytes_left > MEM_ZEROS_SIZE ? MEM_ZEROS_SIZE : bytes_left;
        noc_async_read(zeros_noc_addr, write_addr, read_size);
        write_addr += read_size;
        bytes_left -= read_size;
    }

    noc_async_read_barrier();
    cb_push_back(cb_id, 1);
}

ALWI void push_zero_sticks(const uint32_t cb_id, const uint32_t stick_nbytes, const uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        push_zero_stick(cb_id, stick_nbytes);
    }
}

}  // namespace ttwv::kernels::primitives
