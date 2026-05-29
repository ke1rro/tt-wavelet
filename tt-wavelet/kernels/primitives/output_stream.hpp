#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "blob_common.hpp"
#define ALWI inline __attribute__((always_inline))

namespace ttwv::kernels::primitives {

static_assert(kBlobElements % 16 == 0, "Blob elements must be a multiple of 16 for noc operations.");

constexpr uint32_t kDstL1 = 0;
constexpr uint32_t kDstDRAM = 1;

template <typename DstAccessor>
class OutputStream {
public:
    OutputStream(const uint32_t cb_id, const DstAccessor& dst) : _cb_id(cb_id), _dst(dst) {}

    ALWI void flush() {
        if (_blob_idx != kBlobInvalid) {
            _flush_blob();
            _blob_idx = kBlobInvalid;
        }
    }

    ALWI void push(float value) {
        if (_blob_idx != kBlobInvalid && _blob_offset >= kBlobElements) {
            _flush_blob();
            cb_reserve_back(_cb_id, 1);
        } else if (_blob_idx == kBlobInvalid) {
            cb_reserve_back(_cb_id, 1);
            _blob_idx = 0;
        }

        const auto* blob_data = reinterpret_cast<const float*>(get_write_ptr(_cb_id));

        blob_data[_blob_offset++] = value;
    }

    ALWI void push16(const float* values) {
        for (uint32_t i = 0; i < 16; ++i) {
            push(values[i]);
        }
    }

    ALWI void set_write_destination(uint32_t destination) {
        flush();
        _write_destination = destination;
    }

private:
    uint32_t _cb_id;
    DstAccessor _dst;
    uint32_t _write_destination = kDstL1;
    uint32_t _blob_idx = kBlobInvalid;
    uint32_t _blob_offset = 0;

    // DO NOT call this when blob_idx == kBlobInvalid
    ALWI void _flush_blob() {
        if (_blob_offset == 0) {
            _blob_idx++;
            return;
        }

        cb_push_back(_cb_id, 1);

        if (_write_destination == kDstDRAM) {
            // Write to the DRAM
            cb_wait_front(_cb_id, 1);
            const uint32_t l1_addr = get_read_ptr(_cb_id);
            const uint64_t noc_addr = _dst.get_noc_addr(_blob_idx);
            noc_async_write(noc_addr, l1_addr, _blob_offset * sizeof(float));
            noc_async_write_barrier();
            cb_pop_front(_cb_id, 1);
        }

        _blob_idx++;
        _blob_offset = 0;
    }
};

}  // namespace ttwv::kernels::primitives
