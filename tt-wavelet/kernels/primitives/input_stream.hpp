#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "indexing.hpp"
#define ALWI inline __attribute__((always_inline))

namespace ttwv::kernels::primitives {

constexpr uint32_t kSrcDRAM = 0;
constexpr uint32_t kSrcL1 = 1;

template <typename SrcAccessor>
class InputStream {
public:
    InputStream(const uint32_t cb_id, const SrcAccessor& src, const uint32_t size) :
        _cb_id(cb_id), _src(src), _size(size) {}

    // TODO
    void step(uint32_t step_meta) {
        if (_blob_idx != kBlobInvalid) {
            cb_pop_front(_cb_id, 1);
            _blob_idx = kBlobInvalid;
            _blob_offset = 0;
        }

        _source = new_source;
        _size = new_size;
    }

    void skip(const uint32_t count) {
        if (_size < count) {
            _size = 0;
            return;
        }

        const uint32_t total_offset = _blob_offset + count;
        const uint32_t blobs_to_skip = total_offset / kBlobElements;
        const uint32_t new_blob_offset = total_offset % kBlobElements;

        for (uint32_t i = 0; i < blobs_to_skip; ++i) {
            _next_blob();
        }

        _blob_offset = new_blob_offset;
        _size -= count;
    }

    ALWI float pop() {
        if (_size == 0) {
            return 0.0F;
        }

        if (_blob_idx == kBlobInvalid || _blob_offset >= kBlobElements) {
            _next_blob();
        }

        const auto* blob_data = reinterpret_cast<const float*>(get_read_ptr(_cb_id));

        _size--;
        return blob_data[_blob_offset++];
    }

    ALWI bool empty() const { return _size == 0; }

private:
    uint32_t _cb_id;
    SrcAccessor _src;
    uint32_t _size;
    uint32_t _read_source = kSrcDRAM;
    uint32_t _blob_idx = kBlobInvalid;
    uint32_t _blob_offset = 0;

    ALWI void _next_blob() {
        if (_blob_idx != kBlobInvalid) {
            cb_pop_front(_cb_id, 1);
            _blob_idx++;
        } else {
            _blob_idx = 0;
        }

        if (_read_source == kSrcDRAM) {
            // Retrieve from the DRAM
            cb_reserve_back(_cb_id, 1);
            const uint32_t l1_addr = get_write_ptr(_cb_id);
            const uint64_t noc_addr = _src.get_noc_addr(_blob_idx);
            noc_async_read(noc_addr, l1_addr, kBlobBytes);
            noc_async_read_barrier();
            cb_push_back(_cb_id, 1);
        }  // else, assume writer has written the blob

        cb_wait_front(_cb_id, 1);
        _blob_offset = 0;
    }
};

}  // namespace ttwv::kernels::primitives
