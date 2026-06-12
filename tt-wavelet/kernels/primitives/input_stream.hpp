#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "indexing.hpp"
#define ALWI inline __attribute__((always_inline))

namespace ttwv::kernels::primitives {

// Exactly 3 bits for state
constexpr uint32_t kStateStart = 0;
constexpr uint32_t kStatePeek = 1;
constexpr uint32_t kStatePop = 2;
constexpr uint32_t kStateFetchL1 = 3;
constexpr uint32_t kStatePopL1 = 4;
constexpr uint32_t kStateFetchDRAM = 5;
constexpr uint32_t kStatePopDRAM = 6;
constexpr uint32_t kStateDone = 7;

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
        switch (m_state) {
            case kStateStart:
            case kStateDone:
                break;
            case kStateFetchDRAM:
                if (count <= m_data_size - (m_cursor - m_data_start)) {
                    m_cursor += count;
                } else {
                    m_cursor = m_data_start + m_data_size; // Move to end of current blob
                    if (m_state == kStateFetchL1) {
                        m_state = kStateFetchDRAM;
                    } else {
                        m_state = kStatePopDRAM;
                    }
                }
                break;
            case kStatePopDRAM:
                if (count <= m_data_size - (m_cursor - m_data_start)) {
                    m_cursor += count;
                } else {
                    // Skip remaining data in current blob and move to next blob
                    const uint32_t remaining_in_blob = m_data_size - (m_cursor - m_data_start);
                    m_cursor += remaining_in_blob; // Move to end of current blob
                    if (m_state == kStatePopL1) {
                        m_state = kStatePopDRAM;
                    } else {
                        // Already in DRAM, just move to next blob
                        // No state change needed
                    }
                }
                break;
            case kStateDone:
                // No more data to skip
                break;
        }
    }

    ALWI float pop() {
        if (m_data_cursor >= m_data_size || m_state == kStateStart || m_state == kStateDone) {
            return 0.0F;
        }

        // Not available - fetch
        if (m_data_cursor >= m_data_start + m_data_size) {
            if (m_state == kStatePeek || m_state == kStatePop) {
                return 0.0F; // No more data available
            }

            fetch_blob();
        }

        float value = m_data[m_data_cursor++];

        if (m_data_cursor >= m_data_start + kBlobElements) {
            if (m_state == kStatePeek || m_state == kStateFetchL1 || m_state == kStateFetchDRAM) {
                return value;
            }

            pop_blob();
        }
    }

    ALWI bool empty() const { return m_cursor == m_size; }

private:
    uint32_t m_cb_pipe;
    uint32_t m_cb_cache;
    SrcAccessor m_accessor;
    uint32_t m_size;
    // uin32_t m_blobs() const {
    //     return (m_size + kBlobElements - 1) / kBlobElements;
    // }
    uint32_t m_blobs;

    uint32_t m_state = kStateStart;

    uint32_t m_cursor = 0;
    const float* m_data = nullptr;
    uint32_t m_data_start = 0; // In blobs
    uint32_t m_data_size = 0;


    // Call only if m_state is kStatePopL1, kStatePopDRAM, kStateFetchL1 or kStateFetchDRAM
    ALWI bool fetch_blob() {
        const uint32_t next_blob = m_data_start + m_data_size;
        // Last blob already fetched
        if (next_blob >= m_blobs) {
            return false;
        }

        const uint32_t cb = (m_state == kStatePopDRAM) ? m_cb_cache : m_cb_pipe;

        if(m_state == kStatePopDRAM || m_state == kStateFetchDRAM) {
            // DRAM -> CB_PIPE/CB_CACHE
            cb_reserve_back(cb, 1);
            const uint32_t l1_addr = get_write_ptr(cb);
            const uint64_t noc_addr = m_accessor.get_noc_addr(m_data_start + m_data_size);
            noc_async_read(noc_addr, l1_addr, kBlobBytes);
            noc_async_read_barrier();
            cb_push_back(cb, 1);
        } // else: L1 -> CB_PIPE

        cb_wait_front(cb, 1);
        m_data_size++;
        return true;
    }

    // Call only if m_state is kStatePop, kStatePopL1 or kStatePopDRAM
    ALWI void pop_blob() {
        if (m_data_size == 0) {
            return;
        }

        const uint32_t cb = (m_state == kStatePopDRAM) ? m_cb_cache : m_cb_pipe;
        cb_pop_front(cb, 1);
        m_data_start += 1;
        m_data_size -= 1;

        if (m_data_size > 0) {
            m_data = reinterpret_cast<const float*>(get_read_ptr(cb));
        } else {
            m_data = nullptr;
        }
    }
};

}  // namespace ttwv::kernels::primitives
