// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#ifndef ALWI
#define ALWI inline __attribute__((always_inline))
#endif

namespace ttwv::kernels::primitives {

template <typename Accessor>
ALWI void load_config_page(
    const Accessor& accessor,
    const uint32_t address,
    const uint32_t page_bytes,
    const uint32_t page_index,
    const uint32_t cb,
    uint32_t* words,
    const uint32_t word_count) {
    const auto pages = TensorAccessor(accessor, address, page_bytes);
    cb_reserve_back(cb, 1);
    noc_async_read(pages.get_noc_addr(page_index), get_write_ptr(cb), page_bytes);
    noc_async_read_barrier();
    cb_push_back(cb, 1);
    cb_wait_front(cb, 1);
    const auto* loaded = reinterpret_cast<const uint32_t*>(get_read_ptr(cb));
    for (uint32_t word = 0; word < word_count; ++word) {
        words[word] = loaded[word];
    }
    cb_pop_front(cb, 1);
}

}  // namespace ttwv::kernels::primitives
