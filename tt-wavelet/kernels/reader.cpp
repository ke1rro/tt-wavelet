#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dram_addr_odd = get_arg_val<uint32_t>(0);
    uint32_t dram_addr_even = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t halo_left = get_arg_val<uint32_t>(3);
    uint32_t halo_right = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in_odd = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in_even = tt::CBIndex::c_1;

    constexpr uint32_t element_size_bytes = 4;
    constexpr uint32_t elements_per_tile = 1024;
    constexpr uint32_t tile_size_bytes = elements_per_tile * element_size_bytes;

    // https://deepwiki.com/search/interleavedaddrgen_03f4223c-697c-462e-98de-97585fe16d27?mode=fast
    const InterleavedAddrGen<true> s_odd = {.bank_base_address = dram_addr_odd, .page_size = tile_size_bytes};

    const InterleavedAddrGen<true> s_even = {.bank_base_address = dram_addr_even, .page_size = tile_size_bytes};

    // ---------------------------------------------------------------------------------------------------------------------------------------

    uint32_t step_stride = elements_per_tile;
    if (halo_left > 0 || halo_right > 0) {
        uint32_t halo_size = halo_left + halo_right;
        if (halo_size < elements_per_tile) {
            step_stride = elements_per_tile - halo_size;
        }
    }

    uint32_t current_element_idx = 0;

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_id_in_odd, 1);
        cb_reserve_back(cb_id_in_even, 1);

        uint32_t l1_write_addr_odd = get_write_ptr(cb_id_in_odd);
        uint32_t l1_write_addr_even = get_write_ptr(cb_id_in_even);

        if (step_stride == elements_per_tile) {
            uint64_t odd_dram_noc_addr = get_noc_addr(i, s_odd);
            uint64_t even_dram_noc_addr = get_noc_addr(i, s_even);

            noc_async_read(odd_dram_noc_addr, l1_write_addr_odd, tile_size_bytes);
            noc_async_read(even_dram_noc_addr, l1_write_addr_even, tile_size_bytes);

        } else {
            uint32_t start_page_idx = current_element_idx / elements_per_tile;
            uint32_t offset = current_element_idx % elements_per_tile;

            uint32_t elements_from_first = elements_per_tile - offset;
            uint32_t elements_from_second = elements_per_tile - elements_from_first;

            uint32_t bytes_from_first = elements_from_first * element_size_bytes;
            uint32_t bytes_from_second = elements_from_second * element_size_bytes;

            uint64_t odd_noc_addr_1 = get_noc_addr(start_page_idx, s_odd) + (offset * element_size_bytes);
            noc_async_read(odd_noc_addr_1, l1_write_addr_odd, bytes_from_first);

            if (elements_from_second > 0) {
                uint64_t odd_noc_addr_2 = get_noc_addr(start_page_idx + 1, s_odd);
                noc_async_read(odd_noc_addr_2, l1_write_addr_odd + bytes_from_first, bytes_from_second);
            }

            uint64_t even_noc_addr_1 = get_noc_addr(start_page_idx, s_even) + (offset * element_size_bytes);
            noc_async_read(even_noc_addr_1, l1_write_addr_even, bytes_from_first);

            if (elements_from_second > 0) {
                uint64_t even_noc_addr_2 = get_noc_addr(start_page_idx + 1, s_even);
                noc_async_read(even_noc_addr_2, l1_write_addr_even + bytes_from_first, bytes_from_second);
            }
        }

        noc_async_read_barrier();

        cb_push_back(cb_id_in_odd, 1);
        cb_push_back(cb_id_in_even, 1);

        current_element_idx += step_stride;
    }
}