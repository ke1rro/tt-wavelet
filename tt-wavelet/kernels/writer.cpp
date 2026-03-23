#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dram_addr_odd = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_in_odd = tt::CBIndex::c_0;

    constexpr uint32_t element_size_bytes = 4;
    constexpr uint32_t elements_per_tile = 1024;
    constexpr uint32_t tile_size_bytes = elements_per_tile * element_size_bytes;

    const InterleavedAddrGen<true> s_odd = {.bank_base_address = dram_addr_odd, .page_size = tile_size_bytes};

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_id_in_odd, 1);

        uint32_t l1_read_addr_odd = get_read_ptr(cb_id_in_odd);

        uint64_t odd_dram_noc_addr = get_noc_addr(i, s_odd);

        noc_async_write(l1_read_addr_odd, odd_dram_noc_addr, tile_size_bytes);
        noc_async_write_barrier();

        cb_pop_front(cb_id_in_odd, 1);
    }
}
