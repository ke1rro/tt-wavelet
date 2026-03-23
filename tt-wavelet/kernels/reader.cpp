#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

inline void read_shifted_tile(
    int32_t global_element_idx,
    uint32_t l1_write_addr,
    const InterleavedAddrGen<true>& s,
    uint32_t num_tiles,
    uint32_t elements_per_tile,
    uint32_t element_size_bytes) 
{
    int32_t max_idx = (int32_t)(num_tiles * elements_per_tile) - (int32_t)elements_per_tile;
    
    // symmetric padding for now
    if (global_element_idx < 0) {
        global_element_idx = 0;
    } else if (global_element_idx > max_idx) {
        global_element_idx = max_idx;
    }

    uint32_t start_page_idx = (uint32_t)global_element_idx / elements_per_tile;
    uint32_t offset = (uint32_t)global_element_idx % elements_per_tile;

    if (offset == 0) {
        uint64_t noc_addr = get_noc_addr(start_page_idx, s);
        noc_async_read(noc_addr, l1_write_addr, elements_per_tile * element_size_bytes);
    } else {
        uint32_t elements_from_first = elements_per_tile - offset;
        uint32_t elements_from_second = offset;
        // here reader needs to wait until wtiter give results
        uint32_t bytes_from_first = elements_from_first * element_size_bytes;
        uint32_t bytes_from_second = elements_from_second * element_size_bytes;

        uint64_t noc_addr_1 = get_noc_addr(start_page_idx, s) + (offset * element_size_bytes);
        noc_async_read(noc_addr_1, l1_write_addr, bytes_from_first);

        uint64_t noc_addr_2 = get_noc_addr(start_page_idx + 1, s);
        noc_async_read(noc_addr_2, l1_write_addr + bytes_from_first, bytes_from_second);
    }
}

void kernel_main() {
    uint32_t dram_addr_odd = get_arg_val<uint32_t>(0);
    uint32_t dram_addr_even = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t POLICY_LEN = get_arg_val<uint32_t>(3);
    uint32_t num_steps = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in_odd = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in_even = tt::CBIndex::c_1;

    constexpr uint32_t element_size_bytes = 4;
    constexpr uint32_t elements_per_tile = 1024;
    constexpr uint32_t tile_size_bytes = elements_per_tile * element_size_bytes;

    const InterleavedAddrGen<true> s_odd = {.bank_base_address = dram_addr_odd, .page_size = tile_size_bytes};
    const InterleavedAddrGen<true> s_even = {.bank_base_address = dram_addr_even, .page_size = tile_size_bytes};

    uint32_t arg_idx = 5;
    
    for (uint32_t step = 0; step < num_steps; step++) {
        uint32_t op_type = get_arg_val<uint32_t>(arg_idx++);
        uint32_t filter_len = get_arg_val<uint32_t>(arg_idx++);
        int32_t start_shift = (int32_t)get_arg_val<uint32_t>(arg_idx++);

        uint32_t cb_base = (op_type == 0) ? cb_id_in_odd : cb_id_in_even;
        uint32_t cb_taps = (op_type == 0) ? cb_id_in_even : cb_id_in_odd;
        
        const auto& s_base = (op_type == 0) ? s_odd : s_even;
        const auto& s_taps = (op_type == 0) ? s_even : s_odd;

        for (uint32_t i = 0; i < num_tiles; i++) {
            
            int32_t base_element_idx = (int32_t)(i * elements_per_tile);
            
            // SINGLE PASS
            if (filter_len <= POLICY_LEN) {
                cb_reserve_back(cb_base, 1);
                cb_reserve_back(cb_taps, filter_len);
                
                uint32_t l1_write_addr_base = get_write_ptr(cb_base);
                uint32_t l1_write_addr_taps = get_write_ptr(cb_taps);
                
                read_shifted_tile(base_element_idx, l1_write_addr_base, s_base, num_tiles, elements_per_tile, element_size_bytes);
                
                for (uint32_t f = 0; f < filter_len; f++) {
                    int32_t target_element_idx = base_element_idx + start_shift + (int32_t)f;
                    read_shifted_tile(target_element_idx, l1_write_addr_taps + (f * tile_size_bytes), s_taps, num_tiles, elements_per_tile, element_size_bytes);
                }

                noc_async_read_barrier();
                cb_push_back(cb_base, 1);
                cb_push_back(cb_taps, filter_len);
            }
            else {
                // MULTI PASS
                uint32_t taps_remaining = filter_len;
                uint32_t tap_offset = 0;
                
                while (taps_remaining > 0) {
                    uint32_t taps_this_pass = (taps_remaining > POLICY_LEN) ? POLICY_LEN : taps_remaining;
                    
                    if (tap_offset == 0) {
                        cb_reserve_back(cb_base, 1);
                        cb_reserve_back(cb_taps, taps_this_pass);
                        
                        uint32_t l1_write_addr_base = get_write_ptr(cb_base);
                        uint32_t l1_write_addr_taps = get_write_ptr(cb_taps);
                        
                        read_shifted_tile(base_element_idx, l1_write_addr_base, s_base, num_tiles, elements_per_tile, element_size_bytes);
                        
                        for (uint32_t f = 0; f < taps_this_pass; f++) {
                            int32_t target_element_idx = base_element_idx + start_shift + (int32_t)tap_offset + (int32_t)f;
                            read_shifted_tile(target_element_idx, l1_write_addr_taps + (f * tile_size_bytes), s_taps, num_tiles, elements_per_tile, element_size_bytes);
                        }
                        
                        noc_async_read_barrier();
                        cb_push_back(cb_base, 1);
                        cb_push_back(cb_taps, taps_this_pass);
                    }
                    else {
                        cb_reserve_back(cb_taps, taps_this_pass);
                        uint32_t l1_write_addr_taps = get_write_ptr(cb_taps);
                        
                        for (uint32_t f = 0; f < taps_this_pass; f++) {
                            int32_t target_element_idx = base_element_idx + start_shift + (int32_t)tap_offset + (int32_t)f;
                            read_shifted_tile(target_element_idx, l1_write_addr_taps + (f * tile_size_bytes), s_taps, num_tiles, elements_per_tile, element_size_bytes);
                        }
                        
                        noc_async_read_barrier();
                        cb_push_back(cb_taps, taps_this_pass);
                    }
                    
                    tap_offset += taps_this_pass;
                    taps_remaining -= taps_this_pass;
                }
            }
        }
        
        // TODO: synchronization with writer
    }
}