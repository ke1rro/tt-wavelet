#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dram_addr_odd = get_arg_val<uint32_t>(0); // dram_out_odd (для дампу бази)
    uint32_t dram_addr_even = get_arg_val<uint32_t>(1); // dram_out_even (для дампу тапів)
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t POLICY_LEN = get_arg_val<uint32_t>(3);
    uint32_t num_steps = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in_odd = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in_even = tt::CBIndex::c_1;

    constexpr uint32_t element_size_bytes = 4;
    constexpr uint32_t tile_size_bytes = 1024 * element_size_bytes;

    const InterleavedAddrGen<true> s_out_odd = {.bank_base_address = dram_addr_odd, .page_size = tile_size_bytes};
    const InterleavedAddrGen<true> s_out_even = {.bank_base_address = dram_addr_even, .page_size = tile_size_bytes};

    uint32_t arg_idx = 5;
    
    // Глобальні індекси для запису в масив-дамп
    uint32_t odd_write_idx = 0;
    uint32_t even_write_idx = 0;

    for (uint32_t step = 0; step < num_steps; step++) {
        uint32_t op_type = get_arg_val<uint32_t>(arg_idx++);
        uint32_t filter_len = get_arg_val<uint32_t>(arg_idx++);
        int32_t start_shift = (int32_t)get_arg_val<uint32_t>(arg_idx++);

        // Дзеркально до Reader-а:
        uint32_t cb_base = (op_type == 0) ? cb_id_in_odd : cb_id_in_even;
        uint32_t cb_taps = (op_type == 0) ? cb_id_in_even : cb_id_in_odd;

        for (uint32_t i = 0; i < num_tiles; i++) {
            
            if (filter_len <= POLICY_LEN) {
                cb_wait_front(cb_base, 1);
                cb_wait_front(cb_taps, filter_len);
                
                // Дамп бази
                uint64_t base_noc_addr = get_noc_addr(odd_write_idx++, s_out_odd);
                noc_async_write(get_read_ptr(cb_base), base_noc_addr, tile_size_bytes);
                
                // Дамп тапів
                for(uint32_t f=0; f<filter_len; f++) {
                    uint64_t tap_noc_addr = get_noc_addr(even_write_idx++, s_out_even);
                    noc_async_write(get_read_ptr(cb_taps) + f * tile_size_bytes, tap_noc_addr, tile_size_bytes);
                }
                noc_async_write_barrier();

                cb_pop_front(cb_base, 1);
                cb_pop_front(cb_taps, filter_len);
            }
            else {
                uint32_t taps_remaining = filter_len;
                uint32_t tap_offset = 0;
                
                while (taps_remaining > 0) {
                    uint32_t taps_this_pass = (taps_remaining > POLICY_LEN) ? POLICY_LEN : taps_remaining;
                    
                    if (tap_offset == 0) {
                        cb_wait_front(cb_base, 1);
                        cb_wait_front(cb_taps, taps_this_pass);
                        
                        // Дамп бази
                        uint64_t base_noc_addr = get_noc_addr(odd_write_idx++, s_out_odd);
                        noc_async_write(get_read_ptr(cb_base), base_noc_addr, tile_size_bytes);
                        
                        // Дамп тапів
                        for(uint32_t f=0; f<taps_this_pass; f++) {
                            uint64_t tap_noc_addr = get_noc_addr(even_write_idx++, s_out_even);
                            noc_async_write(get_read_ptr(cb_taps) + f * tile_size_bytes, tap_noc_addr, tile_size_bytes);
                        }
                        noc_async_write_barrier();

                        cb_pop_front(cb_base, 1);
                        cb_pop_front(cb_taps, taps_this_pass);
                    }
                    else {
                        cb_wait_front(cb_taps, taps_this_pass);
                        
                        // Дамп тапів (бази немає)
                        for(uint32_t f=0; f<taps_this_pass; f++) {
                            uint64_t tap_noc_addr = get_noc_addr(even_write_idx++, s_out_even);
                            noc_async_write(get_read_ptr(cb_taps) + f * tile_size_bytes, tap_noc_addr, tile_size_bytes);
                        }
                        noc_async_write_barrier();

                        cb_pop_front(cb_taps, taps_this_pass);
                    }
                    
                    tap_offset += taps_this_pass;
                    taps_remaining -= taps_this_pass;
                }
            }
        }
        
        // TODO: Синхронізація семафорів між кроками, якщо Writer пише результат кроку в DRAM як вхідні дані для наступного!
        // В рамках цього тесту ми просто дампмо все лінійно, тому пропускаємо.
    }
}
