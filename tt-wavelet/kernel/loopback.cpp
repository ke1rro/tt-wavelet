

#include <cstdint>

#include "../../tt-metal/tt_metal/api/tt-metalium/tensor_accessor_args.hpp"
#include "../../tt-metal/tt_metal/hw/inc/api/dataflow/dataflow_api.h"
#include "../../tt-metal/tt_metal/hw/inc/api/tensor/tensor_accessor.h"
#include "../../tt-metal/tt_metal/hw/inc/api/tensor/tensor_accessor_args.h"
#include "../../tt-metal/tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h"

void kernel_main() {
    uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t dram_buffer_src_addr = get_arg_val<uint32_t>(1);
    uint32_t dram_buffer_dst_addr = get_arg_val<uint32_t>(2);
    std::uint32_t num_tiles = get_arg_val<uint32_t>(3);

    const uint32_t tile_size_bytes = 32 * 32 * 2;
    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, dram_buffer_src_addr, tile_size_bytes);

    constexpr auto out0_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out0 = TensorAccessor(out0_args, dram_buffer_dst_addr, tile_size_bytes);

    for (uint32_t i{0}; i < num_tiles; ++i) {
        noc_async_read(i, in0, l1_buffer_addr);
        noc_async_read_barrier();

        noc_async_write(i, out0, l1_buffer_addr);
        noc_async_write_barrier();
    }

}
