#include <cstdint>

#include "../primitives/splice_chain.hpp"
#include "api/dataflow/dataflow_api.h"

namespace splice = ttwv::kernels::primitives::splice;

void kernel_main() {
    const uint32_t splice_number = get_arg_val<uint32_t>(0);
    const uint32_t even_addr = get_arg_val<uint32_t>(1);
    const uint32_t odd_addr = get_arg_val<uint32_t>(2);
    const uint32_t even_length = get_arg_val<uint32_t>(3);
    const uint32_t odd_length = get_arg_val<uint32_t>(4);
    const uint32_t source_start = get_arg_val<uint32_t>(5);
    const uint32_t source_length = get_arg_val<uint32_t>(6);
    const uint32_t source_prefix = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_even = get_named_compile_time_arg_val("cb_even");
    constexpr uint32_t cb_odd = get_named_compile_time_arg_val("cb_odd");
    constexpr uint32_t cb_even_cache = get_named_compile_time_arg_val("cb_even_cache");
    constexpr uint32_t cb_odd_cache = get_named_compile_time_arg_val("cb_odd_cache");
    constexpr uint32_t requested_cache_bytes = get_named_compile_time_arg_val("cache_bytes");
    constexpr uint32_t cache_bytes = requested_cache_bytes == 0 ? splice::kDefaultCacheBytes : requested_cache_bytes;

    constexpr uint32_t tile_nbytes = get_tile_size(cb_even);
    constexpr auto even_accessor_args = TensorAccessorArgs<0>();
    constexpr auto odd_accessor_args = TensorAccessorArgs<even_accessor_args.next_compile_time_args_offset()>();
    const auto even_accessor = TensorAccessor(even_accessor_args, even_addr, cache_bytes);
    const auto odd_accessor = TensorAccessor(odd_accessor_args, odd_addr, cache_bytes);

    cb_reserve_back(cb_even_cache, 1);
    cb_push_back(cb_even_cache, 1);
    cb_wait_front(cb_even_cache, 1);
    cb_reserve_back(cb_odd_cache, 1);
    cb_push_back(cb_odd_cache, 1);
    cb_wait_front(cb_odd_cache, 1);

    splice::StreamReadState even_state{
        even_length, source_start, source_length, source_prefix, get_read_ptr(cb_even_cache), cache_bytes, 0, 0, false};
    splice::StreamReadState odd_state{
        odd_length, source_start, source_length, source_prefix, get_read_ptr(cb_odd_cache), cache_bytes, 0, 0, false};

    splice::build_splice_chain(cb_even, even_accessor, even_state, splice_number, tile_nbytes);
    splice::build_splice_chain(cb_odd, odd_accessor, odd_state, splice_number, tile_nbytes);

    cb_pop_front(cb_even_cache, 1);
    cb_pop_front(cb_odd_cache, 1);
}
