#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "lwt_reader_utils.hpp"

namespace ku = ttwv::kernels::utils;

// Fused reader for the 1D lifting pipeline.
//
// Runtime args:
//   0: src_addr       - base address of the original input signal
//   1: input_length   - number of samples in the original signal
//   2: padded_length  - physical wavelet-padded length before split
//   3: left_pad       - number of symmetric samples prepended before split
//   4: split_phase    - 0 for even stream, 1 for odd stream
//   5: source_offset  - step-dependent shift for the stencil source stream
//   6: stencil_k      - number of stencil coefficients
//   7: num_tiles      - number of halo/current tile pairs to emit
//
// Compile-time args:
//   0: cb_halo
//   1: cb_cur
//   2: stick_nbytes
//   3: cb_cache
//   4: stick_width
//
// The reader performs three stages at once:
//   1. symmetric wavelet padding over the original signal
//   2. even/odd split into a finite logical stream
//   3. physical zero padding for the stencil contract (17 - K)
//
// The emitted layout matches the current compute prototype:
// only row 0 carries signal data; rows 1..31 are zero-filled.
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    
    ku::LwtReaderConfig config;
    config.input_length = get_arg_val<uint32_t>(1);
    config.padded_length = get_arg_val<uint32_t>(2);
    config.left_pad = get_arg_val<uint32_t>(3);
    config.split_phase = get_arg_val<uint32_t>(4) & 1U;
    config.source_offset = get_arg_val<int32_t>(5);
    config.stencil_k = get_arg_val<uint32_t>(6);
    config.num_tiles = get_arg_val<uint32_t>(7);
    config.is_pad_split = get_arg_val<uint32_t>(8) == 1;

    constexpr uint32_t cb_halo = get_compile_time_arg_val(0);
    constexpr uint32_t cb_cur = get_compile_time_arg_val(1);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb_cache = get_compile_time_arg_val(3);
    constexpr uint32_t stick_width = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();
    const auto src = TensorAccessor(src_args, src_addr, stick_nbytes);

    ku::StickReadCache read_cache{cb_cache, stick_nbytes, stick_width, ku::kInvalidStick, false};

    auto generator = ku::make_lwt_tile_generator(
        src, read_cache, config, cb_halo, cb_cur, stick_nbytes, stick_width);

    for (uint32_t tile = 0; tile < config.num_tiles; ++tile) {
        generator.push_tile_pair(tile);
    }

    ku::release_cache(read_cache);
}
