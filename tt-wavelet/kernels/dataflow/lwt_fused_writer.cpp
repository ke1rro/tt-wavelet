#include <cstdint>

#include "../../tt_wavelet/include/device_protocol/lwt_config.hpp"
#include "api/dataflow/dataflow_api.h"
#include "lwt_tile_row_major_utils.hpp"

namespace row_major = ttwv::kernels::primitives;

namespace {

template <typename ConfigAccessor>
ALWI const uint32_t* load_route_config(
    const ConfigAccessor& config,
    const uint32_t cb_config,
    const uint32_t config_page_bytes,
    const uint32_t route_index) {
    cb_reserve_back(cb_config, 1);
    noc_async_read(config.get_noc_addr(route_index), get_write_ptr(cb_config), config_page_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_config, 1);
    cb_wait_front(cb_config, 1);
    return reinterpret_cast<const uint32_t*>(get_read_ptr(cb_config));
}

}  // namespace

void kernel_main() {
    const uint32_t route_config_addr = get_arg_val<uint32_t>(0);
    const uint32_t route_count = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_config = get_compile_time_arg_val(0);
    constexpr uint32_t config_page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t cb_output = get_compile_time_arg_val(2);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(3);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(4);
    constexpr uint32_t tile_nbytes = get_tile_size(cb_output);
    constexpr uint32_t stick_width = stick_nbytes / sizeof(float);
    constexpr auto config_args = TensorAccessorArgs<5>();
    constexpr auto dst_args = TensorAccessorArgs<config_args.next_compile_time_args_offset()>();

    const auto config = TensorAccessor(config_args, route_config_addr, config_page_bytes);

    for (uint32_t route_index = 0; route_index < route_count; ++route_index) {
        const uint32_t* route = load_route_config(config, cb_config, config_page_bytes, route_index);
        const uint32_t output_addr = route[ttwv::device_protocol::kRouteOutputAddr];
        const uint32_t output_length = route[ttwv::device_protocol::kRouteOutputLength];
        const uint32_t output_group_count = route[ttwv::device_protocol::kRouteOutputGroupCount];

        const auto dst = TensorAccessor(dst_args, output_addr, stick_nbytes);

        for (uint32_t group = 0; group < output_group_count; ++group) {
            cb_wait_front(cb_output, 2);
            const uint32_t output_full_tile = get_read_ptr(cb_output);
            const uint32_t output_tail_tile = output_full_tile + tile_nbytes;
            const uint32_t group_base = group * row_major::kLwtGroupOutputElements;

            for (uint32_t row = 0; row < row_major::kLwtRowsPerGroup; ++row) {
                const uint32_t row_base =
                    group_base + row * row_major::kLwtOutputBlocksPerRow * row_major::kLwtHalfStickElements;
                row_major::write_lwt_half_block(dst, output_full_tile, row, 0, row_base, output_length, stick_width);
                row_major::write_lwt_half_block(
                    dst,
                    output_full_tile,
                    row,
                    row_major::kLwtHalfStickElements,
                    row_base + row_major::kLwtHalfStickElements,
                    output_length,
                    stick_width);
                row_major::write_lwt_half_block(
                    dst,
                    output_tail_tile,
                    row,
                    0,
                    row_base + 2 * row_major::kLwtHalfStickElements,
                    output_length,
                    stick_width);
                noc_async_write_barrier();
            }

            cb_pop_front(cb_output, 2);
        }

        cb_pop_front(cb_config, 1);
        cb_reserve_back(cb_sync, 1);
        cb_push_back(cb_sync, 1);
    }
}
