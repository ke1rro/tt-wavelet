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

ALWI void route_barrier(
    const uint32_t barrier_semaphore_id,
    const uint32_t core_index,
    const uint32_t active_core_count,
    const uint32_t* core_noc_coords) {
    if (active_core_count <= 1) {
        return;
    }

    const uint32_t semaphore_addr = get_semaphore(barrier_semaphore_id);
    auto* local_semaphore = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);

    if (core_index == 0) {
        noc_semaphore_wait(local_semaphore, active_core_count - 1);
        noc_semaphore_set(local_semaphore, 0);

        for (uint32_t remote_core = 1; remote_core < active_core_count; ++remote_core) {
            const uint32_t noc_x = core_noc_coords[remote_core * 2];
            const uint32_t noc_y = core_noc_coords[remote_core * 2 + 1];
            noc_semaphore_inc(get_noc_addr(noc_x, noc_y, semaphore_addr), 1);
        }
    } else {
        const uint32_t master_noc_x = core_noc_coords[0];
        const uint32_t master_noc_y = core_noc_coords[1];
        noc_semaphore_inc(get_noc_addr(master_noc_x, master_noc_y, semaphore_addr), 1);
        noc_semaphore_wait(local_semaphore, 1);
        noc_semaphore_set(local_semaphore, 0);
    }
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
    constexpr uint32_t route_barrier_semaphore_id = get_compile_time_arg_val(5);
    constexpr uint32_t tile_nbytes = get_tile_size(cb_output);
    constexpr uint32_t stick_width = stick_nbytes / sizeof(float);
    constexpr auto config_args = TensorAccessorArgs<6>();
    constexpr auto dst_args = TensorAccessorArgs<config_args.next_compile_time_args_offset()>();

    const auto config = TensorAccessor(config_args, route_config_addr, config_page_bytes);
    const uint32_t barrier_args_base = 2 + route_count * 2;
    const uint32_t core_index = get_arg_val<uint32_t>(barrier_args_base);
    const uint32_t active_core_count = get_arg_val<uint32_t>(barrier_args_base + 1);
    const auto* core_noc_coords = reinterpret_cast<const uint32_t*>(get_arg_addr(barrier_args_base + 2));

    for (uint32_t route_index = 0; route_index < route_count; ++route_index) {
        const uint32_t* route = load_route_config(config, cb_config, config_page_bytes, route_index);
        const uint32_t output_addr = route[ttwv::device_protocol::kRouteOutputAddr];
        const uint32_t output_length = route[ttwv::device_protocol::kRouteOutputLength];
        const uint32_t route_range_arg_base = 2 + route_index * 2;
        const uint32_t global_group_begin = get_arg_val<uint32_t>(route_range_arg_base);
        const uint32_t local_group_count = get_arg_val<uint32_t>(route_range_arg_base + 1);

        const auto dst = TensorAccessor(dst_args, output_addr, stick_nbytes);

        for (uint32_t local_group = 0; local_group < local_group_count; ++local_group) {
            cb_wait_front(cb_output, 2);
            const uint32_t output_full_tile = get_read_ptr(cb_output);
            const uint32_t output_tail_tile = output_full_tile + tile_nbytes;
            const uint32_t global_group = global_group_begin + local_group;
            const uint32_t group_base = global_group * row_major::kLwtGroupOutputElements;

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
            }

            cb_pop_front(cb_output, 2);
        }

        noc_async_write_barrier();
        cb_pop_front(cb_config, 1);

        const bool has_next_route = route_index + 1 < route_count;
        if (has_next_route) {
            route_barrier(route_barrier_semaphore_id, core_index, active_core_count, core_noc_coords);
            cb_reserve_back(cb_sync, 1);
            cb_push_back(cb_sync, 1);
        }
    }
}
