#pragma once

#include <cstdint>

#include "../utils/boundary.hpp"
#include "../utils/fill.hpp"
#include "../utils/stick_read_cache.hpp"

namespace ttwv::kernels::utils {

struct LwtReaderConfig {
    uint32_t input_length;
    uint32_t padded_length;
    uint32_t left_pad;
    uint32_t split_phase;
    int32_t source_offset;
    uint32_t stencil_k;
    bool is_pad_split;
    uint32_t num_tiles;
};

template <typename SrcAccessor>
struct LwtTileGenerator {
    const SrcAccessor& src;
    StickReadCache& cache;
    const LwtReaderConfig config;
    const uint32_t cb_halo;
    const uint32_t cb_cur;
    const uint32_t stick_nbytes;
    const uint32_t stick_width;

    ALWI float read_stream_value(const int32_t split_index) const {
        if (config.is_pad_split && config.input_length == 0) {
            return 0.0F;
        }

        const uint32_t logical_split_length = split_length(config.padded_length, config.split_phase);
        if (split_index < 0 || static_cast<uint32_t>(split_index) >= logical_split_length) {
            return 0.0F;
        }

        if (config.is_pad_split) {
            const uint32_t padded_index = static_cast<uint32_t>(split_index) * 2U + config.split_phase;
            const int32_t source_logical_index = static_cast<int32_t>(padded_index) - static_cast<int32_t>(config.left_pad);
            const uint32_t source_index = symmetric_index(source_logical_index, config.input_length);
            return read_source_value(src, cache, source_index);
        } else {
            return read_source_value(src, cache, split_index);
        }
    }

    ALWI void push_stream_stick(const uint32_t cb_id, const int32_t logical_base_index) const {
        cb_reserve_back(cb_id, 1);
        auto* ptr = reinterpret_cast<float*>(get_write_ptr(cb_id));
        fill_zeros(ptr, stick_width);

        for (uint32_t lane = 0; lane < stick_width; ++lane) {
            const int32_t split_index = logical_base_index + static_cast<int32_t>(lane) + config.source_offset;
            ptr[lane] = read_stream_value(split_index);
        }

        cb_push_back(cb_id, 1);
    }

    ALWI void push_tile_pair(const uint32_t tile_index) const {
        const int32_t current_base = static_cast<int32_t>(tile_index * stick_width) - stencil_left_pad(config.stencil_k);
        const int32_t halo_base = current_base - static_cast<int32_t>(stick_width);

        push_stream_stick(cb_halo, halo_base);
        push_zero_sticks(cb_halo, stick_nbytes, 31);

        push_stream_stick(cb_cur, current_base);
        push_zero_sticks(cb_cur, stick_nbytes, 31);
    }
};

template <typename SrcAccessor>
ALWI LwtTileGenerator<SrcAccessor> make_lwt_tile_generator(
    const SrcAccessor& src,
    StickReadCache& cache,
    const LwtReaderConfig& config,
    const uint32_t cb_halo,
    const uint32_t cb_cur,
    const uint32_t stick_nbytes,
    const uint32_t stick_width) {
    return {src, cache, config, cb_halo, cb_cur, stick_nbytes, stick_width};
}

}  // namespace ttwv::kernels::utils
