// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/wavelet/common/wavelet_host.hpp"

#include <cstddef>
#include <string>

#include <tt_stl/assert.hpp>

#include "ttnn/operations/wavelet/common/boundary_parse.hpp"

namespace ttnn::operations::wavelet {

BoundaryMode boundary_mode_from_string(const std::string_view name) {
    BoundaryMode mode{};
    TT_FATAL(
        parse_boundary_mode(name, mode),
        "Unsupported wavelet boundary mode '{}'; expected zero, constant, symmetric, reflect, periodic, smooth, "
        "antisymmetric, or antireflect",
        name);
    return mode;
}

SchemeId scheme_id_from_string(const std::string_view name) {
    const SchemeId id = scheme_id(name);
    TT_FATAL(id != SchemeId::kUnknown, "Unsupported wavelet scheme '{}'", name);
    return id;
}

const SchemeInfo& scheme_info(const SchemeId id) {
    const size_t index = static_cast<size_t>(id);
    TT_FATAL(index < kSchemeInfos.size(), "Invalid wavelet scheme id {}", index);
    return kSchemeInfos[index];
}

}  // namespace ttnn::operations::wavelet
