// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

#include "ttnn/operations/wavelet/common/boundary.hpp"
#include "ttnn/operations/wavelet/generated/schemes/registry.hpp"

namespace ttnn::operations::wavelet {

[[nodiscard]] BoundaryMode boundary_mode_from_string(std::string_view name);

[[nodiscard]] SchemeId scheme_id_from_string(std::string_view name);

[[nodiscard]] const SchemeInfo& scheme_info(SchemeId id);

}  // namespace ttnn::operations::wavelet
