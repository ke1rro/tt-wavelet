// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

#include "ttnn/operations/wavelet/common/boundary.hpp"

namespace ttnn::operations::wavelet {

[[nodiscard]] constexpr std::string_view boundary_mode_name(const BoundaryMode mode) noexcept {
    switch (mode) {
        case BoundaryMode::kZero: return "zero";
        case BoundaryMode::kConstant: return "constant";
        case BoundaryMode::kSymmetric: return "symmetric";
        case BoundaryMode::kPeriodic: return "periodic";
        case BoundaryMode::kAntisymmetric: return "antisymmetric";
        case BoundaryMode::kSmooth: return "smooth";
        case BoundaryMode::kAntireflect: return "antireflect";
        case BoundaryMode::kReflect: return "reflect";
    }
    return "unsupported";
}

[[nodiscard]] constexpr bool parse_boundary_mode(const std::string_view name, BoundaryMode& mode) noexcept {
    if (name == "zero") {
        mode = BoundaryMode::kZero;
    } else if (name == "constant") {
        mode = BoundaryMode::kConstant;
    } else if (name == "symmetric") {
        mode = BoundaryMode::kSymmetric;
    } else if (name == "periodic") {
        mode = BoundaryMode::kPeriodic;
    } else if (name == "antisymmetric") {
        mode = BoundaryMode::kAntisymmetric;
    } else if (name == "smooth") {
        mode = BoundaryMode::kSmooth;
    } else if (name == "antireflect") {
        mode = BoundaryMode::kAntireflect;
    } else if (name == "reflect") {
        mode = BoundaryMode::kReflect;
    } else {
        return false;
    }
    return true;
}

}  // namespace ttnn::operations::wavelet
