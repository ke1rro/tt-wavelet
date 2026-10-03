// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttwt/device/wavelet_device_operation_types.hpp"

namespace ttwt::prim {

struct Ilwt2DProgramFactory {
  static tt::tt_metal::WorkloadDescriptor
  create_workload_descriptor(const Ilwt2DParams &operation_attributes,
                             const Ilwt2DInputs &tensor_args, ttnn::Tensor &tensor_return_value,
                             const ttnn::MeshCoordinateRangeSet &tensor_coords);
};

} // namespace ttwt::prim
