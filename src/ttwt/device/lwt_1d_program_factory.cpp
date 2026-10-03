// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttwt/device/lwt_1d_program_factory.hpp"

#include "ttwt/device/wavelet_1d_operation_impl.hpp"

namespace ttwt::prim {

tt::tt_metal::WorkloadDescriptor Lwt1DProgramFactory::create_workload_descriptor(
    const Lwt1DParams &operation_attributes, const Lwt1DInputs &tensor_args,
    Lwt1DOutputs &tensor_return_value, const ttnn::MeshCoordinateRangeSet &tensor_coords) {
  return detail::create_lwt_1d_workload(operation_attributes, tensor_args, tensor_return_value,
                                        tensor_coords);
}

} // namespace ttwt::prim
