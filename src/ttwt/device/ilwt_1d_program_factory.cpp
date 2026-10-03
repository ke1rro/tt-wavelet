// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttwt/device/ilwt_1d_program_factory.hpp"

#include "ttwt/device/wavelet_1d_operation_impl.hpp"

namespace ttwt::prim {

tt::tt_metal::WorkloadDescriptor Ilwt1DProgramFactory::create_workload_descriptor(
    const Ilwt1DParams& operation_attributes,
    const Ilwt1DInputs& tensor_args,
    ttnn::Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    return detail::create_ilwt_1d_workload(operation_attributes, tensor_args, tensor_return_value, tensor_coords);
}

}  // namespace ttwt::prim
