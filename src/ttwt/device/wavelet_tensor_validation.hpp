// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt::tt_metal::distributed {

class MeshDevice;

} // namespace tt::tt_metal::distributed

namespace ttwt::prim::wavelet_tensor_validation {

void validate_device_tensor(const ttnn::Tensor &tensor, const char *tensor_name);

void validate_input_memory_config(const ttnn::MemoryConfig &memory_config, const char *tensor_name);

void validate_output_memory_config(const ttnn::MemoryConfig &memory_config,
                                   const char *operation_name);

void validate_preallocated_output_placement(
    const ttnn::Tensor &output, const tt::tt_metal::distributed::MeshDevice *expected_device,
    const char *output_name);

void validate_same_device(const ttnn::Tensor &tensor,
                          const tt::tt_metal::distributed::MeshDevice *expected_device,
                          const char *error_message);

void validate_distinct_buffers(const ttnn::Tensor &lhs, const ttnn::Tensor &rhs,
                               const char *error_message);

} // namespace ttwt::prim::wavelet_tensor_validation
