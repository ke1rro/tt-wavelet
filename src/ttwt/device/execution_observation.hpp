// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace ttwt::detail {
// Passive benchmark observation of the existing planner's choice; no policy override.
struct ExecutionObservation {
  uint32_t cores = 0;
  uint32_t programs = 0;
};
inline thread_local ExecutionObservation execution_observation;
} // namespace ttwt::detail
