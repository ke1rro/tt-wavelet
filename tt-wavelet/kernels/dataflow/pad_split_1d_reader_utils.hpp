#pragma once

// TT-Metal adds the kernel source directory to the include path, but not the
// whole kernels/ tree. Keep the relative sibling includes in one place so the
// reader kernel can include a single local header.

#include "../utils/boundary.hpp"
#include "../utils/output_stick_writer.hpp"
#include "../utils/stick_read_cache.hpp"
