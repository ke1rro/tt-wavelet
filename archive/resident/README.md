# Archived Resident LWT

This directory contains the ResidentSharded 1D LWT implementation from commit
`8e83754`. It kept complete even, odd, and scratch streams in height-sharded L1,
used a separate pad/split device program, and synchronized workers after each
lifting route.

It was archived because the production LWT/ILWT path is dependency-local and
scales beyond a complete three-slot resident working set. The files under this
directory are excluded from production builds on the production branch. This
archive branch records their last direct relationship to the shared planner,
FP32 horizontal stencil, route protocol, and layout helpers.

Shared code still used by the dependency-local implementation was deliberately
not moved. In particular, the common lifting geometry, FP32 compute helpers,
stick cache, and row-major/tile-native mapping remain production concepts.
