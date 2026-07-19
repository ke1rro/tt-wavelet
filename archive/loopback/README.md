# Archived DRAM-loopback LWT

This directory contains the last pre-sharding 1D LWT implementation, recovered
from repository commit `7067d3b`. That implementation materialized lifting
intermediates in DRAM ping/pong buffers and launched separate predict/update or
scale programs for successive routes.

It was archived because production uses the dependency-local LWT/ILWT pipeline,
which keeps the three logical working slots in local L1 and writes only terminal
results to DRAM. The archive is not part of any production CMake target and is
not expected to build against the current TT-Metal API without adaptation.

`source/` preserves the historical host dispatch, planner, LWT-specific kernels,
scale kernels, CLI/build glue, comparison script, and documentation as they
existed immediately before commit `2fd9b6c` introduced L1 sharding. Shared
current primitives were not duplicated unless the historical backend had a
backend-specific version of the file.
