# Archived vertical-stencil verification harness

This directory contains the standalone host executable, reader/compute/writer
test kernels, validation script, and historical CMake target for the vertical
SFPU stencil verification program from commit `8e83754`.

The harness was archived because it is verification-only and must not enlarge
the production LWT build surface. Nothing in this directory is included by the
production CMake target.

The reusable production primitive remains at
`tt-wavelet/kernels/sfpi/vertical_stencil_sfpi.h`, and its mathematical and
register-layout documentation remains in `docs/VERTICAL_STENCIL.md`. The
archived kernels are the last known direct consumers of that primitive.
