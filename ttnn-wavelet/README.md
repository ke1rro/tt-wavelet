# TTNN wavelet integration checkpoint

This directory is a source checkpoint of the experimental TTNN wavelet
integration. It is intentionally kept outside the active `tt-metal` submodule
and is not part of the standalone `tt-wavelet` build.

No claim is made that this snapshot builds against the currently pinned
`tt-metal`. Its purpose is to preserve the implementation while the submodule
tracks upstream `main`.

The standalone Wormhole tile-mirror, aligned-NoC, B96, and route-count planner
optimizations are mirrored in this checkpoint. The TTNN port keeps them
Wormhole-specific, adapts the mirror to the static L1 workspace circular
buffer, and retains TTNN row-major tensor page addressing. Blackhole keeps its
independently calibrated tile-native policy. The retained measurements and
policy are documented in `../docs/LWT_TILE_NATIVE_OPTIMIZATION.md`.

## Source revisions

- Integration base: `0e9f008f8f303415445285787437c1a09175ca54`
- Integration head: `95fc5f801be11ebba0e85d49182825a8895bbbac`
- Current parent pin at extraction time:
  `acf1da82d708c9db712ca6e1fbee8f5a4818ed12`
- Extraction date: 2026-08-01

The preserved integration commit series is:

1. `dcf0a50c8ba965197f3a7dcc8df7ee99111d8aaa` — Add TTNN wavelet operation skeleton
2. `1de9648e44ee34d26bae0f4c393bc0d3492a171e` — Port TTNN wavelet host planning infrastructure
3. `a215097b00b6e45a2b6d967f089c60a8e0296f2d` — Port TTNN wavelet device kernels
4. `16962bc79773c53e139117d199a23390f89a397f` — Implement TTNN 1D wavelet primitives
5. `2c3b47f20baffc828989a261534a369424171889` — Implement TTNN 2D wavelet primitives
6. `c3d3e1a12ca987a51041e995fabbb3b73038e242` — Harden TTNN wavelet kernels and cached workspaces
7. `5e056bc6a09a7dd647f0e7e92d8de962e13e9b08` — Add TTNN wavelet correctness and cache coverage
8. `95fc5f801be11ebba0e85d49182825a8895bbbac` — Document TTNN wavelet operation contracts

## Contents

- `ttnn/cpp/ttnn/operations/wavelet/` — public API, host planner,
  device program factories, kernels, generated definitions for all 106
  schemes, build metadata, and operation documentation.
- `tests/ttnn/unit_tests/operations/wavelet/` — correctness, scheme, and
  program-cache tests.
- `patches/integration-hooks.patch` — the original additions to TTNN's
  operations CMake registration and nanobind module. This patch is recorded
  for reference and is not expected to apply cleanly to later upstream trees.

## Future restoration

When adapting this checkpoint to a newer `tt-metal`, copy the preserved
operation and tests into the matching upstream-relative paths, then port the
two registration changes manually using `patches/integration-hooks.patch` as
a reference. Keep the currently pinned submodule untouched until that port is
done in a dedicated branch or worktree.
