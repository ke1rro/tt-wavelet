# Dependency-local LWT memory model

Production LWT and ILWT use a single dependency-local execution model. This file retains its established name because it documents the two physical workspace layouts, not multiple backends.

## Planner

The forward planner first records the complete lifting trace: stream lengths and shifts, source and base offsets, output intervals, swaps, boundary policy, and terminal destinations. For each 1,536-element output group, it propagates requirements backward through the trace. Adjacent groups are combined into chunks subject to the per-core signal budget.

ILWT starts from a cropped interval of the original signal and reverses the recorded forward geometry. The planner maps the requested even and odd polyphase intervals back to canonical coefficient intervals and emits the inverse routes in reverse order.

The planner invariants are:

- `K=1..17`, arbitrary signed scheme shifts, and swaps remain supported;
- each active core owns all intermediate values required by its chunks;
- the largest logical stream in any route is `max_workspace_elements`;
- each physical slot is aligned to the selected layout and has `workspace_elements`;
- terminal forward intervals and the reconstructed signal write directly to DRAM;
- no route depends on another core's workspace.

The default signal budget is controlled by `TT_WAVELET_L1_SIGNAL_BUDGET_BYTES` and is 768 KiB. `TT_WAVELET_LWT_MAX_CORES` can restrict active workers for measurements.

## Workspace layouts

The three physical L1 slots have identical size and serve as `A`, `B`, and `Scratch` under metadata-only swaps.

### Row-major

Values are consecutive FP32 elements. Shifted source/base offsets require no physical remap, so this layout is generally best on Wormhole and for low-order shifted routes such as `db7`.

### Tile-native

Each 1,536-element group occupies three consecutive `32x16` FP32 pages. A logical position is mapped by group, block, row, and lane. Aligned base reads and intermediate writes become three page transfers instead of 96 half-stick packets. Shifted offsets still require reader remapping; the layout is not universally zero-copy.

Forward automatic selection keeps the measured route-geometry heuristic: tile-native is selected when at least half of predict/update base routes are page aligned. ILWT selection comes from the architecture policy in [LWT.md](LWT.md).

Override only for A/B validation:

```bash
export TT_WAVELET_LWT_WORKSPACE_LAYOUT=auto
export TT_WAVELET_LWT_WORKSPACE_LAYOUT=row-major
export TT_WAVELET_LWT_WORKSPACE_LAYOUT=tile-native
```

## Device dataflow

The reader initializes the exact extended input dependencies or canonical inverse coefficients, then supplies source and base pages for every route. The compute kernel consumes four source, three base, and produces three output narrow pages per group. The writer persists intermediate pages to local slots or writes terminal data to DRAM. A local 32-byte synchronization CB prevents a reader from consuming a route until the prior writer has made it visible.

The finite-signal extension is a compile-time reader specialization. The direct interior path is shared. Boundary-only helpers are callable on Wormhole to stay below its NCRISC instruction limit; Blackhole keeps those helpers inline.

## Exact L1 accounting

[`l1_accounting.hpp`](../tt-wavelet/tt_wavelet/include/lifting/l1_accounting.hpp) uses the same workspace lengths and protocol constants as the allocations in `device.cpp`. Each CLI run reports architecture, layout, signal length, active cores, every category below, total, device capacity, and headroom.

| Category | Formula |
| --- | ---: |
| Slots | `3 * max_workspace_elements * 4` |
| Slot padding | `3 * (workspace_elements - max_workspace_elements) * 4` |
| Source and base CBs | `8 * 2,048 + 6 * 2,048 = 28,672` |
| Cache | `4 * 128 = 512` |
| Output CB and interleave CB | `6 * 2,048 + 128 = 12,416` |
| Synchronization | `32` |
| Reader/writer config metadata | `2 * 64 = 128` |
| Alignment | `0`; all listed allocations already meet their required alignment |
| Architecture scratch | `0`; production allocates no additional project-managed scratch |

For the worst tested 8,000,000-element tile-native `bior3.9` forward case on Wormhole B0:

| Field | Bytes/core |
| --- | ---: |
| Logical slots | 755,820 |
| Circular buffers | 28,672 |
| Cache | 512 |
| Outputs | 12,416 |
| Synchronization | 32 |
| Metadata | 128 |
| Alignment | 0 |
| Slot padding | 18,324 |
| Architecture scratch | 0 |
| **Total** | **815,904** |
| Device-reported capacity | 1,499,136 |
| Headroom | 683,232 |

The historical 815,776-byte estimate omitted the 128-byte interleave CB page. The new total is 128 bytes larger because that real allocation is now included, not because the implementation allocates more memory.

Allocation fails before execution if the exact project-managed total exceeds `mesh_device.l1_size_per_core()`.

## NCRISC ELF gate

After a device JIT run:

```bash
python3 scripts/check_ncrisc_elf_size.py --architecture wormhole_b0
```

The script reads successful `lwt_reader` NCRISC ELF files from the TT-Metal cache, derives architecture from compiler dependency metadata, reads `.text` using the pinned `riscv-tt-elf-size`, and fails if any Wormhole variant exceeds `0x4000` bytes. Blackhole is explicitly skipped. The largest required Wormhole variant is tile-native antireflect at 15,004 bytes, leaving 1,380 bytes.

## Telemetry example

```text
lwt_architecture: wormhole_b0
lwt_layout: tile-native
lwt_signal_length: 8000000
lwt_active_core_count: 64
lwt_l1_total_bytes: 815904
lwt_l1_capacity_bytes: 1499136
lwt_l1_headroom_bytes: 683232
```
