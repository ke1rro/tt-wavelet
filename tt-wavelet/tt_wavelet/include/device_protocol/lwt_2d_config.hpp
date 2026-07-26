// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttwv::device_protocol {

inline constexpr uint32_t kLwt2DProtocolVersion = 1;
inline constexpr uint32_t kLwt2DTileHeight = 32;
inline constexpr uint32_t kLwt2DTileWidth = 32;
inline constexpr uint32_t kLwt2DFullTileElements = kLwt2DTileHeight * kLwt2DTileWidth;
inline constexpr uint32_t kLwt2DFullTileBytes = kLwt2DFullTileElements * sizeof(float);
inline constexpr uint32_t kLwt2DPlaneCount = 5;
inline constexpr uint32_t kLwt2DBandCount = 4;
// A 64x64 logical raw window can intersect three physical tiles on either
// axis when the wavelet padding is not tile-aligned.  The fused split reader
// stages the Cartesian product once before producing EE/EO/OE/OO.
inline constexpr uint32_t kLwt2DSplitScratchTileRows = 3;
inline constexpr uint32_t kLwt2DSplitScratchTileColumns = 3;
inline constexpr uint32_t kLwt2DSplitScratchTileCount = kLwt2DSplitScratchTileRows * kLwt2DSplitScratchTileColumns;
inline constexpr uint32_t kLwt2DSplitScratchBytes = kLwt2DSplitScratchTileCount * kLwt2DFullTileBytes;

inline constexpr uint32_t kLwt2DSplitMetricCyclesLow = 0;
inline constexpr uint32_t kLwt2DSplitMetricCyclesHigh = 1;
inline constexpr uint32_t kLwt2DSplitMetricInputBytes = 2;
inline constexpr uint32_t kLwt2DSplitMetricLocalOutputBytes = 3;
inline constexpr uint32_t kLwt2DSplitMetricNocReadCalls = 4;
inline constexpr uint32_t kLwt2DSplitMetricNocReadBarriers = 5;
inline constexpr uint32_t kLwt2DSplitMetricInteriorMacroTiles = 6;
inline constexpr uint32_t kLwt2DSplitMetricBoundaryMacroTiles = 7;
inline constexpr uint32_t kLwt2DSplitMetricWordCount = 16;
inline constexpr uint32_t kLwt2DSplitMetricPageBytes = kLwt2DSplitMetricWordCount * sizeof(uint32_t);

// Transport profiling is compiled out of production variants.  One page is
// emitted per executable route and one summary page per chunk.  The reader
// owns the first cache line and the writer owns the second, so both data-
// movement kernels can report asynchronously without a read/modify/write race.
inline constexpr uint32_t kLwt2DTransportMetricReaderConfigCyclesLow = 0;
inline constexpr uint32_t kLwt2DTransportMetricReaderConfigCyclesHigh = 1;
inline constexpr uint32_t kLwt2DTransportMetricStagingCyclesLow = 2;
inline constexpr uint32_t kLwt2DTransportMetricStagingCyclesHigh = 3;
inline constexpr uint32_t kLwt2DTransportMetricSyncWaitCyclesLow = 4;
inline constexpr uint32_t kLwt2DTransportMetricSyncWaitCyclesHigh = 5;
inline constexpr uint32_t kLwt2DTransportMetricOutputTiles = 6;
inline constexpr uint32_t kLwt2DTransportMetricAxis = 7;
inline constexpr uint32_t kLwt2DTransportMetricStepType = 8;
inline constexpr uint32_t kLwt2DTransportMetricCoefficientCount = 9;
inline constexpr uint32_t kLwt2DTransportMetricExactSourceTiles = 10;
inline constexpr uint32_t kLwt2DTransportMetricShiftedSourceTiles = 11;
inline constexpr uint32_t kLwt2DTransportMetricGenericSourceTiles = 12;
inline constexpr uint32_t kLwt2DTransportMetricExactBaseTiles = 13;
inline constexpr uint32_t kLwt2DTransportMetricShiftedBaseTiles = 14;
inline constexpr uint32_t kLwt2DTransportMetricGenericBaseTiles = 15;
// The per-chunk summary page has no route payload.  Validation-only reader
// variants reuse two otherwise-zero reader fields to report direct CB staging
// checks without growing the normal metrics page.
inline constexpr uint32_t kLwt2DTransportMetricValidatedStagingTiles = kLwt2DTransportMetricOutputTiles;
inline constexpr uint32_t kLwt2DTransportMetricStagingValidationMismatches = kLwt2DTransportMetricAxis;

inline constexpr uint32_t kLwt2DTransportMetricWriterConfigCyclesLow = 16;
inline constexpr uint32_t kLwt2DTransportMetricWriterConfigCyclesHigh = 17;
inline constexpr uint32_t kLwt2DTransportMetricPersistenceCyclesLow = 18;
inline constexpr uint32_t kLwt2DTransportMetricPersistenceCyclesHigh = 19;
inline constexpr uint32_t kLwt2DTransportMetricComputeCyclesLow = 20;
inline constexpr uint32_t kLwt2DTransportMetricComputeCyclesHigh = 21;
inline constexpr uint32_t kLwt2DTransportMetricPersistenceTiles = 22;
inline constexpr uint32_t kLwt2DTransportMetricExactTerminalTiles = 23;
inline constexpr uint32_t kLwt2DTransportMetricFragmentedTerminalTiles = 24;
inline constexpr uint32_t kLwt2DTransportMetricTerminalWriteCyclesLow = 25;
inline constexpr uint32_t kLwt2DTransportMetricTerminalWriteCyclesHigh = 26;
inline constexpr uint32_t kLwt2DTransportMetricKernelCyclesLow = 27;
inline constexpr uint32_t kLwt2DTransportMetricKernelCyclesHigh = 28;
inline constexpr uint32_t kLwt2DTransportMetricReserved0 = 29;
inline constexpr uint32_t kLwt2DTransportMetricReserved1 = 30;
inline constexpr uint32_t kLwt2DTransportMetricReserved2 = 31;
inline constexpr uint32_t kLwt2DTransportMetricValidatedPersistenceTiles = kLwt2DTransportMetricReserved0;
inline constexpr uint32_t kLwt2DTransportMetricPersistenceValidationMismatches = kLwt2DTransportMetricReserved1;
inline constexpr uint32_t kLwt2DTransportMetricWordCount = 32;
inline constexpr uint32_t kLwt2DTransportMetricPageBytes = kLwt2DTransportMetricWordCount * sizeof(uint32_t);
inline constexpr uint32_t kLwt2DTransportMetricHalfPageBytes = kLwt2DTransportMetricPageBytes / 2;

// One chunk page describes both the exact logical dependency cone and the
// padded tile region owned by the worker.
inline constexpr uint32_t kLwt2DFinalYBegin = 0;
inline constexpr uint32_t kLwt2DFinalYLength = 1;
inline constexpr uint32_t kLwt2DFinalXBegin = 2;
inline constexpr uint32_t kLwt2DFinalXLength = 3;
inline constexpr uint32_t kLwt2DExecutionTileYBegin = 4;
inline constexpr uint32_t kLwt2DExecutionTileYCount = 5;
inline constexpr uint32_t kLwt2DExecutionTileXBegin = 6;
inline constexpr uint32_t kLwt2DExecutionTileXCount = 7;

inline constexpr uint32_t kLwt2DInitialEe = 8;
inline constexpr uint32_t kLwt2DInitialEo = 12;
inline constexpr uint32_t kLwt2DInitialOe = 16;
inline constexpr uint32_t kLwt2DInitialOo = 20;
inline constexpr uint32_t kLwt2DRectYBegin = 0;
inline constexpr uint32_t kLwt2DRectYLength = 1;
inline constexpr uint32_t kLwt2DRectXBegin = 2;
inline constexpr uint32_t kLwt2DRectXLength = 3;

inline constexpr uint32_t kLwt2DChunkConfigWordCount = 24;
inline constexpr uint32_t kLwt2DChunkConfigPageBytes = kLwt2DChunkConfigWordCount * sizeof(uint32_t);

// Route pages are fixed-size so reader and writer kernels can use one-packet
// metadata reads. Rectangles remain in logical polyphase coordinates; the
// aligned local origin is floor(begin / 32) * 32 on the active plane.
inline constexpr uint32_t kLwt2DRouteAxis = 0;
inline constexpr uint32_t kLwt2DRouteType = 1;
inline constexpr uint32_t kLwt2DRouteSourceSlot = 2;
inline constexpr uint32_t kLwt2DRouteBaseSlot = 3;
inline constexpr uint32_t kLwt2DRouteOutputSlot = 4;
inline constexpr uint32_t kLwt2DRouteSourceRect = 5;
inline constexpr uint32_t kLwt2DRouteBaseRect = 9;
inline constexpr uint32_t kLwt2DRouteOutputRect = 13;
inline constexpr uint32_t kLwt2DRouteFlags = 17;
inline constexpr uint32_t kLwt2DRouteAxisStepIndex = 18;
inline constexpr uint32_t kLwt2DRouteConfigWordCount = 24;
inline constexpr uint32_t kLwt2DRouteConfigPageBytes = kLwt2DRouteConfigWordCount * sizeof(uint32_t);

inline constexpr uint32_t kLwt2DRouteFlagMetadataOnly = 1U << 0;
inline constexpr uint32_t kLwt2DRouteFlagScale = 1U << 1;
inline constexpr uint32_t kLwt2DRouteFlagInlineTerminalScale = 1U << 2;

inline constexpr uint32_t kLwt2DBandFinalYBegin = 0;
inline constexpr uint32_t kLwt2DBandFinalYLength = 1;
inline constexpr uint32_t kLwt2DBandFinalXBegin = 2;
inline constexpr uint32_t kLwt2DBandFinalXLength = 3;
inline constexpr uint32_t kLwt2DBandLl = 4;
inline constexpr uint32_t kLwt2DBandLh = 9;
inline constexpr uint32_t kLwt2DBandHl = 14;
inline constexpr uint32_t kLwt2DBandHh = 19;
inline constexpr uint32_t kLwt2DBandSourceSlot = 0;
inline constexpr uint32_t kLwt2DBandSourceRect = 1;
inline constexpr uint32_t kLwt2DBandConfigWordCount = 24;
inline constexpr uint32_t kLwt2DBandConfigPageBytes = kLwt2DBandConfigWordCount * sizeof(uint32_t);

static_assert(kLwt2DChunkConfigPageBytes == 96);
static_assert(kLwt2DRouteConfigPageBytes == 96);
static_assert(kLwt2DBandConfigPageBytes == 96);
static_assert(kLwt2DSplitMetricPageBytes == 64);
static_assert(kLwt2DTransportMetricPageBytes == 128);
static_assert(kLwt2DTransportMetricHalfPageBytes == 64);
static_assert(kLwt2DChunkConfigPageBytes % 32 == 0);
static_assert(kLwt2DRouteConfigPageBytes % 32 == 0);
static_assert(kLwt2DBandConfigPageBytes % 32 == 0);

}  // namespace ttwv::device_protocol
