#!/usr/bin/env python3
"""Benchmark and compare production LWT/ILWT workspace layouts.

Timing uses the C++ device-only benchmark boundary.  The executable performs one
additional untimed readback after timing and writes raw FP32 output files for
reference and peer-layout checks.

The all-scheme correctness policy intentionally separates layout correctness
from the documented high-order FP32/PyWavelets compatibility limitation:

* output shape, finiteness, forced-layout selection, and peer bit identity are
  mandatory for every scheme;
* the representative wavelets with established strict tolerances retain those
  tolerances;
* other PyWavelets errors above 1e-2 are recorded as the accepted high-order
  FP32 limitation and do not by themselves fail this layout benchmark.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python3"

try:
    import numpy as np
    import pywt
except ModuleNotFoundError as exc:
    if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])
    raise ModuleNotFoundError(
        "Missing benchmark dependencies. Activate .venv or install numpy and PyWavelets."
    ) from exc

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# isort: off
from compare_timings import TTTimingResult, run_tt_wavelet, sh_quote  # noqa: E402
from runtime_checks import parse_runtime_architecture  # noqa: E402
from validate_ilwt import DEFAULT_TOLERANCES as ILWT_TOLERANCES  # noqa: E402
from validate_lwt_boundaries import (  # noqa: E402
    DEFAULT_TOLERANCES as LWT_TOLERANCES,
)

# isort: on

EXPECTED_SCHEME_COUNT = 106
LAYOUTS = ("row-major", "tile-native", "auto")
TRANSFORMS = ("lwt", "ilwt")
DEFAULT_REFERENCE_TOLERANCE = 1.0e-2

RAW_FIELDS = (
    "architecture",
    "transform",
    "wavelet",
    "signal_length",
    "layout",
    "selected_concrete_layout",
    "warmup_runs",
    "timed_runs",
    "median_ms",
    "min_ms",
    "mean_ms",
    "p10_ms",
    "p90_ms",
    "stddev_ms",
    "reference_max_abs_error",
    "peer_layout_max_abs_difference",
    "peer_layout_bit_identical",
    "first_peer_mismatch_index",
    "status",
    "error",
)

SUMMARY_FIELDS = (
    "architecture",
    "transform",
    "wavelet",
    "signal_length",
    "row_major_median_ms",
    "tile_native_median_ms",
    "auto_median_ms",
    "tile_over_row_speedup",
    "auto_over_best_forced_speedup",
    "difference_percent",
    "winner",
    "auto_selected_layout",
    "tie_threshold_percent",
    "row_major_reference_error",
    "tile_native_reference_error",
    "auto_reference_error",
    "layout_max_abs_difference",
    "layout_bit_identical",
    "correctness_status",
    "auto_result",
)


@dataclass
class LayoutCase:
    transform: str
    wavelet: str
    signal_length: int
    requested_layout: str
    expected_architecture: str
    warmup_runs: int
    timed_runs: int
    timing: TTTimingResult | None = None
    architecture: str = ""
    selected_layout: str = ""
    output: np.ndarray | None = None
    reference_error: float | None = None
    peer_max_difference: float | None = None
    peer_bit_identical: bool | None = None
    first_peer_mismatch: int | None = None
    timing_failed: bool = False
    correctness_failed: bool = False
    accepted_fp32_limit: bool = False
    architecture_failed: bool = False
    errors: list[str] = field(default_factory=list)

    def fail_timing(self, message: str) -> None:
        self.timing_failed = True
        self.errors.append(message)

    def fail_correctness(self, message: str) -> None:
        self.correctness_failed = True
        self.errors.append(message)


@dataclass(frozen=True)
class PairComparison:
    lhs: str
    rhs: str
    max_abs_difference: float
    bit_identical: bool
    first_mismatch: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exhaustively benchmark row-major, tile-native, and automatic production "
            "workspace layouts for LWT and ILWT."
        )
    )
    parser.add_argument("--binary", type=Path, default=PROJECT_ROOT / "build" / "lwt")
    parser.add_argument("--schemes-dir", type=Path, default=PROJECT_ROOT / "wavelets")
    parser.add_argument(
        "--registry",
        type=Path,
        default=(
            PROJECT_ROOT
            / "tt-wavelet"
            / "tt_wavelet"
            / "include"
            / "schemes"
            / "generated"
            / "registry.hpp"
        ),
    )
    parser.add_argument(
        "--wavelets",
        nargs="+",
        help="Selected wavelets (default: all 106 production schemes).",
    )
    parser.add_argument("--transform", choices=["lwt", "ilwt", "both"], default="both")
    parser.add_argument("--lengths", nargs="+", type=int, default=[5_000_000])
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--timed-runs", type=int, default=20)
    parser.add_argument(
        "--tie-threshold-percent",
        type=float,
        default=2.0,
        help="Forced medians closer than this percentage are a tie (default: %(default)s).",
    )
    parser.add_argument(
        "--expected-architecture",
        choices=["wormhole_b0", "blackhole"],
        default="wormhole_b0",
    )
    parser.add_argument(
        "--boundary-mode",
        choices=[
            "symmetric",
            "zero",
            "constant",
            "periodic",
            "antisymmetric",
            "smooth",
            "reflect",
            "antireflect",
        ],
        default="symmetric",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "benchmark_results" / "lwt_ilwt_layout",
        help=(
            "Prefix for <prefix>_raw.csv and <prefix>_summary.csv "
            "(default: %(default)s)."
        ),
    )
    args = parser.parse_args()

    if any(length <= 0 for length in args.lengths):
        parser.error("all signal lengths must be positive")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    if args.timed_runs <= 0:
        parser.error("--timed-runs must be positive")
    if (
        not math.isfinite(args.tie_threshold_percent)
        or args.tie_threshold_percent < 0.0
    ):
        parser.error("--tie-threshold-percent must be finite and non-negative")
    if args.boundary_mode in {"reflect", "antireflect"} and any(
        length == 1 for length in args.lengths
    ):
        parser.error(
            "reflect and antireflect require every signal length to be greater than one"
        )
    return args


def load_supported_wavelets(schemes_dir: Path, registry: Path) -> list[str]:
    json_names = sorted(path.stem for path in schemes_dir.glob("*.json"))
    if not registry.is_file():
        raise FileNotFoundError(f"generated scheme registry not found: {registry}")
    registry_names = re.findall(
        r'SchemeInfo\{"([A-Za-z0-9_.-]+)"', registry.read_text(encoding="utf-8")
    )
    if len(json_names) != EXPECTED_SCHEME_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_SCHEME_COUNT} scheme JSON files, found {len(json_names)}"
        )
    if len(registry_names) != EXPECTED_SCHEME_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_SCHEME_COUNT} generated registry entries, "
            f"found {len(registry_names)}"
        )
    if len(set(registry_names)) != len(registry_names):
        raise RuntimeError("generated scheme registry contains duplicate names")
    if set(json_names) != set(registry_names):
        only_json = sorted(set(json_names) - set(registry_names))
        only_registry = sorted(set(registry_names) - set(json_names))
        raise RuntimeError(
            "scheme JSON/registry mismatch: "
            f"only_json={only_json}, only_registry={only_registry}"
        )
    return sorted(registry_names)


def selected_wavelets(
    all_wavelets: Sequence[str], requested: Sequence[str] | None
) -> list[str]:
    if requested is None:
        return list(all_wavelets)
    duplicates = sorted({name for name in requested if requested.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate --wavelets entries: {', '.join(duplicates)}")
    unknown = sorted(set(requested) - set(all_wavelets))
    if unknown:
        raise ValueError(f"unsupported wavelets: {', '.join(unknown)}")
    return list(requested)


def transforms_from_arg(value: str) -> list[str]:
    return list(TRANSFORMS) if value == "both" else [value]


def format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.12g}"


def bool_field(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def reference_tolerance(transform: str, wavelet: str) -> tuple[float, bool]:
    table = LWT_TOLERANCES if transform == "lwt" else ILWT_TOLERANCES
    if wavelet in table:
        return table[wavelet], True
    return DEFAULT_REFERENCE_TOLERANCE, False


def write_fp32_text(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float32)
    np.savetxt(path, values, fmt="%.9g")


def make_benchmark_signal(length: int) -> np.ndarray:
    """Return a deterministic bounded signal suitable for absolute tolerances."""
    index = np.arange(length, dtype=np.float32)
    return (
        0.7 * np.sin(index * 0.071)
        + 0.2 * np.cos(index * 0.013)
        + 0.1 * np.sin(index * 0.003)
    ).astype(np.float32)


def max_abs_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    if candidate.shape != reference.shape:
        return float("inf")
    if candidate.size == 0:
        return 0.0
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    return float(np.max(np.abs(difference)))


def output_paths(prefix: Path, transform: str) -> list[Path]:
    if transform == "lwt":
        return [
            Path(str(prefix) + ".approximation.f32"),
            Path(str(prefix) + ".detail.f32"),
        ]
    return [Path(str(prefix) + ".reconstructed.f32")]


def read_output(prefix: Path, transform: str) -> np.ndarray:
    paths = output_paths(prefix, transform)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "device run did not write expected FP32 output files: "
            + ", ".join(str(path) for path in missing)
        )
    arrays = [np.fromfile(path, dtype=np.float32) for path in paths]
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays)


def remove_output_files(prefix: Path, transform: str) -> None:
    for path in output_paths(prefix, transform):
        path.unlink(missing_ok=True)


def build_command(
    binary: Path,
    environment_script: Path,
    transform: str,
    wavelet: str,
    signal_length: int,
    signal_path: Path,
    approximation_path: Path,
    detail_path: Path,
    output_prefix: Path,
    boundary_mode: str,
    warmup_runs: int,
    timed_runs: int,
) -> str:
    command = [str(binary)]
    if transform == "ilwt":
        command.append("--inverse")
    command.extend(
        [
            "--benchmark",
            "--repeats",
            str(timed_runs),
            "--warmup-runs",
            str(warmup_runs),
            "--boundary-mode",
            boundary_mode,
            "--output-prefix",
            str(output_prefix),
        ]
    )
    if transform == "lwt":
        command.extend([wavelet, str(signal_path)])
    else:
        command.extend(
            [
                "--original-length",
                str(signal_length),
                "--approximation-file",
                str(approximation_path),
                "--detail-file",
                str(detail_path),
                wavelet,
            ]
        )
    quoted_command = " ".join(sh_quote(argument) for argument in command)
    return (
        f"unset ARCH_NAME && source {sh_quote(str(environment_script))} "
        f"&& {quoted_command}"
    )


def validate_timing(case: LayoutCase) -> None:
    assert case.timing is not None
    required = {
        "mean": case.timing.mean_s,
        "minimum": case.timing.min_s,
        "median": case.timing.median_s,
        "p10": case.timing.p10_s,
        "p90": case.timing.p90_s,
        "standard deviation": case.timing.stddev_s,
    }
    missing = [name for name, value in required.items() if value is None]
    non_finite = [
        name
        for name, value in required.items()
        if value is not None and not math.isfinite(value)
    ]
    if missing:
        case.fail_timing("missing timing statistics: " + ", ".join(missing))
    if non_finite:
        case.fail_timing("non-finite timing statistics: " + ", ".join(non_finite))


def run_layout_case(
    *,
    args: argparse.Namespace,
    transform: str,
    wavelet: str,
    signal_length: int,
    requested_layout: str,
    signal_path: Path,
    approximation_path: Path,
    detail_path: Path,
    reference: np.ndarray,
    temporary_path: Path,
) -> LayoutCase:
    case = LayoutCase(
        transform=transform,
        wavelet=wavelet,
        signal_length=signal_length,
        requested_layout=requested_layout,
        expected_architecture=args.expected_architecture,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
    )
    output_prefix = temporary_path / (
        f"{transform}-{wavelet}-{signal_length}-{requested_layout}"
    )
    remove_output_files(output_prefix, transform)
    command = build_command(
        args.binary,
        PROJECT_ROOT / "scripts" / "set_env.sh",
        transform,
        wavelet,
        signal_length,
        signal_path,
        approximation_path,
        detail_path,
        output_prefix,
        args.boundary_mode,
        args.warmup_runs,
        args.timed_runs,
    )
    try:
        case.timing = run_tt_wavelet(
            command,
            {"TT_WAVELET_LWT_WORKSPACE_LAYOUT": requested_layout},
        )
        validate_timing(case)
        if not case.timing.architecture:
            case.architecture_failed = True
            case.fail_correctness("device timing did not report an architecture")
        else:
            case.architecture = parse_runtime_architecture(
                f"{transform}_architecture: {case.timing.architecture}"
            )
            if case.architecture != args.expected_architecture:
                case.architecture_failed = True
                case.fail_correctness(
                    "architecture assertion failed: "
                    f"expected {args.expected_architecture}, observed {case.architecture}"
                )

        case.selected_layout = case.timing.layout
        if requested_layout != "auto" and case.selected_layout != requested_layout:
            case.fail_correctness(
                "layout override was not honored: "
                f"requested {requested_layout}, selected {case.selected_layout or '<missing>'}"
            )
        if requested_layout == "auto" and case.selected_layout not in LAYOUTS[:2]:
            case.fail_correctness(
                "automatic policy did not report a concrete row-major or tile-native layout"
            )

        case.output = read_output(output_prefix, transform)
        case.reference_error = max_abs_error(case.output, reference)
        if case.output.shape != reference.shape:
            case.fail_correctness(
                f"output shape {case.output.shape} does not match reference {reference.shape}"
            )
        if not np.all(np.isfinite(case.output)):
            case.fail_correctness("device output contains NaN or infinity")
        tolerance, strict = reference_tolerance(transform, wavelet)
        if not math.isfinite(case.reference_error):
            case.fail_correctness("reference maximum absolute error is not finite")
        elif case.reference_error > tolerance:
            if strict:
                case.fail_correctness(
                    f"reference error {case.reference_error:.8e} exceeds {tolerance:.8e}"
                )
            else:
                case.accepted_fp32_limit = True
    except Exception as exc:  # noqa: BLE001
        case.fail_timing(str(exc))
    finally:
        remove_output_files(output_prefix, transform)
    return case


def compare_outputs(lhs: LayoutCase, rhs: LayoutCase) -> PairComparison | None:
    if lhs.output is None or rhs.output is None:
        return None
    if lhs.output.shape != rhs.output.shape:
        return PairComparison(
            lhs=lhs.requested_layout,
            rhs=rhs.requested_layout,
            max_abs_difference=float("inf"),
            bit_identical=False,
            first_mismatch=min(lhs.output.size, rhs.output.size),
        )
    lhs_bits = lhs.output.view(np.uint32)
    rhs_bits = rhs.output.view(np.uint32)
    mismatch = lhs_bits != rhs_bits
    mismatch_indices = np.flatnonzero(mismatch)
    bit_identical = mismatch_indices.size == 0
    max_difference = max_abs_error(lhs.output, rhs.output)
    return PairComparison(
        lhs=lhs.requested_layout,
        rhs=rhs.requested_layout,
        max_abs_difference=max_difference,
        bit_identical=bit_identical,
        first_mismatch=None if bit_identical else int(mismatch_indices[0]),
    )


def add_peer_comparisons(cases: list[LayoutCase]) -> list[PairComparison]:
    comparisons: list[PairComparison] = []
    for lhs_index, lhs in enumerate(cases):
        for rhs in cases[lhs_index + 1 :]:
            comparison = compare_outputs(lhs, rhs)
            if comparison is not None:
                comparisons.append(comparison)

    for case in cases:
        peers = [
            comparison
            for comparison in comparisons
            if case.requested_layout in {comparison.lhs, comparison.rhs}
        ]
        if len(peers) != len(LAYOUTS) - 1:
            continue
        case.peer_max_difference = max(
            comparison.max_abs_difference for comparison in peers
        )
        case.peer_bit_identical = all(comparison.bit_identical for comparison in peers)
        mismatch_indices = [
            comparison.first_mismatch
            for comparison in peers
            if comparison.first_mismatch is not None
        ]
        case.first_peer_mismatch = min(mismatch_indices) if mismatch_indices else None
        if not case.peer_bit_identical:
            case.fail_correctness("output is not bit-identical to every peer layout")
    return comparisons


def seconds_to_ms(value: float | None) -> float | None:
    return None if value is None else value * 1000.0


def median_ms(case: LayoutCase) -> float | None:
    if case.timing is None or case.timing_failed:
        return None
    return seconds_to_ms(case.timing.median_s)


def raw_status(case: LayoutCase) -> str:
    if case.timing_failed and case.correctness_failed:
        return "timing-and-correctness-failed"
    if case.timing_failed:
        return "timing-failed"
    if case.correctness_failed:
        return "correctness-failed"
    if case.accepted_fp32_limit:
        return "ok-accepted-fp32-limit"
    return "ok"


def raw_row(case: LayoutCase) -> dict[str, object]:
    timing = case.timing
    return {
        "architecture": case.architecture,
        "transform": case.transform,
        "wavelet": case.wavelet,
        "signal_length": case.signal_length,
        "layout": case.requested_layout,
        "selected_concrete_layout": case.selected_layout,
        "warmup_runs": case.warmup_runs,
        "timed_runs": case.timed_runs,
        "median_ms": format_number(
            seconds_to_ms(timing.median_s) if timing is not None else None
        ),
        "min_ms": format_number(
            seconds_to_ms(timing.min_s) if timing is not None else None
        ),
        "mean_ms": format_number(
            seconds_to_ms(timing.mean_s) if timing is not None else None
        ),
        "p10_ms": format_number(
            seconds_to_ms(timing.p10_s) if timing is not None else None
        ),
        "p90_ms": format_number(
            seconds_to_ms(timing.p90_s) if timing is not None else None
        ),
        "stddev_ms": format_number(
            seconds_to_ms(timing.stddev_s) if timing is not None else None
        ),
        "reference_max_abs_error": format_number(case.reference_error),
        "peer_layout_max_abs_difference": format_number(case.peer_max_difference),
        "peer_layout_bit_identical": bool_field(case.peer_bit_identical),
        "first_peer_mismatch_index": (
            "" if case.first_peer_mismatch is None else case.first_peer_mismatch
        ),
        "status": raw_status(case),
        "error": " | ".join(case.errors),
    }


def make_summary_row(
    cases: list[LayoutCase],
    comparisons: list[PairComparison],
    tie_threshold_percent: float,
) -> dict[str, object]:
    by_layout = {case.requested_layout: case for case in cases}
    row_median = median_ms(by_layout["row-major"])
    tile_median = median_ms(by_layout["tile-native"])
    auto_median = median_ms(by_layout["auto"])

    tile_over_row: float | None = None
    auto_over_best: float | None = None
    difference_percent: float | None = None
    winner = "failed"
    if row_median is not None and tile_median is not None:
        tile_over_row = tile_median / row_median
        best_forced = min(row_median, tile_median)
        difference_percent = 100.0 * abs(row_median - tile_median) / best_forced
        if difference_percent < tie_threshold_percent:
            winner = "tie"
        elif row_median < tile_median:
            winner = "row-major"
        else:
            winner = "tile-native"
        if auto_median is not None:
            auto_over_best = auto_median / best_forced

    auto_result = "failed"
    auto_selected = by_layout["auto"].selected_layout
    if auto_median is not None and auto_selected in LAYOUTS[:2] and winner != "failed":
        if winner == "tie":
            auto_result = "auto-tie"
        elif auto_selected == winner:
            auto_result = "auto-best"
        else:
            auto_result = "auto-not-best"

    complete_comparison = len(comparisons) == 3
    layout_identical = complete_comparison and all(
        comparison.bit_identical for comparison in comparisons
    )
    layout_max_difference = (
        max(comparison.max_abs_difference for comparison in comparisons)
        if complete_comparison
        else None
    )
    if any(case.correctness_failed for case in cases):
        correctness_status = "failed"
    elif not complete_comparison:
        correctness_status = "incomplete"
    elif any(case.accepted_fp32_limit for case in cases):
        correctness_status = "accepted-fp32-limit"
    else:
        correctness_status = "pass"

    architecture = next((case.architecture for case in cases if case.architecture), "")
    return {
        "architecture": architecture,
        "transform": cases[0].transform,
        "wavelet": cases[0].wavelet,
        "signal_length": cases[0].signal_length,
        "row_major_median_ms": format_number(row_median),
        "tile_native_median_ms": format_number(tile_median),
        "auto_median_ms": format_number(auto_median),
        "tile_over_row_speedup": format_number(tile_over_row),
        "auto_over_best_forced_speedup": format_number(auto_over_best),
        "difference_percent": format_number(difference_percent),
        "winner": winner,
        "auto_selected_layout": auto_selected,
        "tie_threshold_percent": format_number(tie_threshold_percent),
        "row_major_reference_error": format_number(
            by_layout["row-major"].reference_error
        ),
        "tile_native_reference_error": format_number(
            by_layout["tile-native"].reference_error
        ),
        "auto_reference_error": format_number(by_layout["auto"].reference_error),
        "layout_max_abs_difference": format_number(layout_max_difference),
        "layout_bit_identical": bool_field(
            layout_identical if complete_comparison else None
        ),
        "correctness_status": correctness_status,
        "auto_result": auto_result,
    }


def geometric_mean(values: Sequence[float]) -> float | None:
    positive = [value for value in values if math.isfinite(value) and value > 0.0]
    return statistics.geometric_mean(positive) if positive else None


def print_transform_summary(
    transform: str,
    raw_cases: Sequence[LayoutCase],
    summary_rows: Sequence[dict[str, object]],
    non_identical_pairs: Sequence[str],
    expected_cases: int,
    raw_path: Path,
    summary_path: Path,
) -> None:
    transform_cases = [case for case in raw_cases if case.transform == transform]
    transform_summaries = [row for row in summary_rows if row["transform"] == transform]
    ratios = [
        float(row["tile_over_row_speedup"])
        for row in transform_summaries
        if row["tile_over_row_speedup"] != ""
    ]
    auto_ratios = [
        float(row["auto_over_best_forced_speedup"])
        for row in transform_summaries
        if row["auto_over_best_forced_speedup"] != ""
    ]

    def largest_advantage(row_major: bool) -> str:
        candidates: list[tuple[float, dict[str, object]]] = []
        for row in transform_summaries:
            value = row["tile_over_row_speedup"]
            if value == "":
                continue
            ratio = float(value)
            if row_major and ratio > 1.0:
                candidates.append(((ratio - 1.0) * 100.0, row))
            elif not row_major and ratio < 1.0:
                candidates.append(((1.0 / ratio - 1.0) * 100.0, row))
        if not candidates:
            return "none"
        advantage, row = max(candidates, key=lambda item: item[0])
        return (
            f"{row['wavelet']} N={row['signal_length']} "
            f"({advantage:.3f}% lower median)"
        )

    print()
    print(f"{transform.upper()} summary")
    print(f"total_expected_cases: {expected_cases}")
    print(
        "completed_cases: "
        f"{sum(not case.timing_failed and case.output is not None for case in transform_cases)}"
    )
    print(
        "row_major_wins: "
        f"{sum(row['winner'] == 'row-major' for row in transform_summaries)}"
    )
    print(
        "tile_native_wins: "
        f"{sum(row['winner'] == 'tile-native' for row in transform_summaries)}"
    )
    print(f"ties: {sum(row['winner'] == 'tie' for row in transform_summaries)}")
    print(
        "auto_best_count: "
        f"{sum(row['auto_result'] == 'auto-best' for row in transform_summaries)}"
    )
    print(
        "auto_not_best_count: "
        f"{sum(row['auto_result'] == 'auto-not-best' for row in transform_summaries)}"
    )
    print(
        "correctness_failures: "
        f"{sum(case.correctness_failed for case in transform_cases)}"
    )
    print(f"runtime_failures: {sum(case.timing_failed for case in transform_cases)}")
    print("geomean_tile_over_row_speedup: " f"{format_number(geometric_mean(ratios))}")
    print(
        "geomean_auto_over_best_forced_speedup: "
        f"{format_number(geometric_mean(auto_ratios))}"
    )
    print(f"largest_row_major_advantage: {largest_advantage(True)}")
    print(f"largest_tile_native_advantage: {largest_advantage(False)}")
    matching_pairs = [
        pair for pair in non_identical_pairs if pair.startswith(f"{transform} ")
    ]
    if matching_pairs:
        print("non_bit_identical_layout_pairs:")
        for pair in matching_pairs:
            print(f"  {pair}")
    else:
        print("non_bit_identical_layout_pairs: none")
    print(f"raw_csv: {raw_path}")
    print(f"summary_csv: {summary_path}")


def main() -> int:
    args = parse_args()
    args.binary = args.binary.resolve()
    args.schemes_dir = args.schemes_dir.resolve()
    args.registry = args.registry.resolve()
    args.output_prefix = args.output_prefix.resolve()
    os.environ.pop("ARCH_NAME", None)

    if not args.binary.is_file():
        print(f"error: benchmark binary not found: {args.binary}", file=sys.stderr)
        return 1
    try:
        all_wavelets = load_supported_wavelets(args.schemes_dir, args.registry)
        wavelets = selected_wavelets(all_wavelets, args.wavelets)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    transforms = transforms_from_arg(args.transform)
    raw_path = Path(str(args.output_prefix) + "_raw.csv")
    summary_path = Path(str(args.output_prefix) + "_summary.csv")
    try:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_handle = raw_path.open("w", encoding="utf-8", newline="")
        summary_handle = summary_path.open("w", encoding="utf-8", newline="")
    except OSError as exc:
        print(f"error: cannot open benchmark CSV outputs: {exc}", file=sys.stderr)
        return 1

    raw_cases: list[LayoutCase] = []
    summary_rows: list[dict[str, object]] = []
    non_identical_pairs: list[str] = []
    architecture_failed = False
    raw_writer = csv.DictWriter(raw_handle, fieldnames=RAW_FIELDS)
    summary_writer = csv.DictWriter(summary_handle, fieldnames=SUMMARY_FIELDS)
    raw_writer.writeheader()
    summary_writer.writeheader()

    group_total = len(transforms) * len(wavelets) * len(args.lengths)
    group_index = 0
    try:
        with tempfile.TemporaryDirectory(prefix="ttwv-layout-benchmark-") as temporary:
            temporary_path = Path(temporary)
            signal_path = temporary_path / "signal.txt"
            approximation_path = temporary_path / "approximation.txt"
            detail_path = temporary_path / "detail.txt"

            for signal_length in args.lengths:
                signal = make_benchmark_signal(signal_length)
                write_fp32_text(signal_path, signal)
                for wavelet in wavelets:
                    approximation, detail = (
                        values.astype(np.float32)
                        for values in pywt.dwt(signal, wavelet, mode=args.boundary_mode)
                    )
                    write_fp32_text(approximation_path, approximation)
                    write_fp32_text(detail_path, detail)
                    lwt_reference = np.concatenate(
                        [approximation.astype(np.float64), detail.astype(np.float64)]
                    )
                    ilwt_reference = pywt.idwt(
                        approximation, detail, wavelet, mode=args.boundary_mode
                    )[:signal_length].astype(np.float64)

                    for transform in transforms:
                        group_index += 1
                        reference = (
                            lwt_reference if transform == "lwt" else ilwt_reference
                        )
                        cases: list[LayoutCase] = []
                        for layout in LAYOUTS:
                            print(
                                f"[{group_index}/{group_total}] {transform} {wavelet} "
                                f"N={signal_length} layout={layout}",
                                flush=True,
                            )
                            case = run_layout_case(
                                args=args,
                                transform=transform,
                                wavelet=wavelet,
                                signal_length=signal_length,
                                requested_layout=layout,
                                signal_path=signal_path,
                                approximation_path=approximation_path,
                                detail_path=detail_path,
                                reference=reference,
                                temporary_path=temporary_path,
                            )
                            cases.append(case)
                            if case.architecture_failed:
                                architecture_failed = True
                                break

                        if len(cases) < len(LAYOUTS):
                            for layout in LAYOUTS[len(cases) :]:
                                skipped = LayoutCase(
                                    transform=transform,
                                    wavelet=wavelet,
                                    signal_length=signal_length,
                                    requested_layout=layout,
                                    expected_architecture=args.expected_architecture,
                                    warmup_runs=args.warmup_runs,
                                    timed_runs=args.timed_runs,
                                )
                                skipped.fail_timing(
                                    "skipped after architecture assertion failure"
                                )
                                cases.append(skipped)

                        comparisons = add_peer_comparisons(cases)
                        for comparison in comparisons:
                            if not comparison.bit_identical:
                                non_identical_pairs.append(
                                    f"{transform} {wavelet} N={signal_length} "
                                    f"{comparison.lhs}/{comparison.rhs} "
                                    f"max_abs={comparison.max_abs_difference:.8e} "
                                    f"first_mismatch={comparison.first_mismatch}"
                                )
                        summary_row = make_summary_row(
                            cases, comparisons, args.tie_threshold_percent
                        )
                        for case in cases:
                            raw_writer.writerow(raw_row(case))
                        summary_writer.writerow(summary_row)
                        raw_handle.flush()
                        summary_handle.flush()
                        raw_cases.extend(cases)
                        summary_rows.append(summary_row)
                        if architecture_failed:
                            break
                    if architecture_failed:
                        break
                if architecture_failed:
                    break
    finally:
        raw_handle.close()
        summary_handle.close()

    for transform in transforms:
        print_transform_summary(
            transform,
            raw_cases,
            summary_rows,
            non_identical_pairs,
            len(wavelets) * len(args.lengths) * len(LAYOUTS),
            raw_path,
            summary_path,
        )

    correctness_failed = any(case.correctness_failed for case in raw_cases)
    timing_failed = any(case.timing_failed for case in raw_cases)
    return 1 if architecture_failed or correctness_failed or timing_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
