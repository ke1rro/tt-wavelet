#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Runtime-args model extracted from tt-wavelet host runtime:
# - tt-wavelet/tt_wavelet/src/lifting/device.cpp
# - tt-wavelet/tt_wavelet/include/device_protocol/step_desc.hpp
DEFAULT_LIMIT = 341
STEP_COEFF_CAPACITY = 17
STEP_DESC_WORD_COUNT = 2 + STEP_COEFF_CAPACITY

PU_HEADER_WORDS = 1
PU_READER_ARGS_PER_STEP = 9
PU_COMPUTE_ARGS_PER_STEP = 1 + STEP_DESC_WORD_COUNT
PU_WRITER_ARGS_PER_STEP = 2

# Single-step scale program runtime args (constant sizes)
SCALE_READER_ARGS = 4
SCALE_COMPUTE_ARGS = 1
SCALE_WRITER_ARGS = 2

# Pad-split program runtime args (constant sizes)
PAD_SPLIT_READER_ARGS = 4
PAD_SPLIT_WRITER_ARGS = 4

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

VALID_STEP_TYPES = {"predict", "update", "scale-even", "scale-odd", "swap"}
PU_STEP_TYPES = {"predict", "update"}


@dataclass
class SchemeReport:
    name: str
    path: Path
    max_pu_segment_steps: int
    pu_reader_args: int
    pu_compute_args: int
    pu_writer_args: int
    max_runtime_args: int
    ok: bool
    reasons: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check runtime-args capacity per lifting scheme for TT-wavelet and colorize output."
        )
    )
    parser.add_argument(
        "--schemes-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "ttnn-wavelet" / "lifting_schemes",
        help="Directory with lifting scheme JSON files (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Runtime-args limit in words (default: %(default)s)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in output.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def max_predict_update_segment_steps(steps: list[dict[str, Any]]) -> int:
    max_count = 0
    current_count = 0
    for step in steps:
        step_type = str(step.get("type", "")).strip()
        if step_type in PU_STEP_TYPES:
            current_count += 1
            if current_count > max_count:
                max_count = current_count
            continue

        # swap is a logical remap and does not break the PU segment.
        if step_type == "swap":
            continue

        # scale and any other non-PU step terminate current PU segment.
        current_count = 0

    return max_count


def collect_report(path: Path, limit: int) -> SchemeReport:
    obj = load_json(path)
    name = path.stem
    steps = obj.get("steps", [])

    reasons: list[str] = []
    if not isinstance(steps, list) or len(steps) == 0:
        reasons.append("missing or empty steps")
        steps = []

    tap_size = obj.get("tap_size", 0)
    if not isinstance(tap_size, int) or tap_size <= 0:
        reasons.append("tap_size must be positive")

    coeff_sizes: list[int] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            reasons.append(f"step {i}: invalid entry")
            continue
        step_type = str(step.get("type", "")).strip()
        if step_type not in VALID_STEP_TYPES:
            reasons.append(f"step {i}: unsupported type '{step_type}'")

        coeffs = step.get("coefficients", [])
        if not isinstance(coeffs, list):
            reasons.append(f"step {i}: coefficients must be a list")
            coeff_count = 0
        else:
            coeff_count = len(coeffs)
        coeff_sizes.append(coeff_count)

        if coeff_count > STEP_COEFF_CAPACITY:
            reasons.append(
                f"step {i}: coefficients {coeff_count} exceed capacity {STEP_COEFF_CAPACITY}"
            )

    max_pu_segment = max_predict_update_segment_steps(steps)

    pu_reader_args = PU_HEADER_WORDS + max_pu_segment * PU_READER_ARGS_PER_STEP
    pu_compute_args = PU_HEADER_WORDS + max_pu_segment * PU_COMPUTE_ARGS_PER_STEP
    pu_writer_args = PU_HEADER_WORDS + max_pu_segment * PU_WRITER_ARGS_PER_STEP

    max_runtime_args = max(
        pu_reader_args,
        pu_compute_args,
        pu_writer_args,
        PAD_SPLIT_READER_ARGS,
        PAD_SPLIT_WRITER_ARGS,
        SCALE_READER_ARGS,
        SCALE_COMPUTE_ARGS,
        SCALE_WRITER_ARGS,
    )

    if pu_reader_args > limit:
        reasons.append(f"predict/update reader args {pu_reader_args} > limit {limit}")
    if pu_compute_args > limit:
        reasons.append(f"predict/update compute args {pu_compute_args} > limit {limit}")
    if pu_writer_args > limit:
        reasons.append(f"predict/update writer args {pu_writer_args} > limit {limit}")

    ok = len(reasons) == 0
    return SchemeReport(
        name=name,
        path=path,
        max_pu_segment_steps=max_pu_segment,
        pu_reader_args=pu_reader_args,
        pu_compute_args=pu_compute_args,
        pu_writer_args=pu_writer_args,
        max_runtime_args=max_runtime_args,
        ok=ok,
        reasons=reasons,
    )


def colorize(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{color}{text}{RESET}"


def main() -> int:
    args = parse_args()
    schemes_dir: Path = args.schemes_dir

    if not schemes_dir.exists() or not schemes_dir.is_dir():
        raise FileNotFoundError(f"Schemes directory not found: {schemes_dir}")

    json_files = sorted(schemes_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON schemes found in: {schemes_dir}")

    reports = [collect_report(path, args.limit) for path in json_files]

    print(f"Runtime args limit: {args.limit}")
    print(f"Schemes directory: {schemes_dir}")
    print(
        "Model: PU reader=1+9*N, PU compute=1+20*N, PU writer=1+2*N, "
        "scale=(4/1/2), pad_split=(4/4)"
    )
    print()

    ok_reports = [report for report in reports if report.ok]
    fail_reports = [report for report in reports if not report.ok]

    for report in ok_reports + fail_reports:
        status = "OK" if report.ok else "FAIL"
        color = GREEN if report.ok else RED
        header = (
            f"[{status:4}] {report.path.name:<12} "
            f"pu_max_seg={report.max_pu_segment_steps:<2} "
            f"reader={report.pu_reader_args:<3} "
            f"compute={report.pu_compute_args:<3} "
            f"writer={report.pu_writer_args:<3} "
            f"max={report.max_runtime_args:<3}"
        )
        print(colorize(header, color, not args.no_color))
        if not report.ok:
            for reason in report.reasons:
                print(colorize(f"  - {reason}", RED, not args.no_color))

    ok_count = sum(1 for report in reports if report.ok)
    fail_count = len(reports) - ok_count
    print()
    print(f"Total schemes: {len(reports)}")
    print(colorize(f"Runnable (green): {ok_count}", GREEN, not args.no_color))
    print(colorize(f"Not runnable (red): {fail_count}", RED, not args.no_color))

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
