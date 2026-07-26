#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Classify generated lifting schemes against symmetric PyWavelets DWT2.

The C++ scalar vertical-first FP32 implementation remains the internal oracle.
This tool only determines which generated schemes are externally stable against
PyWavelets at the configured absolute-error threshold.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pywt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = PROJECT_ROOT / "build" / "lwt_2d_reference"
DEFAULT_SCHEME_DIR = PROJECT_ROOT / "wavelets"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "tests" / "reference" / "lwt_2d_symmetric_stable_schemes.json"
)
DEFAULT_SHAPES = (
    (1, 1),
    (1, 7),
    (7, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (5, 7),
    (15, 17),
    (17, 15),
    (31, 31),
    (31, 32),
    (32, 31),
    (32, 32),
    (32, 33),
    (33, 32),
    (33, 33),
    (63, 65),
    (64, 64),
    (65, 63),
    (65, 97),
    (127, 129),
    (256, 256),
    (513, 769),
)
DEFAULT_INPUT_TYPES = ("random", "ramp", "checkerboard", "impulse_corner")
BANDS = ("LL", "LH", "HL", "HH")
SHAPE_PATTERN = re.compile(r"^(\d+)[xX](\d+)$")
OUTPUT_SHAPE_PATTERN = re.compile(r"lwt_2d_reference_output_shape: (\d+)x(\d+)")


@dataclass(frozen=True)
class Case:
    index: int
    wavelet: str
    height: int
    width: int
    input_type: str
    seed: int | None


def parse_shape(text: str) -> tuple[int, int]:
    match = SHAPE_PATTERN.fullmatch(text)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"invalid shape '{text}'; expected HEIGHTxWIDTH"
        )
    height, width = (int(match.group(1)), int(match.group(2)))
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return height, width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-binary", type=Path, default=DEFAULT_REFERENCE
    )
    parser.add_argument("--scheme-dir", type=Path, default=DEFAULT_SCHEME_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--wavelets",
        nargs="+",
        help="Classify only these schemes (default: every JSON scheme).",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=parse_shape,
        default=list(DEFAULT_SHAPES),
        metavar="HEIGHTxWIDTH",
    )
    parser.add_argument(
        "--input-types",
        nargs="+",
        choices=DEFAULT_INPUT_TYPES,
        default=list(DEFAULT_INPUT_TYPES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--fp32-arithmetic",
        choices=("ieee", "wormhole-sfpu", "blackhole-sfpu"),
        default="ieee",
        help=(
            "Arithmetic model used by the lifting reference. The checked-in "
            "PyWavelets compatibility manifest uses IEEE arithmetic "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Concurrent reference processes (default: %(default)s).",
    )
    return parser.parse_args()


def generated_wavelets(scheme_dir: Path) -> list[str]:
    return sorted(path.stem for path in scheme_dir.glob("*.json"))


def make_input(case: Case) -> np.ndarray:
    shape = (case.height, case.width)
    element_count = case.height * case.width
    if case.input_type == "random":
        if case.seed is None:
            raise ValueError("random classification case requires a seed")
        generator = np.random.default_rng(case.seed)
        return generator.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    if case.input_type == "ramp":
        return np.linspace(-1.0, 1.0, num=element_count, dtype=np.float32).reshape(
            shape
        )
    if case.input_type == "checkerboard":
        rows, columns = np.indices(shape)
        return np.where((rows + columns) % 2 == 0, 1.0, -1.0).astype(np.float32)
    if case.input_type == "impulse_corner":
        values = np.zeros(shape, dtype=np.float32)
        values[0, 0] = np.float32(1.0)
        return values
    raise ValueError(f"unsupported input type: {case.input_type}")


def pywavelets_bands(
    values: np.ndarray, wavelet: str
) -> dict[str, np.ndarray]:
    approximation, (horizontal, vertical, diagonal) = pywt.dwt2(
        values, wavelet, mode="symmetric"
    )
    # TT-wavelet names the first letter by the vertical result.
    return {
        "LL": np.asarray(approximation),
        "LH": np.asarray(vertical),
        "HL": np.asarray(horizontal),
        "HH": np.asarray(diagonal),
    }


def read_oracle_bands(
    prefix: Path, shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    element_count = shape[0] * shape[1]
    bands: dict[str, np.ndarray] = {}
    for band in BANDS:
        path = Path(f"{prefix}.{band.lower()}.f32")
        values = np.fromfile(path, dtype="<f4")
        path.unlink()
        if values.size != element_count:
            raise RuntimeError(
                f"{band} oracle emitted {values.size} values for shape "
                f"{shape[0]}x{shape[1]}"
            )
        bands[band] = values.reshape(shape)
    return bands


def finite_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def compare_case(
    case: Case,
    reference_binary: Path,
    temporary_root: Path,
    fp32_arithmetic: str,
) -> dict[str, Any]:
    values = make_input(case)
    input_path = temporary_root / f"case_{case.index}.input.f32"
    prefix = temporary_root / f"case_{case.index}.output"
    values.astype("<f4", copy=False).tofile(input_path)
    command = [
        str(reference_binary),
        "--boundary-mode",
        "symmetric",
        "--fp32-arithmetic",
        fp32_arithmetic,
        "--binary-input",
        "--output-prefix",
        str(prefix),
        case.wavelet,
        str(case.height),
        str(case.width),
        str(input_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"reference exited {completed.returncode}: {completed.stderr.strip()}"
            )
        shape_match = OUTPUT_SHAPE_PATTERN.search(completed.stdout)
        if shape_match is None:
            raise RuntimeError("reference did not report its output shape")
        band_shape = (int(shape_match.group(1)), int(shape_match.group(2)))
        oracle = read_oracle_bands(prefix, band_shape)
        expected = pywavelets_bands(values, case.wavelet)
    finally:
        input_path.unlink(missing_ok=True)

    worst: dict[str, Any] | None = None
    max_relative_error = 0.0
    for band in BANDS:
        if expected[band].shape != oracle[band].shape:
            raise RuntimeError(
                f"{band} shape mismatch: oracle={oracle[band].shape}, "
                f"PyWavelets={expected[band].shape}"
            )
        oracle_values = oracle[band].astype(np.float64)
        expected_values = expected[band].astype(np.float64)
        finite = np.isfinite(oracle_values) & np.isfinite(expected_values)
        if not np.all(finite):
            first = int(np.flatnonzero(~finite)[0])
            coordinate = [
                int(value) for value in np.unravel_index(first, oracle_values.shape)
            ]
            candidate = {
                "band": band,
                "max_absolute_error": None,
                "flat_index": first,
                "index": coordinate,
                "oracle_value": finite_float(oracle_values.flat[first]),
                "pywavelets_value": finite_float(expected_values.flat[first]),
                "non_finite": True,
            }
            worst = candidate
            max_relative_error = float("inf")
            break

        absolute = np.abs(oracle_values - expected_values)
        relative = absolute / np.maximum(np.abs(expected_values), 1.0e-30)
        flat_index = int(np.argmax(absolute))
        candidate = {
            "band": band,
            "max_absolute_error": float(absolute.flat[flat_index]),
            "flat_index": flat_index,
            "index": [
                int(value)
                for value in np.unravel_index(flat_index, absolute.shape)
            ],
            "oracle_value": float(oracle_values.flat[flat_index]),
            "pywavelets_value": float(expected_values.flat[flat_index]),
            "non_finite": False,
        }
        max_relative_error = max(max_relative_error, float(np.max(relative)))
        if worst is None or candidate["max_absolute_error"] > worst[
            "max_absolute_error"
        ]:
            worst = candidate

    assert worst is not None
    return {
        "wavelet": case.wavelet,
        "shape": [case.height, case.width],
        "input_type": case.input_type,
        "seed": case.seed,
        "max_relative_error": finite_float(max_relative_error),
        **worst,
    }


def cases_for(
    wavelets: list[str],
    shapes: list[tuple[int, int]],
    input_types: list[str],
    seeds: list[int],
) -> list[Case]:
    cases: list[Case] = []
    for wavelet in wavelets:
        for height, width in shapes:
            for input_type in input_types:
                case_seeds: list[int | None] = seeds if input_type == "random" else [None]
                for seed in case_seeds:
                    cases.append(
                        Case(
                            index=len(cases),
                            wavelet=wavelet,
                            height=height,
                            width=width,
                            input_type=input_type,
                            seed=seed,
                        )
                    )
    return cases


def error_result(case: Case, error: BaseException) -> dict[str, Any]:
    return {
        "wavelet": case.wavelet,
        "shape": [case.height, case.width],
        "input_type": case.input_type,
        "seed": case.seed,
        "error": str(error),
        "max_absolute_error": None,
        "max_relative_error": None,
        "non_finite": False,
    }


def is_worse(candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
    if current is None:
        return True
    if candidate.get("error") is not None or candidate.get("non_finite"):
        return current.get("error") is None and not current.get("non_finite")
    if current.get("error") is not None or current.get("non_finite"):
        return False
    return float(candidate["max_absolute_error"]) > float(
        current["max_absolute_error"]
    )


def classify(args: argparse.Namespace) -> dict[str, Any]:
    if args.tolerance <= 0:
        raise ValueError("--tolerance must be positive")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if not args.reference_binary.is_file():
        raise FileNotFoundError(
            f"2D FP32 reference binary not found: {args.reference_binary}"
        )
    available = set(generated_wavelets(args.scheme_dir))
    wavelets = sorted(args.wavelets) if args.wavelets else sorted(available)
    unknown = sorted(set(wavelets) - available)
    if unknown:
        raise ValueError(f"unknown generated wavelets: {', '.join(unknown)}")
    cases = cases_for(wavelets, args.shapes, args.input_types, args.seeds)
    per_scheme: dict[str, dict[str, Any]] = {
        wavelet: {"status": "pass", "worst": None, "case_count": 0}
        for wavelet in wavelets
    }

    with tempfile.TemporaryDirectory(
        prefix="ttwv_lwt_2d_classify_"
    ) as temporary:
        temporary_root = Path(temporary)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_cases = {
                executor.submit(
                    compare_case,
                    case,
                    args.reference_binary,
                    temporary_root,
                    args.fp32_arithmetic,
                ): case
                for case in cases
            }
            completed_count = 0
            for future in concurrent.futures.as_completed(future_cases):
                case = future_cases[future]
                try:
                    result = future.result()
                except BaseException as error:  # Preserve every failed classification.
                    result = error_result(case, error)
                scheme = per_scheme[case.wavelet]
                scheme["case_count"] += 1
                if (
                    result.get("error") is not None
                    or result.get("non_finite")
                    or result["max_absolute_error"] > args.tolerance
                ):
                    scheme["status"] = "known_fp32_difference"
                if is_worse(result, scheme["worst"]):
                    scheme["worst"] = result
                completed_count += 1
                if completed_count == len(cases) or completed_count % 100 == 0:
                    print(
                        f"classified {completed_count}/{len(cases)} cases",
                        file=sys.stderr,
                    )

    stable = [
        wavelet
        for wavelet, result in per_scheme.items()
        if result["status"] == "pass"
    ]
    return {
        "schema_version": 1,
        "boundary_mode": "symmetric",
        "axis_order": "vertical-first",
        "band_convention": "first letter is vertical",
        "fp32_arithmetic": args.fp32_arithmetic,
        "absolute_tolerance": args.tolerance,
        "target_tolerance": 1.0e-5,
        "input_range": [-1.0, 1.0],
        "shapes": [list(shape) for shape in args.shapes],
        "input_types": args.input_types,
        "random_seeds": args.seeds,
        "numpy_version": np.__version__,
        "pywavelets_version": pywt.__version__,
        "reference_binary": str(args.reference_binary.resolve()),
        "stable_schemes": stable,
        "schemes": per_scheme,
    }


def main() -> int:
    args = parse_args()
    try:
        manifest = classify(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"classification failed: {error}", file=sys.stderr)
        return 1
    print(
        f"wrote {len(manifest['stable_schemes'])}/"
        f"{len(manifest['schemes'])} stable schemes to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
