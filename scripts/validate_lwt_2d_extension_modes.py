#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate every 2D scheme/mode and emit the extension-mode result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pywt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEME_DIR = PROJECT_ROOT / "wavelets"
LWT = PROJECT_ROOT / "build" / "lwt_2d"
ILWT = PROJECT_ROOT / "build" / "ilwt_2d"
SET_ENV = PROJECT_ROOT / "scripts" / "set_env.sh"
DOCS = PROJECT_ROOT / "docs"

BANDS = ("LL", "LH", "HL", "HH")
MODES = (
    "zero",
    "constant",
    "symmetric",
    "reflect",
    "periodic",
    "smooth",
    "antisymmetric",
    "antireflect",
)
SHAPES = (
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (3, 5),
    (5, 3),
    (7, 9),
    (8, 8),
    (31, 33),
    (32, 32),
    (33, 31),
    (64, 65),
    (65, 64),
)
INPUT_TYPES = (
    "ramp",
    "alternating",
    "boundary-impulse",
    "constant",
    "random",
    "large-finite",
)
CORRECTNESS_FIELDS = (
    "transform",
    "scheme",
    "mode",
    "height",
    "width",
    "input_type",
    "layout",
    "architecture",
    "status",
    "max_abs_error",
    "max_rel_error",
    "worst_band",
    "worst_row",
    "worst_col",
    "pywavelets_status",
    "tolerance",
    "error",
)
ROUNDTRIP_FIELDS = (
    "transform",
    "scheme",
    "mode",
    "height",
    "width",
    "input_type",
    "layout",
    "architecture",
    "status",
    "max_abs_error",
    "max_rel_error",
    "worst_band",
    "worst_row",
    "worst_col",
    "tolerance",
    "error",
)


def parse_csv_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schemes",
        help="Comma-separated scheme subset (default: all 106 generated schemes).",
    )
    parser.add_argument(
        "--modes",
        default=",".join(MODES),
        help="Comma-separated signal-extension modes (default: all eight).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Run only the first N scheme/mode pairs (for smoke testing).",
    )
    parser.add_argument(
        "--case-numbers",
        help="Comma-separated one-based case numbers from the full selected matrix.",
    )
    parser.add_argument(
        "--shape",
        help="Use one HEIGHTxWIDTH shape for every selected case (smoke tests).",
    )
    parser.add_argument(
        "--input-type",
        choices=INPUT_TYPES,
        help="Use one input pattern for every selected case (smoke tests).",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Replace matching scheme/mode rows in existing output artifacts.",
    )
    parser.add_argument("--cores", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--architecture",
        choices=("blackhole", "wormhole"),
        default="blackhole",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DOCS,
    )
    return parser.parse_args()


def run_device(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    quoted = " ".join(shlex.quote(item) for item in command)
    shell_command = (
        f"source {shlex.quote(str(SET_ENV))} && "
        "export TT_LOGGER_LEVEL=FATAL && "
        f"{quoted}"
    )
    completed = subprocess.run(
        ["bash", "-lc", shell_command],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}{completed.stderr}"
        )
    return completed


def make_input(
    height: int,
    width: int,
    input_type: str,
    seed: int,
) -> np.ndarray:
    row, column = np.indices((height, width))
    if input_type == "ramp":
        return np.linspace(
            -1.0, 1.0, num=height * width, dtype=np.float32
        ).reshape(height, width)
    if input_type == "alternating":
        magnitude = 0.25 + (row * width + column) / max(1, height * width)
        return np.where((row + column) & 1, -magnitude, magnitude).astype(np.float32)
    if input_type == "constant":
        return np.full((height, width), np.float32(0.375), dtype=np.float32)
    if input_type == "random":
        return np.random.default_rng(seed).uniform(
            -1.0, 1.0, size=(height, width)
        ).astype(np.float32)
    if input_type == "large-finite":
        values = np.random.default_rng(seed).uniform(
            -4096.0, 4096.0, size=(height, width)
        )
        return values.astype(np.float32)
    result = np.zeros((height, width), dtype=np.float32)
    impulses = (
        (0, 0, 1.0),
        (0, width - 1, -0.75),
        (height - 1, 0, 0.5),
        (height - 1, width - 1, -0.25),
    )
    for impulse_row, impulse_column, value in impulses:
        result[impulse_row, impulse_column] = np.float32(value)
    return result


def read_device_bands(prefix: Path) -> dict[str, np.ndarray]:
    return {
        band: np.fromfile(Path(f"{prefix}_{band}.f32"), dtype=np.float32)
        for band in BANDS
    }


def run_lwt(
    scheme: str,
    mode: str,
    height: int,
    width: int,
    signal_path: Path,
    prefix: Path,
    cores: int,
    timeout: float,
) -> dict[str, np.ndarray]:
    run_device(
        [
            str(LWT),
            "--binary-input",
            "--quiet",
            "--boundary-mode",
            mode,
            "--cores",
            str(cores),
            "--output-prefix",
            str(prefix),
            scheme,
            str(height),
            str(width),
            str(signal_path),
        ],
        timeout,
    )
    return read_device_bands(prefix)


def run_ilwt(
    scheme: str,
    mode: str,
    height: int,
    width: int,
    band_prefix: Path,
    output_path: Path,
    cores: int,
    timeout: float,
) -> np.ndarray:
    run_device(
        [
            str(ILWT),
            "--boundary-mode",
            mode,
            "--cores",
            str(cores),
            "--output",
            str(output_path),
            scheme,
            str(height),
            str(width),
            *(str(Path(f"{band_prefix}_{band}.f32")) for band in BANDS),
        ],
        timeout,
    )
    return np.fromfile(output_path, dtype=np.float32)


def pywavelets_bands(
    signal: np.ndarray,
    scheme: str,
    mode: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    approximation, (horizontal, vertical, diagonal) = pywt.dwt2(
        signal, scheme, mode=mode
    )
    # Project convention: first band letter is the vertical transform.
    bands = {
        "LL": np.asarray(approximation, dtype=np.float32),
        "LH": np.asarray(vertical, dtype=np.float32),
        "HL": np.asarray(horizontal, dtype=np.float32),
        "HH": np.asarray(diagonal, dtype=np.float32),
    }
    reconstructed = pywt.idwt2(
        (approximation, (horizontal, vertical, diagonal)),
        scheme,
        mode=mode,
    )[: signal.shape[0], : signal.shape[1]]
    return bands, np.asarray(reconstructed, dtype=np.float32)


def write_bands(prefix: Path, bands: dict[str, np.ndarray]) -> None:
    for band, values in bands.items():
        np.asarray(values, dtype=np.float32).tofile(Path(f"{prefix}_{band}.f32"))


def error_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float, int]:
    candidate_flat = np.asarray(candidate, dtype=np.float32).reshape(-1)
    reference_flat = np.asarray(reference, dtype=np.float32).reshape(-1)
    if candidate_flat.shape != reference_flat.shape:
        return float("inf"), float("inf"), -1
    if candidate_flat.size == 0:
        return 0.0, 0.0, 0
    difference = np.abs(
        candidate_flat.astype(np.float64) - reference_flat.astype(np.float64)
    )
    worst = int(np.argmax(difference))
    denominator = np.maximum(np.abs(reference_flat.astype(np.float64)), 1.0e-30)
    relative = difference / denominator
    return float(difference[worst]), float(np.max(relative)), worst


def band_error_metrics(
    candidate: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> tuple[float, float, str, int, int]:
    worst_abs = -1.0
    worst_rel = 0.0
    worst_band = ""
    worst_row = -1
    worst_column = -1
    for band in BANDS:
        max_abs, max_rel, flat_index = error_metrics(candidate[band], reference[band])
        if max_abs > worst_abs:
            worst_abs = max_abs
            worst_rel = max_rel
            worst_band = band
            if flat_index >= 0 and np.asarray(reference[band]).ndim == 2:
                worst_row, worst_column = np.unravel_index(
                    flat_index, np.asarray(reference[band]).shape
                )
                worst_row = int(worst_row)
                worst_column = int(worst_column)
            else:
                worst_row, worst_column = -1, flat_index
    return worst_abs, worst_rel, worst_band, worst_row, worst_column


def tensor_error_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float, int, int]:
    max_abs, max_rel, flat_index = error_metrics(candidate, reference)
    if flat_index < 0:
        return max_abs, max_rel, -1, -1
    row, column = np.unravel_index(flat_index, np.asarray(reference).shape)
    return max_abs, max_rel, int(row), int(column)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mode_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        statuses = Counter(str(row["status"]) for row in selected)
        finite_errors = [
            float(row["max_abs_error"])
            for row in selected
            if np.isfinite(float(row["max_abs_error"]))
        ]
        summary[mode] = {
            "case_count": len(selected),
            "status_counts": dict(statuses),
            "max_abs_error": max(finite_errors, default=None),
        }
    return summary


def write_json(path: Path, rows: list[dict[str, Any]], tolerance: float) -> None:
    payload = {
        "schema_version": 1,
        "tolerance": tolerance,
        "band_convention": "first letter is vertical; PyWavelets cV=LH and cH=HL",
        "mode_summary": mode_summary(rows),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

def merge_rows(
    existing: list[dict[str, Any]], updates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    replacements = {(row["scheme"], row["mode"]): row for row in updates}
    merged: list[dict[str, Any]] = []
    for row in existing:
        key = (row["scheme"], row["mode"])
        merged.append(replacements.pop(key, row))
    merged.extend(replacements.values())
    return merged


def failure_row(
    transform: str,
    scheme: str,
    mode: str,
    height: int,
    width: int,
    input_type: str,
    architecture: str,
    tolerance: float,
    error: Exception,
) -> dict[str, Any]:
    finite_failure = float(np.finfo(np.float64).max)
    return {
        "transform": transform,
        "scheme": scheme,
        "mode": mode,
        "height": height,
        "width": width,
        "input_type": input_type,
        "layout": "tile-native",
        "architecture": architecture,
        "status": "fail",
        "max_abs_error": finite_failure,
        "max_rel_error": finite_failure,
        "worst_band": "",
        "worst_row": -1,
        "worst_col": -1,
        "pywavelets_status": "fail",
        "tolerance": tolerance,
        "error": str(error),
    }


def main() -> int:
    args = parse_args()
    if args.cores <= 0:
        raise ValueError("--cores must be positive")
    if args.tolerance <= 0:
        raise ValueError("--tolerance must be positive")
    for binary in (LWT, ILWT):
        if not binary.is_file():
            raise FileNotFoundError(f"required binary is missing: {binary}")
    fixed_shape: tuple[int, int] | None = None
    if args.shape:
        dimensions = args.shape.lower().split("x")
        if len(dimensions) != 2:
            raise ValueError("--shape must use HEIGHTxWIDTH syntax")
        fixed_shape = (int(dimensions[0]), int(dimensions[1]))
        if min(fixed_shape) <= 0:
            raise ValueError("--shape dimensions must be positive")

    schemes = (
        parse_csv_values(args.schemes)
        if args.schemes
        else sorted(path.stem for path in SCHEME_DIR.glob("*.json"))
    )
    modes = parse_csv_values(args.modes)
    unsupported_modes = sorted(set(modes) - set(MODES))
    if unsupported_modes:
        raise ValueError(f"unsupported modes: {', '.join(unsupported_modes)}")
    if len(schemes) == 0 or len(modes) == 0:
        raise ValueError("scheme and mode lists must not be empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lwt_rows: list[dict[str, Any]] = []
    ilwt_rows: list[dict[str, Any]] = []
    roundtrip_rows: list[dict[str, Any]] = []
    case_pairs = [(scheme, mode) for scheme in schemes for mode in modes]
    if args.case_numbers:
        requested_numbers = [
            int(value) for value in parse_csv_values(args.case_numbers)
        ]
        invalid_numbers = [
            number
            for number in requested_numbers
            if number < 1 or number > len(case_pairs)
        ]
        if invalid_numbers:
            raise ValueError(
                f"case numbers outside 1..{len(case_pairs)}: {invalid_numbers}"
            )
        case_pairs = [case_pairs[number - 1] for number in requested_numbers]
    if args.max_cases is not None:
        case_pairs = case_pairs[: args.max_cases]

    with tempfile.TemporaryDirectory(prefix="ttwv-extension-matrix-") as temporary:
        root = Path(temporary)
        for case_index, (scheme, mode) in enumerate(case_pairs):
            scheme_index = schemes.index(scheme)
            mode_index = MODES.index(mode)
            shape_index = (3 * scheme_index + mode_index) % len(SHAPES)
            height, width = fixed_shape or SHAPES[shape_index]
            if mode in {"reflect", "antireflect"} and min(height, width) <= 1:
                if fixed_shape is not None:
                    raise ValueError(
                        "reflect and antireflect require both dimensions greater than one"
                    )
                valid_shapes = [shape for shape in SHAPES if min(shape) > 1]
                height, width = valid_shapes[(scheme_index + mode_index) % len(valid_shapes)]
            input_type = args.input_type or INPUT_TYPES[
                (scheme_index + mode_index) % len(INPUT_TYPES)
            ]
            signal = make_input(
                height,
                width,
                input_type,
                args.seed + 1000 * scheme_index + mode_index,
            )
            case_root = root / f"{case_index:04d}_{scheme}_{mode}"
            case_root.mkdir()
            signal_path = case_root / "signal.f32"
            signal.tofile(signal_path)
            lwt_prefix = case_root / "tt"
            pywt_prefix = case_root / "pywt"
            roundtrip_path = case_root / "roundtrip.f32"
            ilwt_path = case_root / "pywt_ilwt.f32"
            lwt_row_count = len(lwt_rows)
            ilwt_row_count = len(ilwt_rows)
            roundtrip_row_count = len(roundtrip_rows)

            try:
                pywt_bands, pywt_reconstruction = pywavelets_bands(
                    signal, scheme, mode
                )
                device_bands = run_lwt(
                    scheme,
                    mode,
                    height,
                    width,
                    signal_path,
                    lwt_prefix,
                    args.cores,
                    args.timeout_seconds,
                )
                pywt_metrics = band_error_metrics(device_bands, pywt_bands)
                pywt_ok = pywt_metrics[0] <= args.tolerance
                lwt_rows.append(
                    {
                        "transform": "lwt_2d",
                        "scheme": scheme,
                        "mode": mode,
                        "height": height,
                        "width": width,
                        "input_type": input_type,
                        "layout": "tile-native",
                        "architecture": args.architecture,
                        "status": "pass" if pywt_ok else "fail",
                        "max_abs_error": pywt_metrics[0],
                        "max_rel_error": pywt_metrics[1],
                        "worst_band": pywt_metrics[2],
                        "worst_row": pywt_metrics[3],
                        "worst_col": pywt_metrics[4],
                        "pywavelets_status": "pass" if pywt_ok else "fail",
                        "tolerance": args.tolerance,
                        "error": "",
                    }
                )

                roundtrip = run_ilwt(
                    scheme,
                    mode,
                    height,
                    width,
                    lwt_prefix,
                    roundtrip_path,
                    args.cores,
                    args.timeout_seconds,
                ).reshape(height, width)
                roundtrip_metrics = tensor_error_metrics(roundtrip, signal)
                roundtrip_rows.append(
                    {
                        "transform": "lwt_2d_ilwt_2d",
                        "scheme": scheme,
                        "mode": mode,
                        "height": height,
                        "width": width,
                        "input_type": input_type,
                        "layout": "tile-native",
                        "architecture": args.architecture,
                        "status": (
                            "pass"
                            if roundtrip_metrics[0] <= args.tolerance
                            else "fail"
                        ),
                        "max_abs_error": roundtrip_metrics[0],
                        "max_rel_error": roundtrip_metrics[1],
                        "worst_band": "reconstruction",
                        "worst_row": roundtrip_metrics[2],
                        "worst_col": roundtrip_metrics[3],
                        "tolerance": args.tolerance,
                        "error": "",
                    }
                )

                write_bands(pywt_prefix, pywt_bands)
                pywt_ilwt = run_ilwt(
                    scheme,
                    mode,
                    height,
                    width,
                    pywt_prefix,
                    ilwt_path,
                    args.cores,
                    args.timeout_seconds,
                ).reshape(height, width)
                ilwt_metrics = tensor_error_metrics(pywt_ilwt, pywt_reconstruction)
                ilwt_rows.append(
                    {
                        "transform": "ilwt_2d",
                        "scheme": scheme,
                        "mode": mode,
                        "height": height,
                        "width": width,
                        "input_type": "pywavelets-coefficients",
                        "layout": "tile-native",
                        "architecture": args.architecture,
                        "status": (
                            "pass" if ilwt_metrics[0] <= args.tolerance else "fail"
                        ),
                        "max_abs_error": ilwt_metrics[0],
                        "max_rel_error": ilwt_metrics[1],
                        "worst_band": "reconstruction",
                        "worst_row": ilwt_metrics[2],
                        "worst_col": ilwt_metrics[3],
                        "pywavelets_status": (
                            "pass" if ilwt_metrics[0] <= args.tolerance else "fail"
                        ),
                        "tolerance": args.tolerance,
                        "error": "",
                    }
                )
            except Exception as error:  # noqa: BLE001
                if len(lwt_rows) == lwt_row_count:
                    lwt_rows.append(
                        failure_row(
                            "lwt_2d",
                            scheme,
                            mode,
                            height,
                            width,
                            input_type,
                            args.architecture,
                            args.tolerance,
                            error,
                        )
                    )
                if len(ilwt_rows) == ilwt_row_count:
                    ilwt_rows.append(
                        failure_row(
                            "ilwt_2d",
                            scheme,
                            mode,
                            height,
                            width,
                            "pywavelets-coefficients",
                            args.architecture,
                            args.tolerance,
                            error,
                        )
                    )
                if len(roundtrip_rows) == roundtrip_row_count:
                    roundtrip_rows.append(
                        failure_row(
                            "lwt_2d_ilwt_2d",
                            scheme,
                            mode,
                            height,
                            width,
                            input_type,
                            args.architecture,
                            args.tolerance,
                            error,
                        )
                    )
                if args.fail_fast:
                    raise

            print(
                f"[{case_index + 1}/{len(case_pairs)}] {scheme} {mode} "
                f"{height}x{width} "
                f"LWT={lwt_rows[-1]['status']} "
                f"ILWT={ilwt_rows[-1]['status']} "
                f"roundtrip={roundtrip_rows[-1]['status']}",
                flush=True,
            )

    lwt_csv = args.output_dir / "lwt_2d_extension_modes_correctness.csv"
    lwt_json = args.output_dir / "lwt_2d_extension_modes_correctness.json"
    ilwt_csv = args.output_dir / "ilwt_2d_extension_modes_correctness.csv"
    ilwt_json = args.output_dir / "ilwt_2d_extension_modes_correctness.json"
    roundtrip_csv = args.output_dir / "lwt_2d_extension_modes_roundtrip.csv"
    if args.merge_existing:
        if lwt_json.is_file():
            lwt_rows = merge_rows(
                json.loads(lwt_json.read_text(encoding="utf-8"))["rows"], lwt_rows
            )
        if ilwt_json.is_file():
            ilwt_rows = merge_rows(
                json.loads(ilwt_json.read_text(encoding="utf-8"))["rows"], ilwt_rows
            )
        if roundtrip_csv.is_file():
            with roundtrip_csv.open(newline="", encoding="utf-8") as file:
                roundtrip_rows = merge_rows(
                    list(csv.DictReader(file)), roundtrip_rows
                )
    write_csv(lwt_csv, lwt_rows, CORRECTNESS_FIELDS)
    write_json(lwt_json, lwt_rows, args.tolerance)
    write_csv(ilwt_csv, ilwt_rows, CORRECTNESS_FIELDS)
    write_json(ilwt_json, ilwt_rows, args.tolerance)
    write_csv(roundtrip_csv, roundtrip_rows, ROUNDTRIP_FIELDS)

    failures = sum(row["status"] != "pass" for row in lwt_rows)
    failures += sum(row["status"] != "pass" for row in ilwt_rows)
    failures += sum(row["status"] != "pass" for row in roundtrip_rows)
    print(
        f"wrote {len(lwt_rows)} LWT, {len(ilwt_rows)} ILWT, and "
        f"{len(roundtrip_rows)} round-trip rows; failures={failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
