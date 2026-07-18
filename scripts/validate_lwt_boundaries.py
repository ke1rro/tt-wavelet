#!/usr/bin/env python3
"""Validate ConeStreamed LWT boundary modes against PyWavelets."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pywt

DEFAULT_TOLERANCES = {
    "db1": 2.0e-6,
    "db2": 2.0e-5,
    "db7": 1.0e-3,
    "bior3.9": 2.0e-5,
}


def parse_coefficients(stdout: str, label: str) -> np.ndarray:
    prefix = f"tt-wavelet {label} coefficients"
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return np.fromstring(
                line.split("[", 1)[1].rsplit("]", 1)[0],
                sep=",",
                dtype=np.float64,
            )
    raise RuntimeError(f"lwt output did not contain {label} coefficients")


def make_signal(length: int) -> np.ndarray:
    index = np.arange(length, dtype=np.float32)
    return (0.7 * np.sin(index * 0.071) + 0.2 * np.cos(index * 0.013) + index * 1.0e-4).astype(
        np.float32
    )


def run_case(
    binary: Path,
    wavelet: str,
    mode: str,
    layout: str,
    signal_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    environment = os.environ.copy()
    environment["TT_LOGGER_LEVEL"] = "FATAL"
    environment["TT_WAVELET_LWT_CONE_WORKSPACE_LAYOUT"] = layout
    command = [
        str(binary),
        "--memory-mode",
        "cone",
        "--boundary-mode",
        mode,
        wavelet,
        str(signal_path),
    ]
    result = subprocess.run(command, text=True, capture_output=True, env=environment, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"LWT failed for {wavelet}, mode={mode}, layout={layout}:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return (
        parse_coefficients(result.stdout, "approximation"),
        parse_coefficients(result.stdout, "detail"),
    )


def max_abs_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    if candidate.shape != reference.shape:
        return float("inf")
    if candidate.size == 0:
        return 0.0
    return float(np.max(np.abs(candidate - reference)))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=root / "build" / "lwt")
    parser.add_argument("--schemes-dir", type=Path, default=root / "wavelets")
    parser.add_argument("--wavelets", nargs="+", default=["db1", "db2", "db7", "bior3.9"])
    parser.add_argument(
        "--all-schemes",
        action="store_true",
        help="Validate every JSON scheme under --schemes-dir.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
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
        default=[
            "symmetric",
            "zero",
            "constant",
            "periodic",
            "antisymmetric",
            "smooth",
            "reflect",
            "antireflect",
        ],
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[1, 2, 3, 17, 31, 32, 33, 3071, 3072, 3073],
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["auto", "row-major", "tile-native"],
        default=["auto"],
    )
    parser.add_argument("--tolerance", type=float)
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="Require successful finite outputs and correct shapes, but do not enforce PyWavelets tolerance.",
    )
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary not found: {binary}")
    if any(length <= 0 for length in args.lengths):
        parser.error("all lengths must be positive")
    wavelets = args.wavelets
    if args.all_schemes:
        wavelets = sorted(path.stem for path in args.schemes_dir.glob("*.json"))
        if not wavelets:
            parser.error(f"no JSON schemes found under {args.schemes_dir}")

    failures: list[str] = []
    max_error = 0.0
    case_count = 0
    with tempfile.TemporaryDirectory(prefix="ttwv-lwt-boundary-") as temporary:
        signal_path = Path(temporary) / "signal.txt"
        for length in args.lengths:
            signal = make_signal(length)
            np.savetxt(signal_path, signal, fmt="%.9g")
            for wavelet in wavelets:
                tolerance = args.tolerance or DEFAULT_TOLERANCES.get(wavelet, 1.0e-2)
                for mode in args.modes:
                    if mode in {"reflect", "antireflect"} and length == 1:
                        print(
                            f"{wavelet:10s} N={length:6d} mode={mode:13s} "
                            "skipped (PyWavelets DWT requires N > 1)"
                        )
                        continue
                    expected_a, expected_d = (
                        values.astype(np.float64) for values in pywt.dwt(signal, wavelet, mode=mode)
                    )
                    for layout in args.layouts:
                        candidate_a, candidate_d = run_case(
                            binary, wavelet, mode, layout, signal_path
                        )
                        error_a = max_abs_error(candidate_a, expected_a)
                        error_d = max_abs_error(candidate_d, expected_d)
                        error = max(error_a, error_d)
                        max_error = max(max_error, error)
                        case_count += 1
                        print(
                            f"{wavelet:10s} N={length:6d} mode={mode:9s} "
                            f"layout={layout:11s} max_abs_a={error_a:.8e} "
                            f"max_abs_d={error_d:.8e}"
                        )
                        shape_mismatch = (
                            candidate_a.shape != expected_a.shape
                            or candidate_d.shape != expected_d.shape
                        )
                        non_finite = not np.all(np.isfinite(candidate_a)) or not np.all(
                            np.isfinite(candidate_d)
                        )
                        tolerance_failure = not args.runtime_only and error > tolerance
                        if shape_mismatch or non_finite or tolerance_failure:
                            failures.append(
                                f"{wavelet} N={length} mode={mode} layout={layout}: "
                                f"shapes {candidate_a.shape}/{candidate_d.shape} vs "
                                f"{expected_a.shape}/{expected_d.shape}, "
                                f"finite={not non_finite}, error={error:.8e}, "
                                f"tolerance={tolerance:.8e}"
                            )

    print(f"validated_device_cases: {case_count}")
    print(f"max_abs_vs_pywavelets: {max_error:.8e}")
    if failures:
        raise SystemExit("\n".join(["LWT boundary validation failed:", *failures]))


if __name__ == "__main__":
    main()
