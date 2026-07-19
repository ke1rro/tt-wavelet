#!/usr/bin/env python3
"""Validate Blackhole/Wormhole 1D ILWT against PyWavelets coefficients."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pywt

from runtime_checks import (
    check_consistent_architecture,
    parse_runtime_architecture,
    run_ncrisc_elf_gate,
)

DEFAULT_TOLERANCES = {
    "db1": 2.0e-6,
    "haar": 2.0e-6,
    "db7": 2.0e-4,
    "bior3.9": 2.0e-5,
}


def parse_reconstructed_signal(stdout: str) -> np.ndarray:
    for line in stdout.splitlines():
        if line.startswith("tt-wavelet reconstructed signal"):
            return np.fromstring(line.split("[", 1)[1].rsplit("]", 1)[0], sep=",", dtype=np.float64)
    raise RuntimeError("lwt output did not contain a reconstructed signal")


def make_signal(length: int) -> np.ndarray:
    index = np.arange(length, dtype=np.float32)
    return (0.7 * np.sin(index * 0.071) + 0.2 * np.cos(index * 0.013) + index * 1.0e-4).astype(
        np.float32
    )


def run_case(
    binary: Path,
    wavelet: str,
    length: int,
    mode: str,
    layout: str,
    approximation_path: Path,
    detail_path: Path,
) -> tuple[np.ndarray, str]:
    environment = os.environ.copy()
    environment["TT_WAVELET_LWT_WORKSPACE_LAYOUT"] = layout
    command = [
        str(binary),
        "--inverse",
        "--original-length",
        str(length),
        "--boundary-mode",
        mode,
        "--approximation-file",
        str(approximation_path),
        "--detail-file",
        str(detail_path),
        wavelet,
    ]
    result = subprocess.run(command, text=True, capture_output=True, env=environment, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ILWT failed for {wavelet}, N={length}, layout={layout}:\n{result.stdout}\n{result.stderr}"
        )
    return (
        parse_reconstructed_signal(result.stdout),
        parse_runtime_architecture(result.stdout + result.stderr),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=root / "build" / "lwt")
    parser.add_argument("--wavelets", nargs="+", default=["db1", "db7", "bior3.9"])
    parser.add_argument("--lengths", nargs="+", type=int, default=[17, 32, 33, 3071, 3072, 3073])
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["auto", "row-major", "tile-native"],
        default=["auto"],
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
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--layout-tolerance", type=float, default=1.0e-7)
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary not found: {binary}")
    if any(length <= 0 for length in args.lengths):
        parser.error("all lengths must be positive")

    failures: list[str] = []
    max_pywavelets_error = 0.0
    max_layout_error = 0.0
    case_count = 0
    architecture: str | None = None
    with tempfile.TemporaryDirectory(prefix="ttwv-ilwt-") as temporary:
        temporary_path = Path(temporary)
        for wavelet in args.wavelets:
            tolerance = args.tolerance or DEFAULT_TOLERANCES.get(wavelet, 1.0e-2)
            for length in args.lengths:
                signal = make_signal(length)
                for mode in args.modes:
                    if mode in {"reflect", "antireflect"} and length == 1:
                        print(
                            f"{wavelet:10s} N={length:6d} mode={mode:13s} "
                            "skipped (PyWavelets DWT requires N > 1)"
                        )
                        continue
                    approximation, detail = (
                        values.astype(np.float32) for values in pywt.dwt(signal, wavelet, mode=mode)
                    )
                    approximation_path = temporary_path / "approximation.txt"
                    detail_path = temporary_path / "detail.txt"
                    np.savetxt(approximation_path, approximation, fmt="%.9g")
                    np.savetxt(detail_path, detail, fmt="%.9g")
                    expected = pywt.idwt(approximation, detail, wavelet, mode=mode)[:length]

                    outputs: dict[str, np.ndarray] = {}
                    for layout in args.layouts:
                        reconstructed, case_architecture = run_case(
                            binary,
                            wavelet,
                            length,
                            mode,
                            layout,
                            approximation_path,
                            detail_path,
                        )
                        architecture = check_consistent_architecture(
                            architecture, case_architecture
                        )
                        outputs[layout] = reconstructed
                        error = float(np.max(np.abs(reconstructed - expected)))
                        max_pywavelets_error = max(max_pywavelets_error, error)
                        case_count += 1
                        print(
                            f"{wavelet:10s} N={length:6d} mode={mode:13s} "
                            f"layout={layout:11s} max_abs_vs_pywavelets={error:.8e}"
                        )
                        if not np.all(np.isfinite(reconstructed)) or error > tolerance:
                            failures.append(
                                f"{wavelet} N={length} mode={mode} layout={layout}: "
                                f"error {error:.8e} > {tolerance:.8e}"
                            )

                    first_layout = args.layouts[0]
                    for layout in args.layouts[1:]:
                        layout_error = float(
                            np.max(np.abs(outputs[layout] - outputs[first_layout]))
                        )
                        max_layout_error = max(max_layout_error, layout_error)
                        if layout_error > args.layout_tolerance:
                            failures.append(
                                f"{wavelet} N={length} mode={mode}: {first_layout}/{layout} "
                                f"difference {layout_error:.8e} > {args.layout_tolerance:.8e}"
                            )

    print(f"validated_device_cases: {case_count}")
    print(f"max_abs_vs_pywavelets: {max_pywavelets_error:.8e}")
    print(f"max_abs_between_layouts: {max_layout_error:.8e}")
    if failures:
        raise SystemExit("\n".join(["ILWT validation failed:", *failures]))
    if architecture is None:
        raise SystemExit("ILWT validation did not execute a device case")
    run_ncrisc_elf_gate(root, architecture)


if __name__ == "__main__":
    main()
