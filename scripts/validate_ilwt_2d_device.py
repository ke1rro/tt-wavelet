#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the focused production-path 2D ILWT validation matrix."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORWARD = PROJECT_ROOT / "build" / "lwt_2d"
INVERSE = PROJECT_ROOT / "build" / "ilwt_2d"
SET_ENV = PROJECT_ROOT / "scripts" / "set_env.sh"
BANDS = ("LL", "LH", "HL", "HH")
BOUNDARY_MODES = (
    "zero",
    "constant",
    "symmetric",
    "reflect",
    "periodic",
    "smooth",
    "antisymmetric",
    "antireflect",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schemes",
        default="db1,db7,bior3.9",
        help="Comma-separated schemes (default: %(default)s).",
    )
    parser.add_argument(
        "--shapes",
        default="32x32,33x35,100x70,1000x100",
        help="Comma-separated HEIGHTxWIDTH shapes (default: %(default)s).",
    )
    parser.add_argument(
        "--modes",
        default="symmetric",
        help="Comma-separated signal-extension modes (default: %(default)s).",
    )
    parser.add_argument("--cores", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--skip-pywavelets",
        action="store_true",
        help="Skip the one small PyWavelets coefficient-input check.",
    )
    return parser.parse_args()


def run_device(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    shell_command = f"source {shlex.quote(str(SET_ENV))} && " + " ".join(
        shlex.quote(item) for item in command
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


def parse_shapes(text: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for item in text.split(","):
        height, width = item.lower().split("x", 1)
        shapes.append((int(height), int(width)))
    return shapes


def forward(
    scheme: str,
    mode: str,
    height: int,
    width: int,
    input_path: Path,
    prefix: Path,
    cores: int,
    timeout: float,
) -> None:
    run_device(
        [
            str(FORWARD),
            "--binary-input",
            "--boundary-mode",
            mode,
            "--quiet",
            "--cores",
            str(cores),
            "--output-prefix",
            str(prefix),
            scheme,
            str(height),
            str(width),
            str(input_path),
        ],
        timeout,
    )


def inverse(
    scheme: str,
    mode: str,
    height: int,
    width: int,
    prefix: Path,
    output: Path,
    cores: int,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    return run_device(
        [
            str(INVERSE),
            "--boundary-mode",
            mode,
            "--cores",
            str(cores),
            "--output",
            str(output),
            scheme,
            str(height),
            str(width),
            *(str(Path(f"{prefix}_{band}.f32")) for band in BANDS),
        ],
        timeout,
    )


def read_reconstruction(reference: np.ndarray, path: Path) -> np.ndarray:
    reconstructed = np.fromfile(path, dtype=np.float32)
    if reconstructed.size != reference.size:
        raise RuntimeError(
            f"reconstructed output has {reconstructed.size} elements; "
            f"expected {reference.size}"
        )
    reconstructed = reconstructed.reshape(reference.shape)
    if not np.all(np.isfinite(reconstructed)):
        raise RuntimeError("reconstructed output contains NaN or Inf")
    return reconstructed


def max_error(reference: np.ndarray, path: Path) -> float:
    reconstructed = read_reconstruction(reference, path)
    return float(np.max(np.abs(reference - reconstructed)))


def pywavelets_check(root: Path, mode: str, timeout: float) -> float:
    import pywt

    height, width = 33, 35
    signal = np.random.default_rng(17).normal(size=(height, width)).astype(np.float32)
    approximation, (horizontal, vertical, diagonal) = pywt.dwt2(
        signal, "db1", mode=mode
    )
    prefix = root / "pywt"
    # Device convention is (low-y,high-x)=LH and (high-y,low-x)=HL.
    for name, values in zip(
        BANDS, (approximation, vertical, horizontal, diagonal), strict=True
    ):
        np.asarray(values, dtype=np.float32).tofile(Path(f"{prefix}_{name}.f32"))
    output = root / "pywt_output.f32"
    inverse("db1", mode, height, width, prefix, output, 1, timeout)
    pywt_reconstructed = pywt.idwt2(
        (approximation, (horizontal, vertical, diagonal)),
        "db1",
        mode=mode,
    )[:height, :width].astype(np.float32)
    return max_error(pywt_reconstructed, output)


def main() -> int:
    args = parse_args()
    schemes = [item.strip() for item in args.schemes.split(",") if item.strip()]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    unsupported = sorted(set(modes) - set(BOUNDARY_MODES))
    if unsupported:
        raise ValueError(f"unsupported signal-extension modes: {', '.join(unsupported)}")
    if not modes:
        raise ValueError("--modes did not contain a signal-extension mode")
    shapes = parse_shapes(args.shapes)
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ttwv-ilwt2d-") as directory:
        root = Path(directory)
        for scheme_index, scheme in enumerate(schemes):
            for mode_index, mode in enumerate(modes):
                for shape_index, (height, width) in enumerate(shapes):
                    if mode in {"reflect", "antireflect"} and min(height, width) <= 1:
                        print(f"UNSUPPORTED {scheme} {mode} {height}x{width}: both dimensions must exceed one")
                        continue
                    signal = np.random.default_rng(
                        args.seed + 1000 * scheme_index + 100 * mode_index + shape_index
                    ).uniform(-1.0, 1.0, size=(height, width)).astype(np.float32)
                    stem = f"{scheme}_{mode}_{height}x{width}"
                    input_path = root / f"{stem}_input.f32"
                    prefix = root / stem
                    output = root / f"{stem}_output.f32"
                    signal.tofile(input_path)
                    forward(
                        scheme,
                        mode,
                        height,
                        width,
                        input_path,
                        prefix,
                        args.cores,
                        args.timeout_seconds,
                    )
                    inverse(
                        scheme,
                        mode,
                        height,
                        width,
                        prefix,
                        output,
                        args.cores,
                        args.timeout_seconds,
                    )
                    error = max_error(signal, output)
                    status = "PASS" if error <= args.tolerance else "FAIL"
                    print(f"{status} {scheme} {mode} {height}x{width} max_abs_error={error:.9g}")
                    if error > args.tolerance:
                        failures.append(f"{scheme} {mode} {height}x{width}: {error}")

                    if scheme == "db7" and (height, width) == (33, 35):
                        single_output = root / f"db7_{mode}_33x35_single_core.f32"
                        inverse(
                            scheme,
                            mode,
                            height,
                            width,
                            prefix,
                            single_output,
                            1,
                            args.timeout_seconds,
                        )
                        single_error = max_error(signal, single_output)
                        multi = np.fromfile(output, dtype=np.float32)
                        single = np.fromfile(single_output, dtype=np.float32)
                        bit_identical = np.array_equal(multi.view(np.uint32), single.view(np.uint32))
                        print(
                            "PASS" if bit_identical and single_error <= args.tolerance else "FAIL",
                            f"db7 {mode} 33x35 single-vs-multi-core",
                            f"bit_identical={bit_identical}",
                            f"max_abs_error={single_error:.9g}",
                        )
                        if not bit_identical or single_error > args.tolerance:
                            failures.append(f"db7 {mode} 33x35 single/multi-core mismatch")

        if not args.skip_pywavelets:
            for mode in modes:
                error = pywavelets_check(root, mode, args.timeout_seconds)
                print(
                    f"{'PASS' if error <= args.tolerance else 'FAIL'} "
                    f"pywt db1 {mode} 33x35 max_abs_error={error:.9g}"
                )
                if error > args.tolerance:
                    failures.append(f"PyWavelets db1 {mode} 33x35: {error}")

    if failures:
        print("2D ILWT validation failures:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("2D ILWT focused validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
