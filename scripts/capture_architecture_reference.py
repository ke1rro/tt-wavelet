#!/usr/bin/env python3
"""Capture and compare deterministic Wormhole/Blackhole FP32 LWT results."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pywt

from runtime_checks import parse_runtime_architecture


FORMAT_VERSION = 1


def float32_bits(values: np.ndarray) -> list[str]:
    packed = np.asarray(values, dtype=np.float32).view(np.uint32)
    return [f"0x{int(value):08x}" for value in packed]


def bits_float32(values: list[str]) -> np.ndarray:
    packed = np.asarray([int(value, 16) for value in values], dtype=np.uint32)
    return packed.view(np.float32)


def parse_vector(output: str, label: str) -> np.ndarray:
    prefix = f"tt-wavelet {label}"
    for line in output.splitlines():
        if line.startswith(prefix):
            text = line.split("[", 1)[1].rsplit("]", 1)[0]
            return np.fromstring(text, sep=",", dtype=np.float32)
    raise RuntimeError(f"device output did not contain {label}")


def parse_layout(output: str, transform: str) -> str:
    prefix = f"{transform}_layout:"
    for line in output.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"device output did not report {transform} layout")


def run_device(command: list[str], layout: str) -> str:
    environment = os.environ.copy()
    environment.pop("ARCH_NAME", None)
    environment["TT_LOGGER_LEVEL"] = "FATAL"
    environment["TT_WAVELET_LWT_WORKSPACE_LAYOUT"] = layout
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            f"device command failed with exit code {result.returncode}:\n{output}"
        )
    return output


def generated_inputs(
    wavelet: str, seed: int, length: int, boundary_mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.Generator(np.random.PCG64(seed))
    signal = generator.uniform(-1.0, 1.0, size=length).astype(np.float32)
    approximation, detail = (
        values.astype(np.float32)
        for values in pywt.dwt(signal, wavelet, mode=boundary_mode)
    )
    return signal, approximation, detail


def stored_inputs(
    path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    capture = json.loads(path.read_text(encoding="utf-8"))
    if capture.get("format_version") != FORMAT_VERSION:
        raise RuntimeError(f"unsupported capture format in {path}")
    inputs = capture["inputs"]
    return (
        capture,
        bits_float32(inputs["signal_bits"]),
        bits_float32(inputs["approximation_bits"]),
        bits_float32(inputs["detail_bits"]),
    )


def write_values(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, np.asarray(values, dtype=np.float32), fmt="%.9g")


def capture(args: argparse.Namespace) -> int:
    binary = args.binary.resolve()
    if not binary.is_file():
        raise RuntimeError(f"binary not found: {binary}")

    if args.inputs_from is not None:
        source, signal, approximation, detail = stored_inputs(args.inputs_from)
        source_case = source["case"]
        wavelet = source_case["wavelet"]
        seed = int(source_case["seed"])
        length = int(source_case["length"])
        boundary_mode = source_case["boundary_mode"]
    else:
        wavelet = args.wavelet
        seed = args.seed
        length = args.length
        boundary_mode = args.boundary_mode
        signal, approximation, detail = generated_inputs(
            wavelet, seed, length, boundary_mode
        )

    with tempfile.TemporaryDirectory(prefix="ttwv-arch-capture-") as temporary:
        temporary_path = Path(temporary)
        signal_path = temporary_path / "signal.txt"
        approximation_path = temporary_path / "approximation.txt"
        detail_path = temporary_path / "detail.txt"
        write_values(signal_path, signal)
        write_values(approximation_path, approximation)
        write_values(detail_path, detail)

        forward_output = run_device(
            [
                str(binary),
                "--boundary-mode",
                boundary_mode,
                wavelet,
                str(signal_path),
            ],
            args.layout,
        )
        inverse_output = run_device(
            [
                str(binary),
                "--inverse",
                "--original-length",
                str(length),
                "--boundary-mode",
                boundary_mode,
                "--approximation-file",
                str(approximation_path),
                "--detail-file",
                str(detail_path),
                wavelet,
            ],
            args.layout,
        )

    detected_forward = parse_runtime_architecture(forward_output)
    detected_inverse = parse_runtime_architecture(inverse_output)
    expected_architecture = (
        "wormhole_b0" if args.architecture == "wormhole" else args.architecture
    )
    if (
        detected_forward != expected_architecture
        or detected_inverse != expected_architecture
    ):
        raise RuntimeError(
            "capture label does not match hardware: "
            f"label={expected_architecture}, forward={detected_forward}, inverse={detected_inverse}"
        )

    candidate_approximation = parse_vector(forward_output, "approximation coefficients")
    candidate_detail = parse_vector(forward_output, "detail coefficients")
    reconstructed = parse_vector(inverse_output, "reconstructed signal")
    capture_data = {
        "format_version": FORMAT_VERSION,
        "architecture": detected_forward,
        "case": {
            "wavelet": wavelet,
            "seed": seed,
            "length": length,
            "boundary_mode": boundary_mode,
            "requested_layout": args.layout,
            "lwt_layout": parse_layout(forward_output, "lwt"),
            "ilwt_layout": parse_layout(inverse_output, "ilwt"),
            "storage": "fp32",
            "arithmetic": "fp32-sfpu",
            "generator": "numpy.PCG64.uniform[-1,1)",
            "coefficient_source": "PyWavelets DWT cast to FP32",
        },
        "inputs": {
            "signal_bits": float32_bits(signal),
            "approximation_bits": float32_bits(approximation),
            "detail_bits": float32_bits(detail),
        },
        "outputs": {
            "lwt_approximation_bits": float32_bits(candidate_approximation),
            "lwt_detail_bits": float32_bits(candidate_detail),
            "ilwt_signal_bits": float32_bits(reconstructed),
        },
    }
    args.output.write_text(
        json.dumps(capture_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"capture: {args.output}")
    print(f"architecture: {detected_forward}")
    print(f"lwt_layout: {capture_data['case']['lwt_layout']}")
    print(f"ilwt_layout: {capture_data['case']['ilwt_layout']}")
    print(f"signal_values: {signal.size}")
    print(f"coefficient_values: {approximation.size + detail.size}")
    return 0


def compare_array(
    label: str, reference_bits: list[str], candidate_bits: list[str]
) -> bool:
    reference = bits_float32(reference_bits)
    candidate = bits_float32(candidate_bits)
    if reference.shape != candidate.shape:
        print(f"{label}_shape: {reference.shape} != {candidate.shape}")
        return False

    mismatch = reference.view(np.uint32) != candidate.view(np.uint32)
    mismatch_indices = np.flatnonzero(mismatch)
    finite_pair = np.isfinite(reference) & np.isfinite(candidate)
    absolute = np.full(reference.shape, np.nan, dtype=np.float64)
    absolute[finite_pair] = np.abs(
        candidate[finite_pair].astype(np.float64)
        - reference[finite_pair].astype(np.float64)
    )
    denominator = np.abs(reference.astype(np.float64))
    relative = np.full(reference.shape, np.nan, dtype=np.float64)
    nonzero = finite_pair & (denominator != 0.0)
    relative[nonzero] = absolute[nonzero] / denominator[nonzero]
    relative[finite_pair & (denominator == 0.0) & (absolute == 0.0)] = 0.0
    relative[finite_pair & (denominator == 0.0) & (absolute != 0.0)] = np.inf

    def maximum_index(values: np.ndarray) -> int:
        valid_indices = np.flatnonzero(~np.isnan(values))
        if not valid_indices.size:
            return -1
        return int(valid_indices[np.argmax(values[valid_indices])])

    max_absolute_index = maximum_index(absolute)
    max_relative_index = maximum_index(relative)
    first_mismatch = int(mismatch_indices[0]) if mismatch_indices.size else -1
    reference_nonfinite = np.flatnonzero(~np.isfinite(reference))
    candidate_nonfinite = np.flatnonzero(~np.isfinite(candidate))
    print(f"{label}_values: {reference.size}")
    print(f"{label}_bit_mismatches: {mismatch_indices.size}")
    print(f"{label}_first_mismatch_index: {first_mismatch}")
    print(
        f"{label}_max_abs_error: "
        f"{absolute[max_absolute_index] if max_absolute_index >= 0 else 0.0:.9e}"
    )
    print(f"{label}_max_abs_error_index: {max_absolute_index}")
    print(
        f"{label}_max_relative_error: "
        f"{relative[max_relative_index] if max_relative_index >= 0 else 0.0:.9e}"
    )
    print(f"{label}_max_relative_error_index: {max_relative_index}")
    print(f"{label}_reference_nan_or_inf: {reference_nonfinite.size}")
    print(
        f"{label}_reference_first_nan_or_inf_index: "
        f"{int(reference_nonfinite[0]) if reference_nonfinite.size else -1}"
    )
    print(f"{label}_candidate_nan_or_inf: {candidate_nonfinite.size}")
    print(
        f"{label}_candidate_first_nan_or_inf_index: "
        f"{int(candidate_nonfinite[0]) if candidate_nonfinite.size else -1}"
    )
    return mismatch_indices.size == 0


def compare(args: argparse.Namespace) -> int:
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    for capture_path, content in (
        (args.reference, reference),
        (args.candidate, candidate),
    ):
        if content.get("format_version") != FORMAT_VERSION:
            raise RuntimeError(f"unsupported capture format in {capture_path}")

    invariant_case_fields = [
        "wavelet",
        "seed",
        "length",
        "boundary_mode",
        "requested_layout",
        "storage",
        "arithmetic",
        "generator",
        "coefficient_source",
    ]
    metadata_match = True
    for field in invariant_case_fields:
        lhs = reference["case"][field]
        rhs = candidate["case"][field]
        equal = lhs == rhs
        metadata_match = metadata_match and equal
        print(f"case_{field}: {lhs!r} / {rhs!r} ({'MATCH' if equal else 'MISMATCH'})")

    input_match = True
    for field in ("signal_bits", "approximation_bits", "detail_bits"):
        equal = reference["inputs"][field] == candidate["inputs"][field]
        input_match = input_match and equal
        print(f"input_{field}: {'MATCH' if equal else 'MISMATCH'}")

    output_match = True
    for field in (
        "lwt_approximation_bits",
        "lwt_detail_bits",
        "ilwt_signal_bits",
    ):
        output_match = (
            compare_array(
                field, reference["outputs"][field], candidate["outputs"][field]
            )
            and output_match
        )

    print(f"reference_architecture: {reference['architecture']}")
    print(f"candidate_architecture: {candidate['architecture']}")
    print(f"metadata_result: {'PASS' if metadata_match else 'FAIL'}")
    print(f"identical_inputs_result: {'PASS' if input_match else 'FAIL'}")
    print(f"bitwise_output_result: {'PASS' if output_match else 'FAIL'}")
    return (
        0
        if metadata_match and input_match and (output_match or args.allow_differences)
        else 1
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--binary", type=Path, default=root / "build" / "lwt")
    capture_parser.add_argument(
        "--architecture", choices=["wormhole", "blackhole"], required=True
    )
    capture_parser.add_argument("--wavelet", default="bior3.9")
    capture_parser.add_argument("--seed", type=int, default=20260719)
    capture_parser.add_argument("--length", type=int, default=3073)
    capture_parser.add_argument(
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
    capture_parser.add_argument(
        "--layout", choices=["auto", "row-major", "tile-native"], default="auto"
    )
    capture_parser.add_argument("--inputs-from", type=Path)
    capture_parser.add_argument("--output", type=Path, required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("reference", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.add_argument("--allow-differences", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return capture(args) if args.command == "capture" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
