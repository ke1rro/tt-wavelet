#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXE_CANDIDATES = (
    PROJECT_ROOT / "build" / "vertical_stencil_test",
    PROJECT_ROOT / "build" / "tt-wavelet" / "vertical_stencil_test",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    input_tensor: np.ndarray
    kernel: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TT-wavelet vertical stencil against generated cases and validate the output."
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=None,
        help="Path to the stencil executable. If omitted, a few common build locations are tried.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Absolute and relative tolerance for validation (default: %(default)s).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1234,
        help="Seed used for deterministic random test generation (default: %(default)s).",
    )
    parser.add_argument(
        "--extra-random-cases",
        type=int,
        default=0,
        help="Additional random cases per kernel size beyond the built-in structured cases.",
    )
    parser.add_argument(
        "--keep-failures",
        action="store_true",
        help="Keep per-case temporary files when a validation failure occurs.",
    )
    return parser.parse_args()


def resolve_executable(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Executable not found: {explicit}")
        return explicit

    env_value = os.environ.get("TT_WAVELET_STENCIL_EXE")
    if env_value:
        candidate = Path(env_value)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"TT_WAVELET_STENCIL_EXE points to a missing executable: {candidate}")

    for candidate in DEFAULT_EXE_CANDIDATES:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in DEFAULT_EXE_CANDIDATES)
    raise FileNotFoundError(
        "Could not find the stencil executable. Tried:\n"
        f"{tried}\n"
        "You can pass --exe or set TT_WAVELET_STENCIL_EXE."
    )


def make_random_tensor(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=0.0, scale=1.0, size=(64, 32)).astype(np.float32)


def make_ramp_tensor() -> np.ndarray:
    rows = np.linspace(-1.0, 1.0, num=64, dtype=np.float32)[:, None]
    cols = np.linspace(-0.5, 0.5, num=32, dtype=np.float32)[None, :]
    return (rows + cols).astype(np.float32)


def make_impulse_tensor() -> np.ndarray:
    tensor = np.zeros((64, 32), dtype=np.float32)
    tensor[16, 7] = 2.5
    tensor[31, 15] = -1.75
    tensor[48, 23] = 3.0
    tensor[63, 31] = -0.5
    return tensor


def make_checkerboard_tensor() -> np.ndarray:
    rows = np.arange(64, dtype=np.float32)[:, None]
    cols = np.arange(32, dtype=np.float32)[None, :]
    return np.where(((rows + cols) % 2) == 0, 1.0, -1.0).astype(np.float32)


def make_random_kernel(k: int, rng: np.random.Generator) -> np.ndarray:
    kernel = rng.uniform(-1.0, 1.0, size=k).astype(np.float32)
    if np.allclose(kernel, 0.0):
        kernel[0] = 1.0
    return kernel


def make_ramp_kernel(k: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, num=k, dtype=np.float32)


def make_alternating_kernel(k: int) -> np.ndarray:
    values = np.array([(-1.0 if (i % 2) else 1.0) * (i + 1) for i in range(k)], dtype=np.float32)
    return values / float(k)


def make_gaussian_kernel(k: int) -> np.ndarray:
    positions = np.arange(k, dtype=np.float32) - (k - 1) / 2.0
    sigma = max(1.0, k / 3.0)
    kernel = np.exp(-0.5 * (positions / sigma) ** 2).astype(np.float32)
    kernel /= np.sum(kernel, dtype=np.float32)
    return kernel


def build_cases_for_k(k: int, rng: np.random.Generator, extra_random_cases: int) -> list[CaseSpec]:
    cases = [
        CaseSpec(name="random/random", input_tensor=make_random_tensor(rng), kernel=make_random_kernel(k, rng)),
        CaseSpec(name="ramp/alternating", input_tensor=make_ramp_tensor(), kernel=make_alternating_kernel(k)),
        CaseSpec(name="impulse/gaussian", input_tensor=make_impulse_tensor(), kernel=make_gaussian_kernel(k)),
        CaseSpec(name="checkerboard/ramp", input_tensor=make_checkerboard_tensor(), kernel=make_ramp_kernel(k)),
    ]

    for index in range(extra_random_cases):
        cases.append(
            CaseSpec(
                name=f"random_extra_{index + 1}",
                input_tensor=make_random_tensor(rng),
                kernel=make_random_kernel(k, rng),
            )
        )

    return cases


def write_input_tensor(path: Path, tensor: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in tensor:
            handle.write(" ".join(f"{float(value):.9g}" for value in row))
            handle.write("\n")


def write_kernel(path: Path, kernel: np.ndarray) -> None:
    coeffs = " ".join(f"{float(value):.9g}" for value in kernel)
    path.write_text(f"{kernel.size} {coeffs}\n", encoding="utf-8")


def run_executable(exe: Path, tensor_path: Path, kernel_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(exe), str(tensor_path), str(kernel_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def parse_output_matrix(stdout: str) -> np.ndarray:
    rows: list[list[float]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 32:
            continue
        try:
            row = [float(value) for value in parts]
        except ValueError:
            continue
        rows.append(row)

    if len(rows) < 32:
        raise ValueError(
            f"Could not parse a 32x32 output matrix from stdout. Parsed {len(rows)} candidate rows.\n"
            f"Captured stdout:\n{stdout}"
        )

    return np.array(rows[-32:], dtype=np.float32)


def reference_valid_convolution(tensor: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    expected = np.zeros((32, 32), dtype=np.float32)
    for col in range(32):
        column = tensor[:, col].astype(np.float32, copy=False)
        valid = np.convolve(column, kernel.astype(np.float32, copy=False), mode="valid")
        expected[:, col] = valid[:32].astype(np.float32, copy=False)
    return expected


def compare_matrices(actual: np.ndarray, expected: np.ndarray, tolerance: float) -> tuple[bool, float, tuple[int, int], float, float]:
    diff = actual.astype(np.float32) - expected.astype(np.float32)
    abs_diff = np.abs(diff)
    max_index = np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)
    max_abs = float(abs_diff[max_index])
    actual_value = float(actual[max_index])
    expected_value = float(expected[max_index])
    ok = np.allclose(actual, expected, rtol=tolerance, atol=tolerance)
    return ok, max_abs, max_index, actual_value, expected_value


def format_case_header(k: int, case: str) -> str:
    return f"K={k:02d} case={case}"


def save_failure_artifact(
    k: int,
    case_name: str,
    tensor: np.ndarray,
    kernel: np.ndarray,
    stdout: str,
    stderr: str,
) -> Path:
    safe_case = case_name.replace("/", "_").replace(" ", "_")
    artifact_dir = PROJECT_ROOT / "validation_failures" / f"k{k:02d}_{safe_case}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_input_tensor(artifact_dir / "input_tensor.txt", tensor)
    write_kernel(artifact_dir / "kernel.txt", kernel)
    (artifact_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (artifact_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    return artifact_dir


def main() -> int:
    args = parse_args()
    exe = resolve_executable(args.exe)
    rng = np.random.default_rng(args.random_seed)

    total = 0
    failures: list[str] = []

    for k in range(2, 14):
        cases = build_cases_for_k(k, rng, args.extra_random_cases)
        for case in cases:
            total += 1
            with tempfile.TemporaryDirectory(prefix=f"stencil_k{k:02d}_") as tmp_dir:
                tmp_path = Path(tmp_dir)
                tensor_path = tmp_path / "input_tensor.txt"
                kernel_path = tmp_path / "kernel.txt"

                write_input_tensor(tensor_path, case.input_tensor)
                write_kernel(kernel_path, case.kernel)

                completed = run_executable(exe, tensor_path, kernel_path)
                if completed.returncode != 0:
                    message = (
                        f"{format_case_header(k, case.name)} failed to execute with exit code {completed.returncode}.\n"
                        f"stdout:\n{completed.stdout}\n"
                        f"stderr:\n{completed.stderr}"
                    )
                    if args.keep_failures:
                        artifact_dir = save_failure_artifact(
                            k, case.name, case.input_tensor, case.kernel, completed.stdout, completed.stderr
                        )
                        message += f"\nSaved failure artifacts to: {artifact_dir}"
                    failures.append(message)
                    print(message)
                    continue

                try:
                    actual = parse_output_matrix(completed.stdout)
                except ValueError as exc:
                    message = f"{format_case_header(k, case.name)} parse failure: {exc}"
                    if args.keep_failures:
                        artifact_dir = save_failure_artifact(
                            k, case.name, case.input_tensor, case.kernel, completed.stdout, completed.stderr
                        )
                        message += f"\nSaved failure artifacts to: {artifact_dir}"
                    failures.append(message)
                    print(failures[-1])
                    continue

                expected = reference_valid_convolution(case.input_tensor, case.kernel)
                ok, max_abs, max_index, actual_value, expected_value = compare_matrices(
                    actual, expected, args.tolerance
                )

                if ok:
                    print(f"PASS {format_case_header(k, case.name)} max_abs={max_abs:.3e}")
                else:
                    message = (
                        f"FAIL {format_case_header(k, case.name)} max_abs={max_abs:.3e} at row={max_index[0]} col={max_index[1]} "
                        f"actual={actual_value:.9g} expected={expected_value:.9g}"
                    )
                    if args.keep_failures:
                        artifact_dir = save_failure_artifact(
                            k, case.name, case.input_tensor, case.kernel, completed.stdout, completed.stderr
                        )
                        message += f"\nSaved failure artifacts to: {artifact_dir}"
                    failures.append(message)
                    print(message)

    print()
    print(f"Completed {total} cases: {total - len(failures)} passed, {len(failures)} failed.")
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
