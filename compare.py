#!/usr/bin/env python3
import argparse
import ast
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
LIFTING_USAGE_DIR = PROJECT_ROOT / "lifting-factorization" / "usage"
TT_WAVELET_BINARY = PROJECT_ROOT / "build" / "tt-wavelet" / "tt_wavelet_lifting_test"
TT_WAVELET_ENV = PROJECT_ROOT / "scripts" / "set_env.sh"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python3"

try:
    from pywt import dwt
except ModuleNotFoundError as exc:
    if VENV_PYTHON.exists() and Path(sys.executable) != VENV_PYTHON:
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])
    raise ModuleNotFoundError(
        "PyWavelets is not installed for the active interpreter. "
        "Use /root/tt-wavelet/.venv/bin/python3 compare_all_lifting.py ..."
    ) from exc

if str(LIFTING_USAGE_DIR) not in sys.path:
    sys.path.insert(0, str(LIFTING_USAGE_DIR))

import dtypes  # noqa: E402  # type: ignore[import-not-found]
from lifting import LiftingScheme  # noqa: E402  # type: ignore[import-not-found]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyWavelets, lifting-factorization, and TT-wavelet outputs for one lifting scheme."
    )
    parser.add_argument(
        "--wavelet",
        default="bior3.9",
        help="Wavelet name / JSON basename under ttnn-wavelet/lifting_schemes (default: %(default)s).",
    )
    parser.add_argument(
        "--signal",
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19",
        help="Comma-separated list of numeric samples.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Absolute tolerance used for pairwise coefficient comparisons (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-tt-wavelet",
        action="store_true",
        help="Skip running the TT-wavelet device executable.",
    )
    return parser.parse_args()


def parse_signal(raw_signal: str) -> list[float]:
    return [float(token.strip()) for token in raw_signal.split(",") if token.strip()]


def format_coeffs(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(value):.8e}" for value in values) + "]"


def print_pairwise_mismatches(
    lhs: Sequence[float], rhs: Sequence[float], name: str, tolerance: float
) -> None:
    mismatches = 0

    if len(lhs) != len(rhs):
        print(f"{name} length differs: lhs={len(lhs)} vs rhs={len(rhs)}")
        return

    for i, (lhs_val, rhs_val) in enumerate(zip(lhs, rhs)):
        if abs(float(lhs_val) - float(rhs_val)) > tolerance:
            mismatches += 1
            print(f"{name} coeff {i} differs: lhs={float(lhs_val):.8e} vs rhs={float(rhs_val):.8e}")

    if mismatches == 0:
        print(f"{name}: all coefficients match within tolerance {tolerance}")


def print_error_metrics(reference: Sequence[float], candidate: Sequence[float], name: str) -> None:
    if len(reference) == 0 or len(candidate) == 0:
        print(f"{name} error metrics: skipped (empty sequence)")
        return

    if len(reference) != len(candidate):
        overlap = min(len(reference), len(candidate))
        print(
            f"{name} error metrics: length differs ref={len(reference)} vs cand={len(candidate)}, "
            f"using overlap={overlap}"
        )
    else:
        overlap = len(reference)

    diffs = [float(candidate[i]) - float(reference[i]) for i in range(overlap)]
    abs_diffs = [abs(value) for value in diffs]
    max_abs = max(abs_diffs)
    mean_abs = sum(abs_diffs) / overlap
    mse = sum(value * value for value in diffs) / overlap
    rmse = math.sqrt(mse)

    print(
        f"{name} error metrics: "
        f"max_abs={max_abs:.8e}, mean_abs={mean_abs:.8e}, rmse={rmse:.8e}, mse={mse:.8e}"
    )


def run_tt_wavelet(scheme_path: Path, raw_signal: str) -> dict[str, list[float]]:
    if not TT_WAVELET_BINARY.exists():
        raise FileNotFoundError(
            f"TT-wavelet binary not found at {TT_WAVELET_BINARY}. Rebuild with ./update.sh Release tt_wavelet_lifting_test"
        )

    command = (
        f"source {sh_quote(str(TT_WAVELET_ENV))} "
        f"&& {sh_quote(str(TT_WAVELET_BINARY))} {sh_quote(str(scheme_path))} {sh_quote(raw_signal)}"
    )
    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )

    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"TT-wavelet run failed with exit code {completed.returncode}.\nCaptured output:\n{output}"
        )

    return {
        "approximation": extract_coeff_line(
            output,
            [
                "tt-wavelet approximation coefficients",
                "lifting approximation coefficients",
            ],
        ),
        "detail": extract_coeff_line(
            output,
            [
                "tt-wavelet detail coefficients",
                "lifting detail coefficients",
            ],
        ),
    }


def extract_coeff_line(output: str, candidate_labels: Sequence[str]) -> list[float]:
    for label in candidate_labels:
        pattern = re.compile(rf"{re.escape(label)} \((\d+)\): \[(.*?)\]")
        match = pattern.search(output)
        if match is None:
            continue

        values_blob = "[" + match.group(2) + "]"
        try:
            parsed = ast.literal_eval(values_blob)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"Failed to parse coefficient list for {label}: {values_blob}"
            ) from exc

        logical_length = int(match.group(1))
        return [float(value) for value in parsed[:logical_length]]

    raise ValueError(f"Unable to find coefficient line for labels: {', '.join(candidate_labels)}")


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> int:
    args = parse_args()
    signal = parse_signal(args.signal)
    scheme_path = PROJECT_ROOT / "ttnn-wavelet" / "lifting_schemes" / f"{args.wavelet}.json"

    if not scheme_path.exists():
        raise FileNotFoundError(f"Wavelet scheme file not found: {scheme_path}")

    print(f"signal: {signal}")
    print(f"scheme path: {scheme_path}")
    print()

    cA_pywt, cD_pywt = dwt(signal, args.wavelet, mode="symmetric")

    scheme = LiftingScheme.from_file(
        str(scheme_path),
        mode="symmetric",
        dtype=dtypes.float32,
    )
    cA_lifting, cD_lifting = scheme.forward(signal)

    print(f"pywt approximation coefficients: {format_coeffs(cA_pywt)}")
    print(f"pywt detail coefficients: {format_coeffs(cD_pywt)}")
    print(f"pywt lengths: cA={len(cA_pywt)}, cD={len(cD_pywt)}")
    print()
    print(f"lifting-factorization approximation coefficients: {format_coeffs(cA_lifting)}")
    print(f"lifting-factorization detail coefficients: {format_coeffs(cD_lifting)}")
    print(f"lifting-factorization lengths: cA={len(cA_lifting)}, cD={len(cD_lifting)}")
    print()

    tt_wavelet = None
    if args.skip_tt_wavelet:
        print("tt-wavelet run skipped.")
        print()
    else:
        tt_wavelet = run_tt_wavelet(scheme_path, args.signal)
        print(
            f"tt-wavelet approximation coefficients: {format_coeffs(tt_wavelet['approximation'])}"
        )
        print(f"tt-wavelet detail coefficients: {format_coeffs(tt_wavelet['detail'])}")
        print(
            "tt-wavelet lengths: "
            f"cA={len(tt_wavelet['approximation'])}, cD={len(tt_wavelet['detail'])}"
        )
        print()

    print("Pairwise comparison")
    print_pairwise_mismatches(
        cA_pywt, cA_lifting, "Approximation pywt vs lifting-factorization", args.tolerance
    )
    print_pairwise_mismatches(
        cD_pywt, cD_lifting, "Detail pywt vs lifting-factorization", args.tolerance
    )
    print_error_metrics(cA_pywt, cA_lifting, "Approximation pywt -> lifting-factorization")
    print_error_metrics(cD_pywt, cD_lifting, "Detail pywt -> lifting-factorization")

    if tt_wavelet is not None:
        print_pairwise_mismatches(
            cA_pywt, tt_wavelet["approximation"], "Approximation pywt vs tt-wavelet", args.tolerance
        )
        print_pairwise_mismatches(
            cD_pywt, tt_wavelet["detail"], "Detail pywt vs tt-wavelet", args.tolerance
        )
        print_error_metrics(
            cA_pywt, tt_wavelet["approximation"], "Approximation pywt -> tt-wavelet"
        )
        print_error_metrics(cD_pywt, tt_wavelet["detail"], "Detail pywt -> tt-wavelet")
        print_pairwise_mismatches(
            cA_lifting,
            tt_wavelet["approximation"],
            "Approximation lifting-factorization vs tt-wavelet",
            args.tolerance,
        )
        print_pairwise_mismatches(
            cD_lifting,
            tt_wavelet["detail"],
            "Detail lifting-factorization vs tt-wavelet",
            args.tolerance,
        )
        if len(cA_pywt) != len(tt_wavelet["approximation"]) or len(cD_pywt) != len(
            tt_wavelet["detail"]
        ):
            print()
            print(
                "Note: tt-wavelet length still differs from pywt, so this comparison is not yet canonicalized."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
