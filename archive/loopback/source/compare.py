#!/usr/bin/env python3
import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
LIFTING_USAGE_DIR = PROJECT_ROOT / "lifting-factorization" / "usage"
TT_WAVELET_BINARY = PROJECT_ROOT / "build" / "tt-wavelet" / "lwt"
TT_WAVELET_ENV = PROJECT_ROOT / "scripts" / "set_env.sh"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python3"

STEP_COEFF_CAPACITY = 17
PU_STEP_TYPES = {"predict", "update"}
VALID_STEP_TYPES = {"predict", "update", "scale-even", "scale-odd", "swap"}

PU_HEADER_WORDS = 1
PU_READER_ARGS_PER_STEP = 9
PU_COMPUTE_ARGS_PER_STEP = 20
PU_WRITER_ARGS_PER_STEP = 2

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
        default="sym12",
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
    parser.add_argument(
        "--all-green",
        action="store_true",
        help="Run comparisons for all runtime-capacity green schemes.",
    )
    parser.add_argument(
        "--runtime-limit",
        type=int,
        default=341,
        help="Runtime-args limit used to classify green schemes (default: %(default)s).",
    )
    parser.add_argument(
        "--schemes-dir",
        type=Path,
        default=PROJECT_ROOT / "ttnn-wavelet" / "lifting_schemes",
        help="Directory with lifting-scheme JSON files (default: %(default)s).",
    )
    return parser.parse_args()


def parse_signal(raw_signal: str) -> list[float]:
    return [float(token.strip()) for token in raw_signal.split(",") if token.strip()]


def format_coeffs(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(value):.8e}" for value in values) + "]"


def print_pairwise_mismatches(
    lhs: Sequence[float], rhs: Sequence[float], name: str, tolerance: float
) -> int:
    mismatches = 0

    if len(lhs) != len(rhs):
        print(f"{name} length differs: lhs={len(lhs)} vs rhs={len(rhs)}")
        return 1

    for i, (lhs_val, rhs_val) in enumerate(zip(lhs, rhs)):
        if abs(float(lhs_val) - float(rhs_val)) > tolerance:
            mismatches += 1
            print(f"{name} coeff {i} differs: lhs={float(lhs_val):.8e} vs rhs={float(rhs_val):.8e}")

    if mismatches == 0:
        print(f"{name}: all coefficients match within tolerance {tolerance}")
    return mismatches


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
            f"TT-wavelet binary not found at {TT_WAVELET_BINARY}. Rebuild with ./update.sh Release lwt"
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


def max_predict_update_segment_steps(steps: Sequence[dict[str, object]]) -> int:
    max_count = 0
    current_count = 0
    for step in steps:
        step_type = str(step.get("type", "")).strip()
        if step_type in PU_STEP_TYPES:
            current_count += 1
            max_count = max(max_count, current_count)
            continue

        if step_type == "swap":
            continue

        current_count = 0

    return max_count


def is_green_scheme(scheme_path: Path, runtime_limit: int) -> tuple[bool, list[str]]:
    errors: list[str] = []

    with scheme_path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)

    tap_size = obj.get("tap_size", 0)
    if not isinstance(tap_size, int) or tap_size <= 0:
        errors.append("tap_size must be positive")

    steps_obj = obj.get("steps", [])
    if not isinstance(steps_obj, list) or len(steps_obj) == 0:
        errors.append("missing or empty steps")
        steps_obj = []

    steps: list[dict[str, object]] = []
    for i, step in enumerate(steps_obj):
        if not isinstance(step, dict):
            errors.append(f"step {i}: invalid entry")
            continue

        step_type = str(step.get("type", "")).strip()
        if step_type not in VALID_STEP_TYPES:
            errors.append(f"step {i}: unsupported type '{step_type}'")

        coeffs = step.get("coefficients", [])
        if not isinstance(coeffs, list):
            errors.append(f"step {i}: coefficients must be a list")
            coeff_count = 0
        else:
            coeff_count = len(coeffs)

        if coeff_count > STEP_COEFF_CAPACITY:
            errors.append(
                f"step {i}: coefficients {coeff_count} exceed capacity {STEP_COEFF_CAPACITY}"
            )

        steps.append(step)

    max_segment = max_predict_update_segment_steps(steps)
    pu_reader_args = PU_HEADER_WORDS + max_segment * PU_READER_ARGS_PER_STEP
    pu_compute_args = PU_HEADER_WORDS + max_segment * PU_COMPUTE_ARGS_PER_STEP
    pu_writer_args = PU_HEADER_WORDS + max_segment * PU_WRITER_ARGS_PER_STEP

    if pu_reader_args > runtime_limit:
        errors.append(f"predict/update reader args {pu_reader_args} > limit {runtime_limit}")
    if pu_compute_args > runtime_limit:
        errors.append(f"predict/update compute args {pu_compute_args} > limit {runtime_limit}")
    if pu_writer_args > runtime_limit:
        errors.append(f"predict/update writer args {pu_writer_args} > limit {runtime_limit}")

    return len(errors) == 0, errors


def discover_green_wavelets(schemes_dir: Path, runtime_limit: int) -> tuple[list[str], list[str]]:
    if not schemes_dir.exists() or not schemes_dir.is_dir():
        raise FileNotFoundError(f"Schemes directory not found: {schemes_dir}")

    json_files = sorted(schemes_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON schemes found in: {schemes_dir}")

    green: list[str] = []
    red: list[str] = []
    for path in json_files:
        ok, _ = is_green_scheme(path, runtime_limit)
        if ok:
            green.append(path.stem)
        else:
            red.append(path.stem)

    return green, red


def run_single_comparison(args: argparse.Namespace, wavelet: str) -> bool:
    signal = parse_signal(args.signal)
    scheme_path = args.schemes_dir / f"{wavelet}.json"

    if not scheme_path.exists():
        raise FileNotFoundError(f"Wavelet scheme file not found: {scheme_path}")

    print(f"signal: {signal}")
    print(f"scheme path: {scheme_path}")
    print()

    cA_pywt, cD_pywt = dwt(signal, wavelet, mode="symmetric")

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
    pywt_vs_lifting_a = print_pairwise_mismatches(
        cA_pywt, cA_lifting, "Approximation pywt vs lifting-factorization", args.tolerance
    )
    pywt_vs_lifting_d = print_pairwise_mismatches(
        cD_pywt, cD_lifting, "Detail pywt vs lifting-factorization", args.tolerance
    )
    print_error_metrics(cA_pywt, cA_lifting, "Approximation pywt -> lifting-factorization")
    print_error_metrics(cD_pywt, cD_lifting, "Detail pywt -> lifting-factorization")

    checks_ok = pywt_vs_lifting_a == 0 and pywt_vs_lifting_d == 0

    if tt_wavelet is not None:
        pywt_vs_tt_a = print_pairwise_mismatches(
            cA_pywt, tt_wavelet["approximation"], "Approximation pywt vs tt-wavelet", args.tolerance
        )
        pywt_vs_tt_d = print_pairwise_mismatches(
            cD_pywt, tt_wavelet["detail"], "Detail pywt vs tt-wavelet", args.tolerance
        )
        print_error_metrics(
            cA_pywt, tt_wavelet["approximation"], "Approximation pywt -> tt-wavelet"
        )
        print_error_metrics(cD_pywt, tt_wavelet["detail"], "Detail pywt -> tt-wavelet")
        lifting_vs_tt_a = print_pairwise_mismatches(
            cA_lifting,
            tt_wavelet["approximation"],
            "Approximation lifting-factorization vs tt-wavelet",
            args.tolerance,
        )
        lifting_vs_tt_d = print_pairwise_mismatches(
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

        checks_ok = checks_ok and all(
            x == 0
            for x in [
                pywt_vs_tt_a,
                pywt_vs_tt_d,
                lifting_vs_tt_a,
                lifting_vs_tt_d,
            ]
        )

    return checks_ok


def main() -> int:
    args = parse_args()
    if args.all_green:
        green, red = discover_green_wavelets(args.schemes_dir, args.runtime_limit)
        if not green:
            print("Green schemes: none")
            print(f"Red schemes: {len(red)}")
            return 1

        print(f"Runtime-args green schemes ({len(green)}): {', '.join(green)}")
        print(f"Runtime-args red schemes ({len(red)}): {', '.join(red)}")
        print()

        passed: list[str] = []
        failed: list[str] = []
        for i, wavelet in enumerate(green, start=1):
            print("=" * 80)
            print(f"[{i}/{len(green)}] Testing wavelet: {wavelet}")
            print("=" * 80)
            try:
                ok = run_single_comparison(args, wavelet)
            except Exception as exc:  # noqa: BLE001
                ok = False
                print(f"ERROR while testing {wavelet}: {exc}")

            if ok:
                passed.append(wavelet)
                print(f"RESULT {wavelet}: PASS")
            else:
                failed.append(wavelet)
                print(f"RESULT {wavelet}: FAIL")
            print()

        print("Summary")
        print(f"Passed: {len(passed)}")
        if passed:
            print(f"Passed list: {', '.join(passed)}")
        print(f"Failed: {len(failed)}")
        if failed:
            print(f"Failed list: {', '.join(failed)}")

        return 0 if not failed else 1

    return 0 if run_single_comparison(args, args.wavelet) else 1


if __name__ == "__main__":
    raise SystemExit(main())


# from pywt import dwt


# signal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# cA, cD = dwt(signal, "coif2", mode="symmetric")

# print(f"signal: {signal}")
# print(f"pywt approximation coefficients: {cA}")
# print(f"pywt detail coefficients: {cD}")
# print(len(cA), len(cD))
