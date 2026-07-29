#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate fused 2D TT-LWT output against the exact FP32 route oracle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE_BINARY = PROJECT_ROOT / "build" / "lwt_2d"
REFERENCE_BINARY = PROJECT_ROOT / "build" / "lwt_2d_reference"
SET_ENV = PROJECT_ROOT / "scripts" / "set_env.sh"
SCHEME_DIR = (
    PROJECT_ROOT / "wavelets"
)
STABLE_MANIFEST = (
    PROJECT_ROOT
    / "tests"
    / "reference"
    / "lwt_2d_symmetric_stable_schemes.json"
)
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
REQUIRED_SHAPES = (
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


def parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for item in value.split(","):
        try:
            height_text, width_text = item.lower().split("x", 1)
            height, width = int(height_text), int(width_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid shape {item!r}; expected HEIGHTxWIDTH"
            ) from exc
        if height <= 0 or width <= 0:
            raise argparse.ArgumentTypeError("shape dimensions must be positive")
        shapes.append((height, width))
    return shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--schemes",
        default="db1",
        help="Comma-separated generated scheme names (default: %(default)s).",
    )
    group.add_argument(
        "--all-schemes",
        action="store_true",
        help="Validate every generated scheme.",
    )
    group.add_argument(
        "--stable-only",
        action="store_true",
        help="Validate schemes in the empirical symmetric-2D stable manifest.",
    )
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        default=list(REQUIRED_SHAPES),
        help="Comma-separated HEIGHTxWIDTH list (default: required corpus).",
    )
    parser.add_argument(
        "--modes",
        default="symmetric",
        help="Comma-separated signal-extension modes (default: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--input-type",
        choices=(
            "zero",
            "constant",
            "random",
            "ramp",
            "row-ramp",
            "column-ramp",
            "checkerboard",
            "corner-impulse",
            "tile-boundary-impulse",
        ),
        default="random",
    )
    parser.add_argument("--multi-core-limit", type=int, default=64)
    parser.add_argument(
        "--route-staging",
        choices=("scalar", "optimized"),
        default="optimized",
    )
    parser.add_argument(
        "--route-persistence",
        choices=("scalar", "full-tile"),
        default="full-tile",
    )
    parser.add_argument(
        "--terminal-writes",
        choices=("fragmented", "tiled"),
        default="tiled",
    )
    parser.add_argument(
        "--scale-policy",
        choices=("explicit", "fused"),
        default="fused",
    )
    parser.add_argument(
        "--route-config",
        choices=("per-route", "preloaded"),
        default="preloaded",
    )
    parser.add_argument(
        "--exact-transfer",
        choices=("local-noc", "l1-copy"),
        default="local-noc",
    )
    parser.add_argument(
        "--route-domain",
        choices=("exact", "tile-closed"),
        default="exact",
    )
    parser.add_argument(
        "--validate-route-domain-ab",
        action="store_true",
        help="Require exact and tile-closed route domains to produce bit-identical bands.",
    )
    parser.add_argument(
        "--fp32-arithmetic",
        choices=("wormhole-sfpu", "blackhole-sfpu"),
        default="wormhole-sfpu",
        help=(
            "Architecture-exact SFPU arithmetic used by the internal oracle "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--target-tolerance",
        type=float,
        default=1.0e-5,
        help="Report whether the preferred precision target is reached.",
    )
    parser.add_argument(
        "--skip-multi-core",
        action="store_true",
        help="Skip the bit-identical single-core versus multi-core check.",
    )
    parser.add_argument(
        "--validate-terminal-ab",
        action="store_true",
        help=(
            "Also run the fragmented terminal writer at the multi-core limit "
            "and require bit-identical bands versus the selected tiled writer."
        ),
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        help="Optional detailed result path.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first execution or validation error.",
    )
    return parser.parse_args()


def scheme_names(args: argparse.Namespace) -> list[str]:
    if args.all_schemes:
        return sorted(
            path.stem
            for path in SCHEME_DIR.glob("*.json")
        )
    if args.stable_only:
        manifest = json.loads(STABLE_MANIFEST.read_text(encoding="utf-8"))
        return [str(name) for name in manifest["stable_schemes"]]
    names = [name.strip() for name in args.schemes.split(",") if name.strip()]
    if not names:
        raise ValueError("--schemes did not contain a scheme name")
    return names


def boundary_modes(args: argparse.Namespace) -> list[str]:
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unsupported = sorted(set(modes) - set(BOUNDARY_MODES))
    if unsupported:
        raise ValueError(f"unsupported signal-extension modes: {', '.join(unsupported)}")
    if not modes:
        raise ValueError("--modes did not contain a signal-extension mode")
    return modes


def make_input(
    height: int, width: int, input_type: str, seed: int
) -> np.ndarray:
    if input_type == "zero":
        return np.zeros((height, width), dtype=np.float32)
    if input_type == "constant":
        return np.full((height, width), np.float32(0.375), dtype=np.float32)
    if input_type == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(-1.0, 1.0, size=(height, width)).astype(np.float32)
    if input_type == "ramp":
        return np.linspace(
            -1.0, 1.0, num=height * width, dtype=np.float32
        ).reshape(height, width)
    if input_type == "row-ramp":
        values = np.linspace(-1.0, 1.0, num=height, dtype=np.float32)
        return np.broadcast_to(values[:, None], (height, width)).copy()
    if input_type == "column-ramp":
        values = np.linspace(-1.0, 1.0, num=width, dtype=np.float32)
        return np.broadcast_to(values[None, :], (height, width)).copy()
    if input_type == "checkerboard":
        rows, columns = np.indices((height, width))
        return np.where((rows + columns) & 1, -1.0, 1.0).astype(np.float32)
    result = np.zeros((height, width), dtype=np.float32)
    if input_type == "corner-impulse":
        result[0, 0] = np.float32(1.0)
        result[0, width - 1] = np.float32(-0.75)
        result[height - 1, 0] = np.float32(0.5)
        result[height - 1, width - 1] = np.float32(-0.25)
        return result
    rows = sorted({min(height - 1, index) for index in (15, 16, 31, 32, 63, 64)})
    columns = sorted({min(width - 1, index) for index in (15, 16, 31, 32, 63, 64)})
    for impulse, (row, column) in enumerate(zip(rows, columns), start=1):
        result[row, column] = np.float32(((-1.0) ** impulse) / impulse)
    return result


def run(
    command: list[str], timeout_seconds: float, *, device: bool = False
) -> subprocess.CompletedProcess[str]:
    if device:
        quoted = " ".join(
            "'" + item.replace("'", "'\"'\"'") + "'" for item in command
        )
        command = [
            "bash",
            "-lc",
            f"source '{SET_ENV}' && {quoted}",
        ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}{completed.stderr}"
        )
    return completed


def read_reference(prefix: Path) -> dict[str, np.ndarray]:
    return {
        band: np.fromfile(
            Path(f"{prefix}.{band.lower()}.f32"), dtype=np.float32
        )
        for band in BANDS
    }


def read_device(prefix: Path) -> dict[str, np.ndarray]:
    return {
        band: np.fromfile(Path(f"{prefix}_{band}.f32"), dtype=np.float32)
        for band in BANDS
    }


def run_reference(
    scheme: str,
    mode: str,
    height: int,
    width: int,
    input_path: Path,
    prefix: Path,
    timeout_seconds: float,
    fp32_arithmetic: str,
) -> dict[str, np.ndarray]:
    run(
        [
            str(REFERENCE_BINARY),
            "--binary-input",
            "--boundary-mode",
            mode,
            "--fp32-arithmetic",
            fp32_arithmetic,
            "--output-prefix",
            str(prefix),
            scheme,
            str(height),
            str(width),
            str(input_path),
        ],
        timeout_seconds,
    )
    return read_reference(prefix)


def run_device(
    scheme: str,
    mode: str,
    height: int,
    width: int,
    input_path: Path,
    prefix: Path,
    cores: int,
    timeout_seconds: float,
    route_staging: str,
    route_persistence: str,
    terminal_writes: str,
    scale_policy: str,
    route_config: str,
    exact_transfer: str,
    route_domain: str,
) -> dict[str, np.ndarray]:
    run(
        [
            str(DEVICE_BINARY),
            "--binary-input",
            "--boundary-mode",
            mode,
            "--quiet",
            "--output-prefix",
            str(prefix),
            "--cores",
            str(cores),
            "--route-staging",
            route_staging,
            "--route-persistence",
            route_persistence,
            "--terminal-writes",
            terminal_writes,
            "--scale-policy",
            scale_policy,
            "--route-config",
            route_config,
            "--exact-transfer",
            exact_transfer,
            "--route-domain",
            route_domain,
            scheme,
            str(height),
            str(width),
            str(input_path),
        ],
        timeout_seconds,
        device=True,
    )
    return read_device(prefix)


def validate_case(
    scheme: str,
    mode: str,
    shape: tuple[int, int],
    args: argparse.Namespace,
    directory: Path,
) -> dict[str, object]:
    height, width = shape
    input_path = directory / "input.f32"
    make_input(height, width, args.input_type, args.seed).tofile(input_path)
    reference = run_reference(
        scheme,
        mode,
        height,
        width,
        input_path,
        directory / "reference",
        args.timeout_seconds,
        args.fp32_arithmetic,
    )
    single = run_device(
        scheme,
        mode,
        height,
        width,
        input_path,
        directory / "single",
        1,
        args.timeout_seconds,
        args.route_staging,
        args.route_persistence,
        args.terminal_writes,
        args.scale_policy,
        args.route_config,
        args.exact_transfer,
        args.route_domain,
    )
    multi = None
    if not args.skip_multi_core:
        multi = run_device(
            scheme,
            mode,
            height,
            width,
            input_path,
            directory / "multi",
            args.multi_core_limit,
            args.timeout_seconds,
            args.route_staging,
            args.route_persistence,
            args.terminal_writes,
            args.scale_policy,
            args.route_config,
            args.exact_transfer,
            args.route_domain,
        )
    fragmented = None
    if args.validate_terminal_ab:
        if args.terminal_writes != "tiled":
            raise RuntimeError(
                "--validate-terminal-ab requires --terminal-writes tiled"
            )
        fragmented = run_device(
            scheme,
            mode,
            height,
            width,
            input_path,
            directory / "fragmented",
            args.multi_core_limit,
            args.timeout_seconds,
            args.route_staging,
            args.route_persistence,
            "fragmented",
            args.scale_policy,
            args.route_config,
            args.exact_transfer,
            args.route_domain,
        )
    alternate_domain = None
    if args.validate_route_domain_ab:
        alternate_domain = run_device(
            scheme,
            mode,
            height,
            width,
            input_path,
            directory / "alternate-domain",
            args.multi_core_limit,
            args.timeout_seconds,
            args.route_staging,
            args.route_persistence,
            args.terminal_writes,
            args.scale_policy,
            args.route_config,
            args.exact_transfer,
            "tile-closed" if args.route_domain == "exact" else "exact",
        )

    band_results: dict[str, dict[str, object]] = {}
    passed = True
    reached_target = True
    for band in BANDS:
        if reference[band].shape != single[band].shape:
            raise RuntimeError(
                f"{scheme} {shape} {band}: device/oracle shapes differ"
            )
        if not np.isfinite(single[band]).all():
            raise RuntimeError(
                f"{scheme} {shape} {band}: single-core output is nonfinite"
            )
        difference = np.abs(
            single[band].astype(np.float64)
            - reference[band].astype(np.float64)
        )
        max_abs = float(difference.max(initial=0.0))
        multi_identical: bool | None = None
        if multi is not None:
            multi_identical = single[band].tobytes() == multi[band].tobytes()
        terminal_identical: bool | None = None
        if fragmented is not None:
            selected = multi if multi is not None else single
            terminal_identical = (
                selected[band].tobytes() == fragmented[band].tobytes()
            )
        route_domain_identical: bool | None = None
        if alternate_domain is not None:
            selected = multi if multi is not None else single
            route_domain_identical = (
                selected[band].tobytes() == alternate_domain[band].tobytes()
            )
        band_passed = (
            max_abs <= args.tolerance
            and multi_identical is not False
            and terminal_identical is not False
            and route_domain_identical is not False
        )
        passed = passed and band_passed
        reached_target = reached_target and max_abs <= args.target_tolerance
        band_results[band] = {
            "max_abs_error": max_abs,
            "multi_core_bit_identical": multi_identical,
            "fragmented_tiled_bit_identical": terminal_identical,
            "route_domain_bit_identical": route_domain_identical,
            "passed": band_passed,
        }
    return {
        "scheme": scheme,
        "mode": mode,
        "shape": [height, width],
        "input_type": args.input_type,
        "seed": args.seed,
        "passed": passed,
        "reached_target": reached_target,
        "bands": band_results,
    }


def main() -> int:
    args = parse_args()
    if args.multi_core_limit <= 0:
        raise ValueError("--multi-core-limit must be positive")
    if not DEVICE_BINARY.exists() or not REFERENCE_BINARY.exists():
        raise FileNotFoundError(
            "Build lwt_2d and lwt_2d_reference before validation"
        )
    schemes = scheme_names(args)
    modes = boundary_modes(args)
    results: list[dict[str, object]] = []
    failed = 0
    target_misses = 0
    total = len(schemes) * len(modes) * len(args.shapes)
    case_index = 0
    for scheme in schemes:
        for mode in modes:
            for shape in args.shapes:
                case_index += 1
                with tempfile.TemporaryDirectory(
                    prefix="ttwv-lwt2d-"
                ) as temporary:
                    try:
                        result = validate_case(
                            scheme, mode, shape, args, Path(temporary)
                        )
                    except Exception as error:  # noqa: BLE001
                        if args.fail_fast:
                            raise
                        result = {
                            "scheme": scheme,
                            "mode": mode,
                            "shape": [shape[0], shape[1]],
                            "input_type": args.input_type,
                            "seed": args.seed,
                            "passed": False,
                            "reached_target": False,
                            "error": str(error),
                            "bands": {},
                        }
                results.append(result)
                failed += not bool(result["passed"])
                target_misses += not bool(result["reached_target"])
                worst = max(
                    (
                        float(band["max_abs_error"])
                        for band in result["bands"].values()
                    ),
                    default=float("inf"),
                )
                print(
                    f"[{case_index}/{total}] {scheme} {mode} {shape[0]}x{shape[1]} "
                    f"{'PASS' if result['passed'] else 'FAIL'} "
                    f"max_abs={worst:.8e} "
                    f"target={'yes' if result['reached_target'] else 'no'}"
                )
    report = {
        "schema_version": 1,
        "fp32_arithmetic": args.fp32_arithmetic,
        "boundary_modes": modes,
        "hard_tolerance": args.tolerance,
        "target_tolerance": args.target_tolerance,
        "multi_core_limit": args.multi_core_limit,
        "multi_core_checked": not args.skip_multi_core,
        "case_count": len(results),
        "failed_case_count": failed,
        "target_miss_count": target_misses,
        "results": results,
    }
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"validated {len(results)} cases: failures={failed}, "
        f"target_misses={target_misses}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
