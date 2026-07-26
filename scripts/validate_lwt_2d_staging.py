#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Directly validate optimized 2D route CB staging before SFPU consumption."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE_BINARY = PROJECT_ROOT / "build" / "lwt_2d"
SET_ENV = PROJECT_ROOT / "scripts" / "set_env.sh"
DEFAULT_SCHEMES = ("db1", "db7", "bior3.9", "synthetic-k17")
DEFAULT_SHAPES = (
    (1, 1),
    (2, 3),
    (15, 17),
    (31, 31),
    (32, 32),
    (32, 33),
    (33, 32),
    (33, 33),
    (63, 65),
    (64, 64),
    (65, 63),
    (1000, 100),
    (1000, 200),
    (513, 769),
    (1024, 1024),
)
VALIDATED_PATTERN = re.compile(
    r"lwt_2d_transport_validated_staging_tiles:\s*(\d+)"
)
MISMATCH_PATTERN = re.compile(
    r"lwt_2d_transport_staging_validation_mismatches:\s*(\d+)"
)
PERSISTED_PATTERN = re.compile(
    r"lwt_2d_transport_validated_persistence_tiles:\s*(\d+)"
)
PERSISTENCE_MISMATCH_PATTERN = re.compile(
    r"lwt_2d_transport_persistence_validation_mismatches:\s*(\d+)"
)


def parse_shapes(value: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for item in value.split(","):
        try:
            height_text, width_text = item.lower().split("x", 1)
            shape = (int(height_text), int(width_text))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid shape {item!r}; expected HEIGHTxWIDTH"
            ) from exc
        if min(shape) <= 0:
            raise argparse.ArgumentTypeError("shape dimensions must be positive")
        result.append(shape)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schemes",
        default=",".join(DEFAULT_SCHEMES),
        help="Comma-separated generated scheme names (default: representative set).",
    )
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        default=list(DEFAULT_SHAPES),
        help="Comma-separated HEIGHTxWIDTH corpus (default: required transport corpus).",
    )
    parser.add_argument("--cores", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def run_case(
    scheme: str,
    shape: tuple[int, int],
    input_path: Path,
    cores: int,
    timeout_seconds: float,
) -> tuple[int, int, int, int]:
    height, width = shape
    command = [
        str(DEVICE_BINARY),
        "--binary-input",
        "--benchmark",
        "--repeats",
        "1",
        "--warmup-runs",
        "0",
        "--cores",
        str(cores),
        "--validate-route-staging",
        scheme,
        str(height),
        str(width),
        str(input_path),
    ]
    quoted = " ".join(shell_quote(item) for item in command)
    completed = subprocess.run(
        ["bash", "-lc", f"source {shell_quote(str(SET_ENV))} && {quoted}"],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(output)
    validated_match = VALIDATED_PATTERN.search(output)
    mismatch_match = MISMATCH_PATTERN.search(output)
    persisted_match = PERSISTED_PATTERN.search(output)
    persistence_mismatch_match = PERSISTENCE_MISMATCH_PATTERN.search(output)
    if (
        validated_match is None
        or mismatch_match is None
        or persisted_match is None
        or persistence_mismatch_match is None
    ):
        raise RuntimeError("device output omitted route-transport validation counters")
    return (
        int(validated_match.group(1)),
        int(mismatch_match.group(1)),
        int(persisted_match.group(1)),
        int(persistence_mismatch_match.group(1)),
    )


def main() -> int:
    args = parse_args()
    if not DEVICE_BINARY.exists():
        raise SystemExit("Build lwt_2d before staging validation")
    if args.cores <= 0:
        raise SystemExit("--cores must be positive")
    schemes = [item.strip() for item in args.schemes.split(",") if item.strip()]
    cases: list[dict[str, object]] = []
    total_validated = 0
    total_mismatches = 0
    total_persisted = 0
    total_persistence_mismatches = 0
    failures = 0
    with tempfile.TemporaryDirectory(prefix="lwt-2d-staging-") as temporary:
        input_path = Path(temporary) / "input.f32"
        for scheme_index, scheme in enumerate(schemes):
            for shape_index, shape in enumerate(args.shapes):
                rng = np.random.default_rng(
                    args.seed + scheme_index * len(args.shapes) + shape_index
                )
                rng.uniform(-1.0, 1.0, size=shape).astype(np.float32).tofile(
                    input_path
                )
                try:
                    (
                        validated,
                        mismatches,
                        persisted,
                        persistence_mismatches,
                    ) = run_case(
                        scheme,
                        shape,
                        input_path,
                        args.cores,
                        args.timeout_seconds,
                    )
                    passed = mismatches == 0 and persistence_mismatches == 0
                    error = ""
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    validated = 0
                    mismatches = 0
                    persisted = 0
                    persistence_mismatches = 0
                    passed = False
                    error = str(exc)
                total_validated += validated
                total_mismatches += mismatches
                total_persisted += persisted
                total_persistence_mismatches += persistence_mismatches
                failures += not passed
                cases.append(
                    {
                        "scheme": scheme,
                        "height": shape[0],
                        "width": shape[1],
                        "validated_fast_path_tiles": validated,
                        "mismatched_words": mismatches,
                        "validated_persistence_tiles": persisted,
                        "persistence_mismatched_words": persistence_mismatches,
                        "passed": passed,
                        "error": error,
                    }
                )
                if args.fail_fast and not passed:
                    break
            if args.fail_fast and failures:
                break
    result = {
        "case_count": len(cases),
        "passed": len(cases) - failures,
        "validated_fast_path_tiles": total_validated,
        "mismatched_words": total_mismatches,
        "validated_persistence_tiles": total_persisted,
        "persistence_mismatched_words": total_persistence_mismatches,
        "cases": cases,
    }
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "route staging validation: "
        f"{result['passed']}/{result['case_count']} passed; "
        f"{total_validated} optimized CB tiles; "
        f"{total_mismatches} staging mismatches; "
        f"{total_persisted} persisted tiles; "
        f"{total_persistence_mismatches} persistence mismatches"
    )
    return (
        0
        if failures == 0
        and total_mismatches == 0
        and total_persistence_mismatches == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
