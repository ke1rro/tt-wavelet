#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate device EE/EO/OE/OO snapshots against a scalar split reference."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE_BINARY = PROJECT_ROOT / "build" / "lwt_2d"
SET_ENV = PROJECT_ROOT / "scripts" / "set_env.sh"
SCHEME_DIR = PROJECT_ROOT / "wavelets"
REQUIRED_SHAPES = (
    (1, 1),
    (1, 7),
    (7, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (15, 17),
    (31, 31),
    (31, 32),
    (32, 31),
    (32, 32),
    (32, 33),
    (33, 32),
    (33, 33),
    (63, 63),
    (63, 64),
    (64, 63),
    (64, 64),
    (64, 65),
    (65, 64),
    (65, 65),
    (1000, 100),
)
PATTERNS = (
    "zeros",
    "constant",
    "sequence",
    "row-ramp",
    "column-ramp",
    "checkerboard",
    "corner-impulses",
    "row-31-impulse",
    "row-32-impulse",
    "column-31-impulse",
    "column-32-impulse",
    "random",
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


def parse_patterns(value: str) -> list[str]:
    patterns = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(patterns) - set(PATTERNS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown patterns: {', '.join(unknown)}"
        )
    return patterns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelet", default="db7")
    parser.add_argument(
        "--implementation",
        choices=("scalar", "tiled"),
        default="tiled",
    )
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        default=list(REQUIRED_SHAPES),
        help="Comma-separated HEIGHTxWIDTH list (default: required corpus).",
    )
    parser.add_argument(
        "--patterns",
        type=parse_patterns,
        default=list(PATTERNS),
        help="Comma-separated input patterns (default: required corpus).",
    )
    parser.add_argument("--cores", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--keep-work-dir", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def make_input(
    height: int, width: int, pattern: str, seed: int
) -> np.ndarray:
    if pattern == "zeros":
        return np.zeros((height, width), dtype=np.float32)
    if pattern == "constant":
        return np.full((height, width), np.float32(3.25), dtype=np.float32)
    if pattern == "sequence":
        return np.arange(height * width, dtype=np.float32).reshape(height, width)
    if pattern == "row-ramp":
        return np.broadcast_to(
            np.arange(height, dtype=np.float32)[:, None], (height, width)
        ).copy()
    if pattern == "column-ramp":
        return np.broadcast_to(
            np.arange(width, dtype=np.float32)[None, :], (height, width)
        ).copy()
    if pattern == "checkerboard":
        rows, columns = np.indices((height, width))
        return np.where((rows + columns) & 1, -1.0, 1.0).astype(np.float32)
    if pattern == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(-1.0, 1.0, size=(height, width)).astype(np.float32)

    result = np.zeros((height, width), dtype=np.float32)
    if pattern == "corner-impulses":
        result[0, 0] += np.float32(1.0)
        result[0, width - 1] += np.float32(2.0)
        result[height - 1, 0] += np.float32(4.0)
        result[height - 1, width - 1] += np.float32(8.0)
        return result
    if pattern.startswith("row-"):
        row = int(pattern.split("-", 2)[1])
        if row < height:
            result[row, width // 2] = np.float32(1.0)
        return result
    column = int(pattern.split("-", 2)[1])
    if column < width:
        result[height // 2, column] = np.float32(1.0)
    return result


def symmetric_indices(indices: np.ndarray, length: int) -> np.ndarray:
    phase = np.mod(indices, 2 * length)
    return np.where(phase < length, phase, 2 * length - 1 - phase)


def expected_plane(
    logical_input: np.ndarray,
    snapshot: dict[str, object],
    pad: int,
) -> np.ndarray:
    y_begin = int(snapshot["y_begin"])
    x_begin = int(snapshot["x_begin"])
    height = int(snapshot["height"])
    width = int(snapshot["width"])
    parity_y = int(snapshot["parity_y"])
    parity_x = int(snapshot["parity_x"])
    polyphase_y = np.arange(y_begin, y_begin + height, dtype=np.int64)
    polyphase_x = np.arange(x_begin, x_begin + width, dtype=np.int64)
    source_y = symmetric_indices(
        2 * polyphase_y + parity_y - pad, logical_input.shape[0]
    )
    source_x = symmetric_indices(
        2 * polyphase_x + parity_x - pad, logical_input.shape[1]
    )
    return logical_input[np.ix_(source_y, source_x)]


def run_device(
    args: argparse.Namespace,
    height: int,
    width: int,
    input_path: Path,
    prefix: Path,
) -> None:
    command = [
        str(DEVICE_BINARY),
        "--binary-input",
        "--quiet",
        "--cores",
        str(args.cores),
        "--split-implementation",
        args.implementation,
        "--split-snapshot-prefix",
        str(prefix),
        args.wavelet,
        str(height),
        str(width),
        str(input_path),
    ]
    shell_command = (
        f"source {shlex.quote(str(SET_ENV))} >/dev/null && "
        + " ".join(shlex.quote(item) for item in command)
    )
    completed = subprocess.run(
        ["bash", "-lc", shell_command],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=args.timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"device command failed ({completed.returncode})\n"
            f"{completed.stdout}{completed.stderr}"
        )


def validate_case(
    args: argparse.Namespace,
    work_dir: Path,
    pad: int,
    height: int,
    width: int,
    pattern: str,
    case_index: int,
) -> dict[str, object]:
    logical_input = make_input(
        height, width, pattern, args.seed + case_index
    )
    case_name = f"{height}x{width}_{pattern}"
    input_path = work_dir / f"{case_name}.f32"
    prefix = work_dir / case_name
    logical_input.tofile(input_path)
    run_device(args, height, width, input_path, prefix)

    manifest_path = Path(f"{prefix}_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = 0
    checked_values = 0
    for snapshot in manifest["snapshots"]:
        actual = np.fromfile(
            work_dir / str(snapshot["file"]), dtype=np.float32
        ).reshape(int(snapshot["height"]), int(snapshot["width"]))
        expected = expected_plane(logical_input, snapshot, pad)
        unequal = actual.view(np.uint32) != expected.view(np.uint32)
        mismatches += int(np.count_nonzero(unequal))
        checked_values += actual.size
    return {
        "shape": f"{height}x{width}",
        "pattern": pattern,
        "snapshot_count": len(manifest["snapshots"]),
        "checked_values": checked_values,
        "mismatches": mismatches,
        "passed": mismatches == 0,
    }


def main() -> int:
    args = parse_args()
    if args.cores <= 0:
        raise ValueError("--cores must be positive")
    scheme_path = SCHEME_DIR / f"{args.wavelet}.json"
    scheme = json.loads(scheme_path.read_text(encoding="utf-8"))
    pad = int(scheme["tap_size"]) - 1

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_work_dir:
        work_dir = args.keep_work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="ttwv-split-validate-")
        work_dir = Path(temporary.name)

    results: list[dict[str, object]] = []
    try:
        case_index = 0
        for height, width in args.shapes:
            for pattern in args.patterns:
                try:
                    result = validate_case(
                        args,
                        work_dir,
                        pad,
                        height,
                        width,
                        pattern,
                        case_index,
                    )
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    result = {
                        "shape": f"{height}x{width}",
                        "pattern": pattern,
                        "passed": False,
                        "error": str(exc),
                    }
                results.append(result)
                status = "PASS" if result["passed"] else "FAIL"
                print(f"{status} {height}x{width} {pattern}")
                if not result["passed"] and args.fail_fast:
                    break
                case_index += 1
            if results and not results[-1]["passed"] and args.fail_fast:
                break
    finally:
        if temporary is not None:
            temporary.cleanup()

    summary = {
        "wavelet": args.wavelet,
        "implementation": args.implementation,
        "pad": pad,
        "case_count": len(results),
        "passed": sum(bool(item["passed"]) for item in results),
        "failed": sum(not bool(item["passed"]) for item in results),
        "results": results,
    }
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"split validation: {summary['passed']}/{summary['case_count']} passed"
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
