#!/usr/bin/env python3
"""Run the ConeStreamed ILWT kernel chain for every generated scheme."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=root / "build" / "lwt")
    parser.add_argument("--wavelets", type=Path, default=root / "wavelets")
    parser.add_argument("--length", type=int, default=33)
    parser.add_argument(
        "--layout", choices=["auto", "row-major", "tile-native"], default="auto"
    )
    args = parser.parse_args()

    if args.length <= 0:
        parser.error("--length must be positive")
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary not found: {binary}")

    names = sorted(path.stem for path in args.wavelets.glob("*.json"))
    environment = os.environ.copy()
    environment["TT_WAVELET_LWT_CONE_WORKSPACE_LAYOUT"] = args.layout
    failures: list[str] = []
    for index, name in enumerate(names, start=1):
        command = [
            str(binary),
            "--inverse",
            "--benchmark",
            "--warmup-runs",
            "0",
            "--repeats",
            "1",
            "--length",
            str(args.length),
            name,
        ]
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=environment,
        )
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"[{index:3d}/{len(names)}] {status} {name}", flush=True)
        if result.returncode != 0:
            failures.append(f"{name}:\n{result.stderr}")

    print(f"validated_schemes: {len(names)}")
    print(f"failed_schemes: {len(failures)}")
    if failures:
        raise SystemExit("\n".join(["ILWT runtime stability failed:", *failures]))


if __name__ == "__main__":
    main()
