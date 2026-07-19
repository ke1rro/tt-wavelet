#!/usr/bin/env python3
"""Run the production ILWT kernel chain for every generated scheme."""

import argparse
import os
import subprocess
from pathlib import Path

from runtime_checks import (
    check_consistent_architecture,
    parse_runtime_architecture,
    run_ncrisc_elf_gate,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=root / "build" / "lwt")
    parser.add_argument("--wavelets", type=Path, default=root / "wavelets")
    parser.add_argument("--length", type=int, default=33)
    parser.add_argument("--layout", choices=["auto", "row-major", "tile-native"], default="auto")
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
        default=["symmetric"],
    )
    args = parser.parse_args()

    if args.length <= 0:
        parser.error("--length must be positive")
    if args.length == 1 and any(mode in {"reflect", "antireflect"} for mode in args.modes):
        parser.error("reflect and antireflect modes require --length greater than one")
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary not found: {binary}")

    names = sorted(path.stem for path in args.wavelets.glob("*.json"))
    environment = os.environ.copy()
    environment["TT_WAVELET_LWT_WORKSPACE_LAYOUT"] = args.layout
    failures: list[str] = []
    architecture: str | None = None
    case_count = len(names) * len(args.modes)
    case_index = 0
    for name in names:
        for mode in args.modes:
            case_index += 1
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
                "--boundary-mode",
                mode,
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
            print(
                f"[{case_index:3d}/{case_count}] {status} {name} mode={mode}",
                flush=True,
            )
            if result.returncode != 0:
                failures.append(f"{name} mode={mode}:\n{result.stderr}")
            else:
                architecture = check_consistent_architecture(
                    architecture, parse_runtime_architecture(result.stderr)
                )

    print(f"validated_schemes: {len(names)}")
    print(f"validated_device_cases: {case_count}")
    print(f"failed_schemes: {len(failures)}")
    if failures:
        raise SystemExit("\n".join(["ILWT runtime stability failed:", *failures]))
    if architecture is None:
        raise SystemExit("ILWT stability validation did not execute a device case")
    run_ncrisc_elf_gate(root, architecture)


if __name__ == "__main__":
    main()
