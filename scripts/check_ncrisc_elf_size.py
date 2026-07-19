#!/usr/bin/env python3
"""Gate Wormhole LWT NCRISC instruction size from generated ELF metadata."""

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


WORMHOLE_TEXT_LIMIT_BYTES = 0x4000


@dataclass(frozen=True)
class ElfTextSize:
    kernel: str
    architecture: str
    path: Path
    text_bytes: int


def architecture_from_dependency_file(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"missing NCRISC dependency metadata: {path}")
    dependencies = path.read_text(encoding="utf-8", errors="replace")
    if "/wormhole/" in dependencies or "tt_llk_wormhole" in dependencies:
        return "wormhole_b0"
    if "/blackhole/" in dependencies or "tt_llk_blackhole" in dependencies:
        return "blackhole"
    raise RuntimeError(f"cannot determine architecture from NCRISC metadata: {path}")


def text_size(size_tool: Path, elf_path: Path) -> int:
    result = subprocess.run(
        [str(size_tool), "-A", str(elf_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ELF size tool failed for {elf_path}:\n{result.stdout}{result.stderr}"
        )
    match = re.search(r"^\.text\s+(\d+)\s+", result.stdout, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"ELF metadata has no .text section: {elf_path}")
    return int(match.group(1))


def collect_elf_sizes(
    cache_root: Path,
    size_tool: Path,
    kernel_name: str,
    architecture: str,
) -> list[ElfTextSize]:
    sizes: list[ElfTextSize] = []
    pattern = f"**/kernels/{kernel_name}/*/ncrisc/ncrisc.elf"
    for elf_path in sorted(cache_root.glob(pattern)):
        build_directory = elf_path.parents[1]
        if not (build_directory / ".SUCCESS").is_file():
            continue
        detected_architecture = architecture_from_dependency_file(
            elf_path.with_name("ncrisck.d")
        )
        if detected_architecture != architecture:
            continue
        sizes.append(
            ElfTextSize(
                kernel=f"{kernel_name}/{build_directory.name}",
                architecture=detected_architecture,
                path=elf_path,
                text_bytes=text_size(size_tool, elf_path),
            )
        )
    return sizes


def print_result(result: ElfTextSize) -> bool:
    print(f"kernel: {result.kernel}")
    print(f"architecture: {result.architecture}")
    print(f".text bytes: {result.text_bytes}")
    print(f"limit bytes: {WORMHOLE_TEXT_LIMIT_BYTES}")
    if result.text_bytes <= WORMHOLE_TEXT_LIMIT_BYTES:
        print(f"headroom bytes: {WORMHOLE_TEXT_LIMIT_BYTES - result.text_bytes}")
        print("result: PASS")
        return True
    print(f"overflow bytes: {result.text_bytes - WORMHOLE_TEXT_LIMIT_BYTES}")
    print("result: FAIL")
    return False


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=["wormhole_b0", "blackhole"],
        required=True,
        help="Architecture reported by the hardware run that generated the ELF.",
    )
    parser.add_argument("--kernel", default="lwt_reader")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".cache" / "tt-metal-cache",
    )
    parser.add_argument(
        "--size-tool",
        type=Path,
        default=root
        / "tt-metal"
        / "runtime"
        / "sfpi"
        / "compiler"
        / "bin"
        / "riscv-tt-elf-size",
    )
    args = parser.parse_args()

    if args.architecture == "blackhole":
        print(f"kernel: {args.kernel}")
        print("architecture: blackhole")
        print("result: SKIP (the Wormhole NCRISC limit does not apply)")
        return 0

    if not args.cache_root.is_dir():
        parser.error(f"TT-Metal cache root not found: {args.cache_root}")
    if not args.size_tool.is_file():
        parser.error(f"ELF size tool not found: {args.size_tool}")

    results = collect_elf_sizes(
        args.cache_root.resolve(),
        args.size_tool.resolve(),
        args.kernel,
        args.architecture,
    )
    if not results:
        parser.error(
            f"no successful {args.architecture} NCRISC ELF found for kernel {args.kernel}; "
            "run an LWT/ILWT device test first"
        )

    passed = True
    for result in results:
        passed = print_result(result) and passed
    largest = max(results, key=lambda result: result.text_bytes)
    print(f"checked_ncrisc_elfs: {len(results)}")
    print(f"maximum_text_kernel: {largest.kernel}")
    print(f"maximum_text_bytes: {largest.text_bytes}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
