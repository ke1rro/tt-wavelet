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
    executable_segment_bytes: int
    elf_bytes: int


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
        raise RuntimeError(f"ELF size tool failed for {elf_path}:\n{result.stdout}{result.stderr}")
    match = re.search(r"^\.text\s+(\d+)\s+", result.stdout, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"ELF metadata has no .text section: {elf_path}")
    return int(match.group(1))


def executable_segment_size(readelf_tool: Path, elf_path: Path) -> int:
    result = subprocess.run(
        [str(readelf_tool), "-W", "-l", str(elf_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ELF readelf tool failed for {elf_path}:\n" f"{result.stdout}{result.stderr}"
        )

    executable_sizes: list[int] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[0] != "LOAD" or "E" not in fields[6:-1]:
            continue
        executable_sizes.append(int(fields[5], 16))
    if not executable_sizes:
        raise RuntimeError(f"ELF metadata has no executable LOAD segment: {elf_path}")
    return max(executable_sizes)


def collect_elf_sizes(
    cache_root: Path,
    size_tool: Path,
    readelf_tool: Path,
    kernel_name: str,
    architecture: str,
) -> list[ElfTextSize]:
    sizes: list[ElfTextSize] = []
    pattern = f"**/kernels/{kernel_name}/*/ncrisc/ncrisc.elf"
    for elf_path in sorted(cache_root.glob(pattern)):
        build_directory = elf_path.parents[1]
        if not (build_directory / ".SUCCESS").is_file() and not (elf_path.parent / ".build_state").is_file():
            continue
        dependency_file = elf_path.with_name("ncrisck.d")
        if not dependency_file.is_file():
            dependency_file = elf_path.with_name("ncrisck.o.dephash")
        detected_architecture = architecture_from_dependency_file(dependency_file)
        if detected_architecture != architecture:
            continue
        sizes.append(
            ElfTextSize(
                kernel=f"{kernel_name}/{build_directory.name}",
                architecture=detected_architecture,
                path=elf_path,
                text_bytes=text_size(size_tool, elf_path),
                executable_segment_bytes=executable_segment_size(readelf_tool, elf_path),
                elf_bytes=elf_path.stat().st_size,
            )
        )
    return sizes


def print_result(result: ElfTextSize) -> bool:
    print(f"kernel: {result.kernel}")
    print(f"architecture: {result.architecture}")
    print(f".text bytes: {result.text_bytes}")
    print(f"executable LOAD segment bytes: {result.executable_segment_bytes}")
    print(f"ELF file bytes: {result.elf_bytes}")
    print(f"limit bytes: {WORMHOLE_TEXT_LIMIT_BYTES}")
    if result.executable_segment_bytes <= WORMHOLE_TEXT_LIMIT_BYTES:
        print("headroom bytes: " f"{WORMHOLE_TEXT_LIMIT_BYTES - result.executable_segment_bytes}")
        print("result: PASS")
        return True
    print("overflow bytes: " f"{result.executable_segment_bytes - WORMHOLE_TEXT_LIMIT_BYTES}")
    print("result: FAIL")
    return False


def print_blackhole_result(result: ElfTextSize) -> None:
    print(f"kernel: {result.kernel}")
    print(f"architecture: {result.architecture}")
    print(f".text bytes: {result.text_bytes}")
    print(f"executable LOAD segment bytes: {result.executable_segment_bytes}")
    print(f"ELF file bytes: {result.elf_bytes}")
    print("result: REPORT (no Wormhole 0x4000 gate applied)")


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
        default=root / "tt-metal" / "runtime" / "sfpi" / "compiler" / "bin" / "riscv-tt-elf-size",
    )
    parser.add_argument(
        "--readelf-tool",
        type=Path,
        default=root
        / "tt-metal"
        / "runtime"
        / "sfpi"
        / "compiler"
        / "bin"
        / "riscv-tt-elf-readelf",
    )
    args = parser.parse_args()

    if not args.cache_root.is_dir():
        parser.error(f"TT-Metal cache root not found: {args.cache_root}")
    if not args.size_tool.is_file():
        parser.error(f"ELF size tool not found: {args.size_tool}")
    if not args.readelf_tool.is_file():
        parser.error(f"ELF readelf tool not found: {args.readelf_tool}")

    results = collect_elf_sizes(
        args.cache_root.resolve(),
        args.size_tool.resolve(),
        args.readelf_tool.resolve(),
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
        if args.architecture == "blackhole":
            print_blackhole_result(result)
        else:
            passed = print_result(result) and passed
    largest = max(results, key=lambda result: result.executable_segment_bytes)
    print(f"checked_ncrisc_elfs: {len(results)}")
    print(f"maximum_text_kernel: {largest.kernel}")
    print(f"maximum_text_bytes: {largest.text_bytes}")
    print(f"maximum_executable_segment_bytes: {largest.executable_segment_bytes}")
    print(f"maximum_elf_file_bytes: {largest.elf_bytes}")
    if args.architecture == "blackhole":
        print("result: PASS (reported only; Blackhole has no Wormhole NCRISC gate)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
