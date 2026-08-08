"""Idempotently register the out-of-tree Wavelet operation in TTNN."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

WAVELET_INCLUDE = '#include "ttnn/operations/wavelet/wavelet_nanobind.hpp"'
WAVELET_BIND = "wavelet::bind_wavelet_operations(mod);"
WAVELET_SUBDIRECTORY = "add_subdirectory(wavelet)"
WAVELET_LIBRARY = "TTNN::Ops::Wavelet"
WAVELET_OBJECTS = "$<TARGET_OBJECTS:TTNN::Ops::Wavelet>"


def write_atomic(path: Path, text: str) -> None:
    mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        output.write(text)
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def without_exact_line(lines: list[str], value: str) -> list[str]:
    return [line for line in lines if line.strip() != value]


def find_command_block(lines: list[str], command: str, target: str) -> tuple[int, int]:
    command_start = re.compile(rf"^\s*{re.escape(command)}\s*\(")
    for start, line in enumerate(lines):
        if not command_start.match(line):
            continue
        depth = 0
        for end in range(start, len(lines)):
            content = lines[end].split("#", 1)[0]
            depth += content.count("(") - content.count(")")
            if depth == 0:
                block = "\n".join(lines[start : end + 1])
                if target in block:
                    if start == end:
                        raise RuntimeError(
                            f"Unsupported one-line {command}({target}) command"
                        )
                    return start, end
                break
    raise RuntimeError(f"Could not find {command}(...) block for {target}")


def integrate_cmake(path: Path) -> bool:
    original = path.read_text()
    lines = original.splitlines()
    for hook in (WAVELET_SUBDIRECTORY, WAVELET_LIBRARY, WAVELET_OBJECTS):
        lines = without_exact_line(lines, hook)

    umbrella = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(r"^\s*add_library\s*\(\s*ttnn_op_operations_shared\b", line)
        ),
        None,
    )
    if umbrella is None:
        raise RuntimeError("Could not find ttnn_op_operations_shared declaration")
    subdirectories = [
        index
        for index, line in enumerate(lines[:umbrella])
        if re.match(r"^\s*add_subdirectory\s*\(", line)
    ]
    if not subdirectories:
        raise RuntimeError("Could not find TTNN operation add_subdirectory block")
    lines.insert(subdirectories[-1] + 1, WAVELET_SUBDIRECTORY)

    _, link_end = find_command_block(
        lines, "target_link_libraries", "ttnn_op_operations_shared"
    )
    lines.insert(link_end, f"        {WAVELET_LIBRARY}")
    _, sources_end = find_command_block(
        lines, "target_sources", "ttnn_op_operations_shared"
    )
    lines.insert(sources_end, f"        {WAVELET_OBJECTS}")

    _, link_end = find_command_block(
        lines, "target_link_libraries", "ttnn_op_operations_shared"
    )
    _, sources_end = find_command_block(
        lines, "target_sources", "ttnn_op_operations_shared"
    )
    library_index = next(
        index for index, line in enumerate(lines) if line.strip() == WAVELET_LIBRARY
    )
    objects_index = next(
        index for index, line in enumerate(lines) if line.strip() == WAVELET_OBJECTS
    )
    if not (library_index < link_end and objects_index < sources_end):
        raise RuntimeError("Wavelet hooks were not placed inside their CMake commands")

    updated = "\n".join(lines) + "\n"
    if updated == original:
        return False
    write_atomic(path, updated)
    return True


def integrate_nanobind(path: Path) -> bool:
    original = path.read_text()
    lines = without_exact_line(original.splitlines(), WAVELET_INCLUDE)
    lines = without_exact_line(lines, WAVELET_BIND)

    namespace_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip() == "namespace nb = nanobind;"
        ),
        None,
    )
    if namespace_index is None:
        raise RuntimeError("Could not find nanobind namespace declaration")
    include_index = namespace_index
    while include_index > 0 and not lines[include_index - 1].strip():
        include_index -= 1
    lines.insert(include_index, WAVELET_INCLUDE)

    function_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(
                r"^\s*void\s+py_module\s*\(\s*nb::module_&\s+mod\s*\)\s*\{", line
            )
        ),
        None,
    )
    if function_index is None:
        raise RuntimeError("Could not find TTNN nanobind py_module function")
    lines.insert(function_index + 1, f"    {WAVELET_BIND}")

    updated = "\n".join(lines) + "\n"
    if updated == original:
        return False
    write_atomic(path, updated)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tt-metal", type=Path, required=True)
    args = parser.parse_args()
    root = args.tt_metal.resolve()
    cmake = root / "ttnn/cpp/ttnn/operations/CMakeLists.txt"
    nanobind = root / "ttnn/cpp/ttnn-nanobind/__init__.cpp"
    for path in (cmake, nanobind):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required TT-Metal integration file is missing: {path}"
            )

    changed = integrate_cmake(cmake)
    changed = integrate_nanobind(nanobind) or changed
    print(
        "TTNN-Wavelet registration hooks updated"
        if changed
        else "TTNN-Wavelet registration hooks already valid"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
