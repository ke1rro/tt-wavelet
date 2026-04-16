#!/usr/bin/env python3
"""
Usage
-----
  # re-generate everything from lifting_schemes/:
  python scripts/add_lifting_scheme.py

  # add / update a single scheme from an explicit JSON file:
  python scripts/add_lifting_scheme.py path/to/wavelet.json

  # same, but override the canonical name (default = JSON stem):
  python scripts/add_lifting_scheme.py path/to/wavelet.json --name wavelet

  # help:
  python scripts/add_lifting_scheme.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
JSON_DIR = REPO_ROOT / "ttnn-wavelet" / "lifting_schemes"
INCLUDE_ROOT = REPO_ROOT / "tt-wavelet" / "tt_wavelet" / "include"
WAVELET_DIR = INCLUDE_ROOT / "lifting" / "wavelets"
REGISTRY_HPP = INCLUDE_ROOT / "lifting" / "wavelets" / "wavelet_registry.hpp"
KERNEL_CHUNK_SIZE = 4

# --------------------------------------------------------------------------------------


def ident(name: str) -> str:
    return name.replace(".", "_").replace("-", "_")


def hpp_filename(name: str) -> str:
    return f"{ident(name)}.hpp"


def float_lit(v: float) -> str:
    s = f"{v:.17g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s + "f"


# parse ---------------------------------------------------------------------------


class StepSpec:
    __slots__ = ("kind", "shift", "coefficients")

    def __init__(self, kind: str, shift: int, coefficients: list[float]):
        self.kind = kind
        self.shift = shift
        self.coefficients = coefficients

    @classmethod
    def from_dict(cls, d: dict) -> "StepSpec":
        raw_coeffs = d.get("coefficients", [])
        coeffs: list[float] = []
        for c in raw_coeffs:
            if isinstance(c, (int, float)):
                coeffs.append(float(c))
            elif isinstance(c, dict):
                coeffs.append(float(c["numerator"]) / float(c["denominator"]))
            else:
                raise TypeError(f"Unknown coefficient format: {c!r}")
        return cls(
            kind=d["type"],
            shift=int(d["shift"]),
            coefficients=coeffs,
        )


class SchemeSpec:
    __slots__ = ("name", "tap_size", "delay_even", "delay_odd", "steps")

    def __init__(
        self,
        name: str,
        tap_size: int,
        delay_even: int,
        delay_odd: int,
        steps: list[StepSpec],
    ):
        self.name = name
        self.tap_size = tap_size
        self.delay_even = delay_even
        self.delay_odd = delay_odd
        self.steps = steps

    @classmethod
    def from_file(cls, path: Path, name: str | None = None) -> "SchemeSpec":
        with path.open(encoding="utf-8") as fh:
            obj = json.load(fh)
        canon_name = name or path.stem
        return cls(
            name=canon_name,
            tap_size=int(obj["tap_size"]),
            delay_even=int(obj["delay"]["even"]),
            delay_odd=int(obj["delay"]["odd"]),
            steps=[StepSpec.from_dict(s) for s in obj["steps"]],
        )


# code generation ------------------------------------------------------

_STEP_ALIAS = {
    "predict": "PredictStep",
    "update": "UpdateStep",
    "scale-even": "ScaleEvenStep",
    "scale-odd": "ScaleOddStep",
    "swap": "SwapStep",
}


def _render_step(step: StepSpec) -> str:
    alias = _STEP_ALIAS[step.kind]
    n = len(step.coefficients)

    if step.kind == "swap":
        return f"SwapStep{{ {{}}, {step.shift} }}"

    coeffs_str = ", ".join(float_lit(c) for c in step.coefficients)
    # Inner braces for std::array; outer single braces for struct aggregate
    return f"{alias}<{n}>{{ {{ {coeffs_str} }}, {step.shift} }}"


def generate_wavelet_header(spec: SchemeSpec, wavelet_id: int) -> str:
    tag_id = ident(spec.name)
    tag_type = f"{tag_id}_tag"
    scheme_id = f"{tag_id}_scheme"

    step_lines: list[str] = []
    for step in spec.steps:
        step_lines.append("    " + _render_step(step) + ",")
    if step_lines:
        step_lines[-1] = step_lines[-1].rstrip(",")
    steps_block = "\n".join(step_lines)

    return (
        f"#pragma once\n"
        f"\n"
        f'#include "../scheme.hpp"\n'
        f"\n"
        f"namespace ttwv {{\n"
        f"\n"
        f"struct {tag_type} {{}};\n"
        f"\n"
        f"inline constexpr auto {scheme_id} = make_lifting_scheme(\n"
        f"    {spec.tap_size},\n"
        f"    {spec.delay_even},\n"
        f"    {spec.delay_odd},\n"
        f"{steps_block}\n"
        f");\n"
        f"\n"
        f"template <>\n"
        f"struct scheme_traits<{tag_type}> {{\n"
        f"    using SchemeType = decltype({scheme_id});\n"
        f'    static constexpr const char* name = "{spec.name}";\n'
        f"    static constexpr int id         = {wavelet_id};\n"
        f"    static constexpr int tap_size   = {spec.tap_size};\n"
        f"    static constexpr int delay_even = {spec.delay_even};\n"
        f"    static constexpr int delay_odd  = {spec.delay_odd};\n"
        f"    static constexpr int num_steps  = {len(spec.steps)};\n"
        f"    static constexpr const auto& scheme = {scheme_id};\n"
        f"}};\n"
        f"\n"
        f"}}\n"
    )


def generate_registry(specs: list[SchemeSpec], sorted_specs: list[SchemeSpec]) -> str:
    count = len(sorted_specs)

    includes = "\n".join(f'#include "{hpp_filename(s.name)}"' for s in sorted_specs)

    wavelet_scheme_specs: list[str] = []
    for wavelet_id, s in enumerate(sorted_specs):
        tag_id = ident(s.name)
        tag_type = f"{tag_id}_tag"
        scheme_id = f"{tag_id}_scheme"
        wavelet_scheme_specs.append(
            f"template<> struct WaveletScheme<{wavelet_id}> {{\n"
            f"    using Tag = {tag_type};\n"
            f"    static constexpr const auto& scheme = {scheme_id};\n"
            f"}};"
        )
    wavelet_scheme_block = "\n\n".join(wavelet_scheme_specs)

    entries: list[str] = []
    for wavelet_id, s in enumerate(sorted_specs):
        entries.append(
            f'    {{"{s.name}", {wavelet_id}, {s.tap_size}, {s.delay_even}, {s.delay_odd}, {len(s.steps)}}},'
        )
    entries_block = "\n".join(entries)

    lines: list[str] = [
        "#pragma once",
        "",
        includes,
        "",
        "#ifndef TTWV_WAVELET_REGISTRY_KERNEL_ONLY",
        "#include <string_view>",
        "#endif",
        "",
        "namespace ttwv {",
        "",
        f"inline constexpr int kWaveletSchemeCount = {count};",
        "",
        wavelet_scheme_block,
        "",
        "#ifndef TTWV_WAVELET_REGISTRY_KERNEL_ONLY",
        "[[nodiscard]] inline const SchemeInfo* scheme_table() noexcept {",
        f"    static constexpr SchemeInfo table[{count}] = {{",
        entries_block,
        "    };",
        "    return table;",
        "}",
        "",
        "[[nodiscard]] inline const SchemeInfo* find_scheme(std::string_view name) noexcept {",
        "    const auto* table = scheme_table();",
        "    for (int i = 0; i < kWaveletSchemeCount; ++i) {",
        "        const auto& info = table[i];",
        "        if (name == info.name) {",
        "            return &table[i];",
        "        }",
        "    }",
        "    return nullptr;",
        "}",
        "#endif",
        "",
        "}",
        "",
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------
def write_if_changed(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


# cli -----------------------------------------------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "json_file",
        nargs="?",
        type=Path,
        help=(
            "Path to a single JSON lifting-scheme file to add / update. "
            "If omitted, all JSON files in the lifting_schemes/ directory are processed."
        ),
    )
    p.add_argument(
        "--name",
        default=None,
        help=(
            "Override the canonical wavelet name (default: JSON file stem). "
            "Only valid when a single json_file is provided."
        ),
    )
    p.add_argument(
        "--json-dir",
        type=Path,
        default=JSON_DIR,
        help=f"Directory containing JSON lifting-scheme files (default: {JSON_DIR})",
    )
    p.add_argument(
        "--include-root",
        type=Path,
        default=INCLUDE_ROOT,
        help=f"Root include directory for generated headers (default: {INCLUDE_ROOT})",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    include_root: Path = args.include_root
    wavelet_dir = include_root / "lifting" / "wavelets"
    registry_hpp = include_root / "lifting" / "wavelets" / "wavelet_registry.hpp"
    json_dir: Path = args.json_dir

    if args.json_file is not None:
        if args.json_file.suffix.lower() != ".json":
            print(f"ERROR: expected a .json file, got: {args.json_file}", file=sys.stderr)
            return 1
        if not args.json_file.exists():
            print(f"ERROR: file not found: {args.json_file}", file=sys.stderr)
            return 1

        new_spec = SchemeSpec.from_file(args.json_file, name=args.name)
        print(f"Processing: {args.json_file.name}  →  '{new_spec.name}'")

        all_specs: list[SchemeSpec] = []
        if json_dir.is_dir():
            for path in sorted(json_dir.glob("*.json")):
                if path == args.json_file.resolve():
                    continue
                try:
                    all_specs.append(SchemeSpec.from_file(path))
                except Exception as exc:
                    print(f"  WARN: skipping {path.name}: {exc}", file=sys.stderr)
        all_specs.append(new_spec)

    else:
        if not json_dir.is_dir():
            print(f"ERROR: JSON directory not found: {json_dir}", file=sys.stderr)
            return 1

        json_files = sorted(json_dir.glob("*.json"))
        if not json_files:
            print(f"ERROR: no *.json files in {json_dir}", file=sys.stderr)
            return 1

        print(f"Found {len(json_files)} JSON files in {json_dir.relative_to(REPO_ROOT)}")
        all_specs = []
        errors = 0
        for path in json_files:
            try:
                all_specs.append(SchemeSpec.from_file(path))
            except Exception as exc:
                print(f"  ERROR: {path.name}: {exc}", file=sys.stderr)
                errors += 1
        if errors:
            return 1

    sorted_specs = sorted(all_specs, key=lambda s: s.name)

    written = 0
    skipped = 0
    for wavelet_id, spec in enumerate(sorted_specs):
        out_path = wavelet_dir / hpp_filename(spec.name)
        content = generate_wavelet_header(spec, wavelet_id)
        if write_if_changed(out_path, content):
            print(f"  WRITE  {out_path.relative_to(REPO_ROOT)}  (id={wavelet_id})")
            written += 1
        else:
            skipped += 1

    if skipped:
        print(f"  (skipped {skipped} already up-to-date files)")

    registry_content = generate_registry(all_specs, sorted_specs)
    if write_if_changed(registry_hpp, registry_content):
        print(f"  WRITE  {registry_hpp.relative_to(REPO_ROOT)}")
    else:
        print(f"  (registry already up-to-date)")

    print(f"\nDone. {written} header(s) written, {len(sorted_specs)} scheme(s) registered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())