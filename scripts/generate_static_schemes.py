#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_JSON_DIR = REPO_ROOT / "ttnn-wavelet" / "lifting_schemes"
DEFAULT_SCHEME_DIR = REPO_ROOT / "tt-wavelet" / "tt_wavelet" / "include" / "schemes" / "generated"
DEFAULT_REGISTRY = DEFAULT_SCHEME_DIR / "registry.hpp"
DEFAULT_COMPUTE_DIR = REPO_ROOT / "tt-wavelet" / "kernels" / "compute" / "generated"

STEP_TYPES = {
    "predict": "StepType::kPredict",
    "update": "StepType::kUpdate",
    "scale-even": "StepType::kScaleEven",
    "scale-odd": "StepType::kScaleOdd",
    "swap": "StepType::kSwap",
}


@dataclass(frozen=True)
class Step:
    kind: str
    shift: int
    coeff_bits: tuple[int, ...]


@dataclass(frozen=True)
class Scheme:
    name: str
    ident: str
    tap_size: int
    delay_even: int
    delay_odd: int
    steps: tuple[Step, ...]


def make_ident(name: str) -> str:
    ident = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"w_{ident}"
    return ident


def parse_coeff(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, dict):
        return float(raw["numerator"]) / float(raw["denominator"])
    raise TypeError(f"Unsupported coefficient encoding: {raw!r}")


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def load_scheme(path: Path) -> Scheme:
    obj = json.loads(path.read_text(encoding="utf-8"))
    name = path.stem
    steps: list[Step] = []
    for raw_step in obj["steps"]:
        kind = raw_step["type"]
        if kind not in STEP_TYPES:
            raise ValueError(f"{path.name}: unsupported step type {kind!r}")
        coeff_bits = tuple(
            f32_bits(parse_coeff(coeff)) for coeff in raw_step.get("coefficients", [])
        )
        if kind in {"scale-even", "scale-odd"} and len(coeff_bits) != 1:
            raise ValueError(f"{path.name}: {kind} must have exactly one coefficient")
        if kind == "swap" and coeff_bits:
            raise ValueError(f"{path.name}: swap must not have coefficients")
        steps.append(Step(kind=kind, shift=int(raw_step["shift"]), coeff_bits=coeff_bits))

    return Scheme(
        name=name,
        ident=make_ident(name),
        tap_size=int(obj["tap_size"]),
        delay_even=int(obj["delay"]["even"]),
        delay_odd=int(obj["delay"]["odd"]),
        steps=tuple(steps),
    )


def coeff_args(step: Step) -> str:
    if not step.coeff_bits:
        return ""
    return ", " + ", ".join(f"0x{bits:08x}U" for bits in step.coeff_bits)


def render_scheme_header(scheme: Scheme) -> str:
    lines: list[str] = [
        "#pragma once",
        "",
        '#include "../../lifting/static_scheme.hpp"',
        "",
        "namespace ttwv::schemes {",
        "",
        f"struct {scheme.ident} {{",
        f'    static constexpr const char* name = "{scheme.name}";',
        f"    static constexpr uint32_t tap_size = {scheme.tap_size}U;",
        f"    static constexpr int32_t delay_even = {scheme.delay_even};",
        f"    static constexpr int32_t delay_odd = {scheme.delay_odd};",
        f"    static constexpr uint32_t num_steps = {len(scheme.steps)}U;",
        (
            f"    static constexpr const char* compute_kernel_path = "
            f'"kernels/compute/generated/lwt_fused_{scheme.ident}_compute.cpp";'
        ),
        "",
        "    template <std::size_t I>",
        "    struct step;",
        "};",
        "",
    ]

    for index, step in enumerate(scheme.steps):
        lines.extend(
            [
                "template <>",
                f"struct {scheme.ident}::step<{index}> {{",
                (
                    f"    using type = StaticStep<{STEP_TYPES[step.kind]}, {step.shift}"
                    f"{coeff_args(step)}>;"
                ),
                f"    static_assert(type::k == {len(step.coeff_bits)}U);",
                "};",
                "",
            ]
        )

    lines.extend(["}  // namespace ttwv::schemes", ""])
    return "\n".join(lines)


def render_registry(schemes: list[Scheme]) -> str:
    first_ident = schemes[0].ident
    includes = "\n".join(f'#include "{scheme.ident}.hpp"' for scheme in schemes)
    enum_entries = "\n".join(f"    k{scheme.ident}," for scheme in schemes)
    info_entries = "\n".join(
        (
            f'    SchemeInfo{{"{scheme.name}", {scheme.tap_size}U, '
            f"{scheme.delay_even}, {scheme.delay_odd}, {len(scheme.steps)}U}},"
        )
        for scheme in schemes
    )

    scheme_id_checks = "\n".join(
        f'    if (name == "{scheme.name}") return SchemeId::k{scheme.ident};' for scheme in schemes
    )
    dispatch_cases = "\n".join(
        (
            f"        case SchemeId::k{scheme.ident}: "
            f"return fn.template operator()<schemes::{scheme.ident}>();"
        )
        for scheme in schemes
    )

    return "\n".join(
        [
            "#pragma once",
            "",
            "#include <array>",
            "#include <cstdint>",
            "#include <span>",
            "#include <string>",
            "#include <string_view>",
            "",
            includes,
            "",
            "#include <tt_stl/assert.hpp>",
            "",
            "namespace ttwv {",
            "",
            "struct SchemeInfo {",
            "    std::string_view name;",
            "    uint32_t tap_size;",
            "    int32_t delay_even;",
            "    int32_t delay_odd;",
            "    uint32_t num_steps;",
            "};",
            "",
            "enum class SchemeId : uint32_t {",
            enum_entries,
            "    kUnknown,",
            "};",
            "",
            f"inline constexpr std::array<SchemeInfo, {len(schemes)}> kSchemeInfos = {{",
            info_entries,
            "};",
            "",
            "[[nodiscard]] inline std::span<const SchemeInfo> available_wavelets() noexcept {",
            "    return kSchemeInfos;",
            "}",
            "",
            "[[nodiscard]] inline SchemeId scheme_id(std::string_view name) noexcept {",
            scheme_id_checks,
            "    return SchemeId::kUnknown;",
            "}",
            "",
            "template <typename Fn>",
            "decltype(auto) dispatch_scheme(std::string_view name, Fn&& fn) {",
            "    switch (scheme_id(name)) {",
            dispatch_cases,
            "        case SchemeId::kUnknown: break;",
            "    }",
            '    TT_THROW("Unsupported wavelet scheme: {}", std::string{name});',
            f"    return fn.template operator()<schemes::{first_ident}>();",
            "}",
            "",
            "}  // namespace ttwv",
            "",
        ]
    )


def render_compute_wrapper(scheme: Scheme) -> str:
    return "\n".join(
        [
            '#include "../lwt_fused_compute_template.hpp"',
            f'#include "../../../tt_wavelet/include/schemes/generated/{scheme.ident}.hpp"',
            "",
            "void kernel_main() {",
            f"    ttwv::kernels::lwt_fused_compute<ttwv::schemes::{scheme.ident}>();",
            "}",
            "",
        ]
    )


def write_if_changed(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def remove_stale(directory: Path, keep: set[Path], pattern: str) -> None:
    if not directory.exists():
        return
    for path in directory.glob(pattern):
        if path not in keep:
            path.unlink()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate static TT-wavelet lifting schemes from JSON."
    )
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--scheme-dir", type=Path, default=DEFAULT_SCHEME_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--compute-dir", type=Path, default=DEFAULT_COMPUTE_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    json_files = sorted(args.json_dir.glob("*.json"))
    if not json_files:
        print(f"ERROR: no JSON schemes found in {args.json_dir}", file=sys.stderr)
        return 1

    schemes = sorted((load_scheme(path) for path in json_files), key=lambda scheme: scheme.name)
    if len({scheme.ident for scheme in schemes}) != len(schemes):
        raise RuntimeError("Generated scheme identifiers are not unique")

    kept_scheme_headers: set[Path] = set()
    kept_compute_wrappers: set[Path] = set()
    writes = 0

    for scheme in schemes:
        scheme_path = args.scheme_dir / f"{scheme.ident}.hpp"
        compute_path = args.compute_dir / f"lwt_fused_{scheme.ident}_compute.cpp"
        kept_scheme_headers.add(scheme_path)
        kept_compute_wrappers.add(compute_path)
        writes += write_if_changed(scheme_path, render_scheme_header(scheme))
        writes += write_if_changed(compute_path, render_compute_wrapper(scheme))

    writes += write_if_changed(args.registry, render_registry(schemes))
    remove_stale(args.scheme_dir, kept_scheme_headers | {args.registry}, "*.hpp")
    remove_stale(args.compute_dir, kept_compute_wrappers, "lwt_fused_*_compute.cpp")

    print(f"Generated {len(schemes)} static schemes ({writes} files changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
