import argparse
import json
import re
from pathlib import Path

VALID_STEP_TYPES = {
    "swap": "LiftingStepType::kSwap",
    "predict": "LiftingStepType::kPredict",
    "update": "LiftingStepType::kUpdate",
    "scale-even": "LiftingStepType::kScaleEven",
    "scale-odd": "LiftingStepType::kScaleOdd",
}
MAX_STENCIL_WIDTH = 17


def cpp_identifier(scheme_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", scheme_name)


def cpp_float(value: float) -> str:
    rendered = format(float(value), ".17g")
    if "." not in rendered and "e" not in rendered and "E" not in rendered:
        rendered += ".0"
    return rendered + "f"


def validate_scheme(scheme_name: str, scheme_data: dict) -> None:
    if not isinstance(scheme_data.get("tap_size"), int) or scheme_data["tap_size"] <= 0:
        raise ValueError(f"{scheme_name}: tap_size must be positive")
    if not scheme_data.get("steps"):
        raise ValueError(f"{scheme_name}: steps must not be empty")
    for index, step in enumerate(scheme_data["steps"]):
        kind = step.get("type")
        if kind not in VALID_STEP_TYPES:
            raise ValueError(f"{scheme_name}: unsupported step type at {index}: {kind}")
        coefficients = step.get("coefficients", [])
        if kind in {"predict", "update"} and not (1 <= len(coefficients) <= MAX_STENCIL_WIDTH):
            raise ValueError(f"{scheme_name}: predict/update width at {index} is unsupported")
        if kind in {"scale-even", "scale-odd"} and len(coefficients) != 1:
            raise ValueError(f"{scheme_name}: scale step {index} must have one coefficient")
        if kind == "swap" and coefficients:
            raise ValueError(f"{scheme_name}: swap step {index} must have no coefficients")
        shift = step.get("shift", 0)
        if not isinstance(shift, int) or shift < -16 or shift > 16:
            raise ValueError(f"{scheme_name}: step {index} shift exceeds device range")


def generate_header_content(scheme_name: str, scheme_data: dict, base_header: Path) -> str:
    validate_scheme(scheme_name, scheme_data)
    variable = cpp_identifier(scheme_name)
    code = f"""
#pragma once

#include "{str(base_header)}"

namespace ttwv::schemes {{

inline const LiftingScheme {variable}{{
    {scheme_data['delay']['even']}, {scheme_data['delay']['odd']}, {scheme_data['tap_size']}, {{
""".strip()

    for step in scheme_data["steps"]:
        type_ = VALID_STEP_TYPES[step["type"]]
        coefficients = "{" + ", ".join(cpp_float(coef) for coef in step["coefficients"]) + "}"
        code += f"\n        LiftingStep({type_}, {coefficients}, {step['shift']}),"

    code += "\n    }\n};\n\n"
    code += "}  // namespace ttwv::schemes\n"
    return code


def generate_registry_content(scheme_names: list[str]) -> str:
    includes = "\n".join(f'#include "{name}.hpp"' for name in scheme_names)
    entries = "\n".join(
        f'    SchemeEntry{{"{name}", &schemes::{cpp_identifier(name)}}},' for name in scheme_names
    )
    return f"""#pragma once

#include <array>
#include <string_view>

#include "scheme.hpp"
{includes}

namespace ttwv {{

struct SchemeEntry {{
    std::string_view name;
    const LiftingScheme* scheme;
}};

inline constexpr std::array<SchemeEntry, {len(scheme_names)}> kSchemeRegistry{{{{
{entries}
}}}};

[[nodiscard]] inline const LiftingScheme* find_scheme(const std::string_view name) noexcept {{
    for (const auto& entry : kSchemeRegistry) {{
        if (entry.name == name) {{
            return entry.scheme;
        }}
    }}
    return nullptr;
}}

}}  // namespace ttwv
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate compile-time lifting scheme descriptors."
    )
    parser.add_argument(
        "--schemes_dir",
        type=Path,
        default=Path("ttnn-wavelet/lifting_schemes/"),
        help="The directory where the input scheme files are located.",
    )
    parser.add_argument(
        "--headers_dir",
        type=Path,
        default=Path("tt-wavelet/tt_wavelet/include/schemes/"),
        help="The directory where the generated header files will be saved.",
    )
    parser.add_argument(
        "--base-header",
        type=Path,
        default=Path("schemes/scheme.hpp"),
        help="The base header file to include in the generated headers.",
    )
    parser.add_argument(
        "--registry-header",
        type=Path,
        default=Path("registry.hpp"),
        help="Registry header path, relative to headers_dir.",
    )
    args = parser.parse_args()

    args.headers_dir.mkdir(parents=True, exist_ok=True)

    scheme_names = []
    for scheme_file in sorted(args.schemes_dir.glob("*.json")):
        with open(scheme_file, "r") as f:
            scheme_data = json.load(f)

        header_content = generate_header_content(scheme_file.stem, scheme_data, args.base_header)
        header_file = args.headers_dir / f"{scheme_file.stem}.hpp"
        with open(header_file, "w") as f:
            f.write(header_content)
        scheme_names.append(scheme_file.stem)
        print(f"Generated {header_file}")

    registry_file = args.headers_dir / args.registry_header
    with open(registry_file, "w") as f:
        f.write(generate_registry_content(scheme_names))
    print(f"Generated {registry_file}")


if __name__ == "__main__":
    main()
