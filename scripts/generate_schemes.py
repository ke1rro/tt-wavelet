
from pathlib import Path
import argparse
import json


def generate_header_content(scheme_name: str, scheme_data: dict, base_header: Path = Path("wavelet/scheme.hpp")) -> str:
    scheme_name = scheme_name.replace(".", "_")
    code = f"""
#pragma once

#include "{str(base_header)}"

namespace ttwv::schemes {{

const LiftingScheme {scheme_name}(
    {scheme_data['delay']['even']}, {scheme_data['delay']['odd']}, {scheme_data['tap_size']}
""".strip()

    for step in scheme_data["steps"]:
        type_ = {
            "swap": "kSwap",
            "predict": "kPredict",
            "update": "kUpdate",
            "scale-even": "kScaleEven",
            "scale-odd": "kScaleOdd",
        }[step["type"]]
        coefficients = "{" + ", ".join(
            f"{coef}f" for coef in step["coefficients"]
        ) + "}"
        code += f",\n    LiftingStep({type_}, {coefficients}, {step['shift']})"

    code += "\n)\n\n"
    code += "}  // namespace ttwv::schemes\n"


    return code

def main():
    parser = argparse.ArgumentParser(description="Generate color schemes for the terminal.")
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
    args = parser.parse_args()

    args.headers_dir.mkdir(parents=True, exist_ok=True)

    for scheme_file in args.schemes_dir.glob("*.json"):
        with open(scheme_file, "r") as f:
            scheme_data = json.load(f)

        header_content = generate_header_content(scheme_file.stem, scheme_data, args.base_header)
        header_file = args.headers_dir / f"{scheme_file.stem}.hpp"
        with open(header_file, "w") as f:
            f.write(header_content)
        print(f"Generated {header_file}")


if __name__ == "__main__":
    main()
