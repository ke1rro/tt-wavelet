import argparse
import tomllib
from itertools import product
from os import PathLike
from pathlib import Path

import numpy as np
import pywt
from tqdm import tqdm


def save(path: Path, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data.astype("<f8").tobytes())


def main(
    config_path: str = "config.toml",
    input_path: PathLike = "data/input",
    output_path: PathLike = "data/output-pywt",
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    for wavelet, input_descriptor in tqdm(
        product(config["wavelets"], config["inputs"]),
        total=len(config["wavelets"]) * len(config["inputs"]),
        desc="Computing wavelet transforms",
    ):
        wavelet_name = wavelet["name"]
        pywvt_name = wavelet["id"]
        mode = wavelet["mode"]
        input_name = input_descriptor["name"]
        shape = input_descriptor["shape"]

        with open(input_path / input_name, "rb") as f:
            data = (
                np.frombuffer(f.read(), dtype="<f8").astype(np.float64).reshape(shape)
            )

        if len(shape) == 1:
            l, h = pywt.dwt(data, pywvt_name, mode=mode)
            save(output_path / input_name / (wavelet_name + "_l_fwd"), l)
            save(output_path / input_name / (wavelet_name + "_h_fwd"), h)
            inv = pywt.idwt(l, h, pywvt_name, mode=mode)
            save(output_path / input_name / (wavelet_name + "_inv"), inv)
        elif len(shape) == 2:
            ll, (lh, hl, hh) = pywt.dwt2(data, pywvt_name, mode=mode)
            save(output_path / input_name / (wavelet_name + "_ll_fwd"), ll)
            save(output_path / input_name / (wavelet_name + "_lh_fwd"), lh)
            save(output_path / input_name / (wavelet_name + "_hl_fwd"), hl)
            save(output_path / input_name / (wavelet_name + "_hh_fwd"), hh)
            inv = pywt.idwt2((ll, (lh, hl, hh)), pywvt_name, mode=mode)
            save(output_path / input_name / (wavelet_name + "_inv"), inv)
        else:
            raise ValueError("Unsupported dimension.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute wavelet transforms using PyWavelets for precision experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input",
        help="Directory containing input signals.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output-pywt",
        help="Directory to save wavelet coefficients.",
    )
    args = parser.parse_args()

    main(config_path=args.config, input_path=args.input, output_path=args.output)
