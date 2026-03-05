import argparse
import tomllib
from itertools import product
from os import PathLike
from pathlib import Path

import numpy as np
import pywt
import dtypes
from tqdm import tqdm


def save(path: Path, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        data.dump(f)


def pad(data: np.ndarray, left: int, right: int, mode: str) -> np.ndarray:
    pad_shape = ((left, right),) * data.ndim
    if mode == "symmetric":
        return np.pad(data, pad_shape, mode="symmetric")
    elif mode == "periodic":
        return np.pad(data, pad_shape, mode="wrap")
    elif mode == "zero":
        return np.pad(data, pad_shape, mode="constant", constant_values=0)
    elif mode == "constant":
        return np.pad(data, pad_shape, mode="edge")
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")

def dwt(
    data: np.ndarray, wavelet: pywt.Wavelet, mode: str, dtype: dtypes.dtype
) -> tuple[np.ndarray, np.ndarray]:
    L = wavelet.dec_len

    data = pad(data, L - 1, L - 1, mode=mode)

    approx = dtype.conv(data, wavelet.dec_lo)
    detail = dtype.conv(data, wavelet.dec_hi)

    approx = approx[1::2]
    detail = detail[1::2]

    return approx, detail


def idwt(
    approx: np.ndarray,
    detail: np.ndarray,
    wavelet: pywt.Wavelet,
    mode: str,
    dtype: dtypes.dtype,
) -> np.ndarray:
    if approx.shape != detail.shape:
        raise ValueError(
            f"Shape mismatch for approx and detail coefficients: {approx.shape} vs {detail.shape}"
        )

    approx_up = np.zeros(len(approx) * 2 + 1)
    approx_up[1::2] = approx

    detail_up = np.zeros(len(detail) * 2 + 1)
    detail_up[1::2] = detail

    rec_approx = dtype.conv(approx_up, wavelet.rec_lo)
    rec_detail = dtype.conv(detail_up, wavelet.rec_hi)

    return dtype.add(rec_approx, rec_detail)


def main(
    config_path: str = "config.toml",
    input_path: PathLike = "data/input",
    output_path: PathLike = "data/output-pywt",
    dtype: dtypes.dtype = dtypes.float64,
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

        wavelet = pywt.Wavelet(pywvt_name)

        if len(shape) == 1:
            l, h = dwt(data, wavelet, mode, dtype)

            if l.shape != h.shape:
                raise ValueError(
                    f"Unexpected shapes for approx and detail coefficients: {l.shape} vs {h.shape}"
                )

            save(output_path / input_name / (wavelet_name + "_l_fwd"), l)
            save(output_path / input_name / (wavelet_name + "_h_fwd"), h)
            inv = idwt(l, h, wavelet, mode, dtype)
            save(output_path / input_name / (wavelet_name + "_inv"), inv)
        elif len(shape) == 2:
            continue
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
        default="data/output",
        help="Directory to save wavelet coefficients.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=dtypes.map.keys(),
        default=dtypes.map["float64"],
        help="Data type for computations.",
    )
    args = parser.parse_args()

    dtype = dtypes.map[args.dtype]
    main(
        config_path=args.config,
        input_path=args.input,
        output_path=args.output,
        dtype=dtype,
    )
