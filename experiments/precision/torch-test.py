import argparse
import tomllib
from itertools import product
from os import PathLike
from pathlib import Path

import numpy as np
import pywt
import torch
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
    data: np.ndarray, wavelet: pywt.Wavelet, mode: str, dtype: torch.dtype
) -> tuple[np.ndarray, np.ndarray]:
    L = wavelet.dec_len

    ldec = torch.asarray(wavelet.dec_lo[::-1]).to(dtype)
    hdec = torch.asarray(wavelet.dec_hi[::-1]).to(dtype)

    data = pad(data, L - 1, L - 1, mode=mode)
    data = torch.from_numpy(data).to(dtype)

    approx = torch.nn.functional.conv1d(
        data.view(1, 1, -1), ldec.view(1, 1, -1), padding="valid"
    ).view(-1)
    detail = torch.nn.functional.conv1d(
        data.view(1, 1, -1), hdec.view(1, 1, -1), padding="valid"
    ).view(-1)

    approx = approx[1::2]
    detail = detail[1::2]

    return (
        approx.to(torch.float64).numpy(),
        detail.to(torch.float64).numpy(),
    )


def idwt(
    approx: np.ndarray,
    detail: np.ndarray,
    wavelet: pywt.Wavelet,
    mode: str,
    dtype: torch.dtype,
) -> np.ndarray:
    if approx.shape != detail.shape:
        raise ValueError(
            f"Shape mismatch for approx and detail coefficients: {approx.shape} vs {detail.shape}"
        )

    lrec = torch.asarray(wavelet.rec_lo[::-1]).to(dtype)
    hrec = torch.asarray(wavelet.rec_hi[::-1]).to(dtype)

    approx_up = np.zeros(len(approx) * 2 + 1)
    approx_up[1::2] = approx

    detail_up = np.zeros(len(detail) * 2 + 1)
    detail_up[1::2] = detail

    approx_up = torch.from_numpy(approx_up).to(dtype)
    detail_up = torch.from_numpy(detail_up).to(dtype)

    rec_approx = torch.nn.functional.conv1d(
        approx_up.view(1, 1, -1), lrec.view(1, 1, -1), padding="valid"
    ).view(-1)

    rec_detail = torch.nn.functional.conv1d(
        detail_up.view(1, 1, -1), hrec.view(1, 1, -1), padding="valid"
    ).view(-1)

    return (rec_approx + rec_detail).to(torch.float64).numpy()


def main(
    config_path: str = "config.toml",
    input_path: PathLike = "data/input",
    output_path: PathLike = "data/output-pywt",
    dtype: torch.dtype = torch.float64,
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
        choices=["float32", "float64", "bfloat16", "float16"],
        default="float64",
        help="Data type for computations.",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    main(
        config_path=args.config,
        input_path=args.input,
        output_path=args.output,
        dtype=dtype,
    )
