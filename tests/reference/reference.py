"""
pywavelet reference implementation
"""

from typing import List, Tuple

import numpy as np
import pywt


def generate_reference(wavelet: str, shape: List[int], dim: str) -> np.ndarray:
    """
    Args:
        wavelet: Wavelet name (e.g., 'haar', 'db4', 'bior2.2')
        shape: Input shape [batch, channels, height, width]
        dim: '1d' or '2d'

    Returns:
        Wavelet coefficients as numpy array
    """
    np.random.seed(42)  # <--------------
    input_data = np.random.randn(*shape).astype(np.float32)

    if dim == "1d":
        return compute_1d_dwt(input_data, wavelet)
    elif dim == "2d":
        return compute_2d_dwt(input_data, wavelet)
    else:
        raise ValueError(f"Unknown dimension: {dim}")


def compute_1d_dwt(data: np.ndarray, wavelet: str) -> np.ndarray:
    batch, channels, height, width = data.shape

    data_2d = data.reshape(-1, width)

    output_list = []
    for i in range(data_2d.shape[0]):
        cA, cD = pywt.dwt(data_2d[i], wavelet, mode="periodization")
        output_list.append(np.concatenate([cA, cD]))

    output = np.array(output_list)
    output = output.reshape(batch, channels, height, -1)

    return output


def compute_2d_dwt(data: np.ndarray, wavelet: str) -> np.ndarray:
    batch, channels, height, width = data.shape

    output_list = []
    for b in range(batch):
        for c in range(channels):
            coeffs = pywt.dwt2(data[b, c], wavelet, mode="periodization")
            cA, (cH, cV, cD) = coeffs

            # [cA | cH | cV | cD]
            top = np.concatenate([cA, cH], axis=1)
            bottom = np.concatenate([cV, cD], axis=1)
            packed = np.concatenate([top, bottom], axis=0)

            output_list.append(packed)

    output = np.array(output_list)
    output = output.reshape(batch, channels, height, width)

    return output
