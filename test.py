import os
import sys

import numpy as np
from pywt import Wavelet, dwt

import dtypes
from lifting2 import LiftingScheme

sys.set_int_max_str_digits(1000000)


def print_cmp(err_mse, name):
    message = " " + name.upper() + ": "
    message = message + " " * (17 - len(message))

    if err_mse < 1e-8:
        message += f"\033[32m{err_mse:.2e}\033[0m"
    elif err_mse < 1e-7:
        message += f"\033[33m{err_mse:.2e}\033[0m"
    else:
        message += f"\033[31m{err_mse:.2e}\033[0m"

    print(message)


recon_hi = []
recon_lo = []
recon_no = []
fwd_hi = []
fwd_lo = []
fwd_no = []


counter = 0
for filename in os.listdir("../coeffs/"):
    wavelet_name = filename.removesuffix(".json")
    print(f"WAVELET: {wavelet_name}")

    wavelet = Wavelet(wavelet_name)

    scheme = LiftingScheme.from_file(f"../coeffs/{filename}", "symmetric", dtype=dtypes.float64)

    test_signal = np.random.normal(scale=1, size=(1001,))

    approx_lifting, details_lifting = scheme.forward(test_signal)
    approx_dwt, details_dwt = dwt(test_signal, wavelet=wavelet, mode="symmetric")

    approx_err = np.abs(approx_lifting - approx_dwt)
    approx_err_mse = np.sum(approx_err**2) / len(approx_err)

    details_err = np.abs(details_lifting - details_dwt)
    details_err_mse = np.sum(details_err**2) / len(details_err)

    if details_err_mse < 1e-16 and details_err_mse < 1e-16:
        fwd_hi.append(wavelet_name)
    elif details_err_mse < 1e-6 and details_err_mse < 1e-6:
        fwd_lo.append(wavelet_name)
    else:
        fwd_no.append(wavelet_name)

    print_cmp(approx_err_mse, "approximation")
    print_cmp(details_err_mse, "details")

    data_lifting = scheme.inverse(approx_lifting, details_lifting)
    data_lifting = data_lifting[: len(test_signal)]

    recon_err = np.abs(data_lifting - test_signal)
    recon_err_mse = np.sum(recon_err**2) / len(recon_err)

    if recon_err_mse < 1e-16:
        recon_hi.append(wavelet_name)
    elif recon_err_mse < 1e-6:
        recon_lo.append(wavelet_name)
    else:
        recon_no.append(wavelet_name)

    print_cmp(recon_err_mse, "reconstruction")
    counter += 1

print(f"Tested {counter} wavelets")

print()
print(" Reconstruction High:", sorted(recon_hi))
print(" Reconstruction Low:", sorted(recon_lo))
print(" Reconstruction None:", sorted(recon_no))
print()

print(" Forward High:", sorted(fwd_hi))
print(" Forward Low:", sorted(fwd_lo))
print(" Forward None:", sorted(fwd_no))
