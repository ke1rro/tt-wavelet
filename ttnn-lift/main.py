import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tt-metal", "ttnn"))

import pywt
import torch
import ttnn
from tt_wavelet.lifting import BoundaryModes, LiftingWaveletTransform, load_lifting_scheme
from tt_wavelet.signal import InputSpec, Signal, SignalType

sig = Signal(InputSpec(SignalType.NORMAL, 32, 1.0, 1))
device = ttnn.open_device(device_id=0)
x = torch.from_numpy(sig.data).to(torch.float32).unsqueeze(0)
try:
    ttnn_out = LiftingWaveletTransform(
        load_lifting_scheme("schemes/db1.json"), device, BoundaryModes.SYMMETRIC
    ).forward(x)
    pywt_cA, pywt_cD = pywt.dwt(sig.data, "db1", mode="symmetric")
    print(
        "ttnn",
        {
            "cA": ttnn.to_torch(ttnn_out["cA"]).tolist(),
            "cD": ttnn.to_torch(ttnn_out["cD"]).tolist(),
        },
    )
    print("pywt", {"cA": pywt_cA.tolist(), "cD": pywt_cD.tolist()})
finally:
    ttnn.close_device(device)

with open("output.txt", "w") as f:
    f.write(
        f"ttnn: {ttnn.to_torch(ttnn_out['cA']).tolist()}, {ttnn.to_torch(ttnn_out['cD']).tolist()}\n"
    )
    f.write(f"pywt: {pywt_cA.tolist()}, {pywt_cD.tolist()}\n")
