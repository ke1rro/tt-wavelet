import torch
import ttnn
from tt_wavelet.lifting import BoundaryModes, LiftingWaveletTransform, load_lifting_scheme
from tt_wavelet.signal import InputSpec, Signal, SignalType

device = ttnn.open_device(device_id=0)
x = (
    torch.from_numpy(Signal(InputSpec(SignalType.NORMAL, 32, 1.0, 1)).data)
    .to(torch.float32)
    .unsqueeze(0)
)
try:
    print(
        LiftingWaveletTransform(
            load_lifting_scheme("schemes/db1.json"), device, BoundaryModes.SYMMETRIC
        ).forward(x)
    )
finally:
    ttnn.close_device(device)
