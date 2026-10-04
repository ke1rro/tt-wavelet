# SPDX-FileCopyrightText: © 2026 Nikita Lenyk
#
# SPDX-License-Identifier: Apache-2.0

"""TT-Wavelet operations on existing TTNN tensors."""

from pathlib import Path as _Path

import ttnn as _ttnn
from . import _ttwt as _native

_native._set_resource_root(str(_Path(__file__).resolve().parent / "resources"))

from ._ttwt import dwt_coeff_len, dwt, idwt, dwt_2d, idwt_2d

__all__ = ["dwt_coeff_len", "dwt", "idwt", "dwt_2d", "idwt_2d"]
