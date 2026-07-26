#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import unittest

from util.check_tiling_contract import ContractError, validate_contract


def valid_contract() -> dict:
    return {
        "preprocessing": ["zero_pad", "split2d"],
        "input": {
            "logical_shape": [33, 65],
            "storage_shape": [64, 96],
            "layout": "tile",
        },
        "kernel_shapes": [
            {"name": "input", "shape": [64, 96]},
            {"name": "workspace", "shape": [32, 64]},
        ],
        "intermediates": [
            {"name": name, "shape": [32, 64], "layout": "tile"}
            for name in ("P0", "P1", "P2", "P3", "Scratch")
        ],
        "outputs": [
            {"name": name, "shape": [32, 64], "layout": "tile"}
            for name in ("LL", "LH", "HL", "HH")
        ],
    }


class CheckTilingContractTest(unittest.TestCase):
    def test_valid_contract(self) -> None:
        validate_contract(valid_contract())

    def test_input_requires_minimal_padding(self) -> None:
        contract = valid_contract()
        contract["input"]["storage_shape"] = [96, 96]
        with self.assertRaisesRegex(ContractError, "minimal tile expansion"):
            validate_contract(contract)

    def test_padding_precedes_split(self) -> None:
        contract = valid_contract()
        contract["preprocessing"] = ["split2d", "zero_pad"]
        with self.assertRaisesRegex(ContractError, "zero_pad must precede split2d"):
            validate_contract(contract)

    def test_kernel_shape_is_tile_aligned(self) -> None:
        contract = valid_contract()
        contract["kernel_shapes"][1]["shape"] = [31, 64]
        with self.assertRaisesRegex(ContractError, "violates the 32x32"):
            validate_contract(contract)

    def test_intermediate_uses_tile_layout(self) -> None:
        contract = valid_contract()
        contract["intermediates"][0]["layout"] = "row-major"
        with self.assertRaisesRegex(ContractError, "must use tile layout"):
            validate_contract(contract)

    def test_four_output_bands_are_required(self) -> None:
        contract = valid_contract()
        contract["outputs"].pop()
        with self.assertRaisesRegex(ContractError, "must contain LL, LH, HL, and HH"):
            validate_contract(contract)


if __name__ == "__main__":
    unittest.main()
