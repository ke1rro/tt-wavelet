#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate the serialized 2D LWT TTNN tiling contract.

The JSON contract is intentionally independent of Metalium implementation
types so it can validate planner dumps and hand-authored device test fixtures.
Required top-level fields:

{
  "preprocessing": ["zero_pad", "split2d"],
  "input": {
    "logical_shape": [H, W],
    "storage_shape": [H_PADDED, W_PADDED],
    "layout": "tile"
  },
  "kernel_shapes": [{"name": "...", "shape": [H, W]}],
  "intermediates": [{"name": "...", "shape": [H, W], "layout": "tile"}],
  "outputs": [{"name": "LL", "shape": [H, W], "layout": "tile"}, ...]
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TILE_HEIGHT = 32
TILE_WIDTH = 32
OUTPUT_NAMES = {"LL", "LH", "HL", "HH"}


class ContractError(ValueError):
    """Raised when a tiling-contract invariant is violated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path, help="Path to the JSON contract")
    return parser.parse_args()


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def require_sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{label} must be a JSON array")
    return value


def parse_shape(value: Any, label: str) -> tuple[int, int]:
    dimensions = require_sequence(value, label)
    if (
        len(dimensions) != 2
        or any(isinstance(dimension, bool) for dimension in dimensions)
        or any(not isinstance(dimension, int) for dimension in dimensions)
        or any(dimension <= 0 for dimension in dimensions)
    ):
        raise ContractError(f"{label} must contain two positive integer dimensions")
    return dimensions[0], dimensions[1]


def require_tile_shape(value: Any, label: str) -> tuple[int, int]:
    height, width = parse_shape(value, label)
    if height % TILE_HEIGHT != 0 or width % TILE_WIDTH != 0:
        raise ContractError(
            f"{label} {height}x{width} violates the {TILE_HEIGHT}x{TILE_WIDTH} tiling contract"
        )
    return height, width


def require_tile_layout(tensor: dict[str, Any], label: str) -> None:
    if tensor.get("layout") != "tile":
        raise ContractError(f"{label} must use tile layout")


def validate_input(contract: dict[str, Any]) -> None:
    input_tensor = require_mapping(contract.get("input"), "input")
    require_tile_layout(input_tensor, "input")
    logical_height, logical_width = parse_shape(
        input_tensor.get("logical_shape"), "input.logical_shape"
    )
    storage_height, storage_width = require_tile_shape(
        input_tensor.get("storage_shape"), "input.storage_shape"
    )
    expected_height = ((logical_height + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
    expected_width = ((logical_width + TILE_WIDTH - 1) // TILE_WIDTH) * TILE_WIDTH
    if (storage_height, storage_width) != (expected_height, expected_width):
        raise ContractError(
            "input.storage_shape must be the minimal tile expansion of "
            f"{logical_height}x{logical_width}; expected {expected_height}x{expected_width}"
        )


def validate_preprocessing(contract: dict[str, Any]) -> None:
    preprocessing = require_sequence(contract.get("preprocessing"), "preprocessing")
    try:
        padding_index = preprocessing.index("zero_pad")
        split_index = preprocessing.index("split2d")
    except ValueError as error:
        raise ContractError(
            "preprocessing must contain both zero_pad and split2d stages"
        ) from error
    if padding_index >= split_index:
        raise ContractError("zero_pad must precede split2d")


def validate_named_tensors(
    contract: dict[str, Any], field: str, require_layout: bool
) -> list[str]:
    tensors = require_sequence(contract.get(field), field)
    names: list[str] = []
    for index, raw_tensor in enumerate(tensors):
        label = f"{field}[{index}]"
        tensor = require_mapping(raw_tensor, label)
        name = tensor.get("name")
        if not isinstance(name, str) or not name:
            raise ContractError(f"{label}.name must be a non-empty string")
        require_tile_shape(tensor.get("shape"), f"{label}.shape")
        if require_layout:
            require_tile_layout(tensor, label)
        names.append(name)
    return names


def validate_contract(contract: dict[str, Any]) -> None:
    validate_preprocessing(contract)
    validate_input(contract)
    validate_named_tensors(contract, "kernel_shapes", require_layout=False)
    validate_named_tensors(contract, "intermediates", require_layout=True)
    output_names = set(
        validate_named_tensors(contract, "outputs", require_layout=True)
    )
    if output_names != OUTPUT_NAMES:
        missing = sorted(OUTPUT_NAMES - output_names)
        extra = sorted(output_names - OUTPUT_NAMES)
        raise ContractError(
            f"outputs must contain LL, LH, HL, and HH exactly; missing={missing}, extra={extra}"
        )


def main() -> int:
    args = parse_args()
    try:
        with args.contract.open(encoding="utf-8") as source:
            contract = require_mapping(json.load(source), "contract")
        validate_contract(contract)
    except (OSError, json.JSONDecodeError, ContractError) as error:
        print(f"tiling contract violation: {error}", file=sys.stderr)
        return 1
    print(f"tiling contract valid: {args.contract}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
