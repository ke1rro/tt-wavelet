#!/usr/bin/env python3
import argparse
from typing import List

TILE_HEIGHT = 32
TILE_WIDTH = 32
TILE_SIZE = TILE_HEIGHT * TILE_WIDTH


def parse_signal(signal_text: str) -> List[float]:
    values: List[float] = []
    for token in signal_text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def to_tiles_row_major(signal: List[float]) -> List[List[float]]:
    if not signal:
        return [[0.0] * TILE_SIZE]

    tiles: List[List[float]] = []
    total = len(signal)
    for start in range(0, total, TILE_SIZE):
        chunk = signal[start : start + TILE_SIZE]
        if len(chunk) < TILE_SIZE:
            chunk = chunk + [0.0] * (TILE_SIZE - len(chunk))
        tiles.append(chunk)
    return tiles


def pretty_value(value: float):
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return int(rounded)
    return value


def format_cell(value: float) -> str:
    pv = pretty_value(value)
    return f"{str(pv):>6}"


def print_tile(tile: List[float], tile_index: int, prefix_len: int = 64) -> None:
    prefix = [pretty_value(v) for v in tile[:prefix_len]]
    print(f"Tile {tile_index} flattened prefix {prefix}")
    print()
    print(f"Tile {tile_index} (ROW_MAJOR, {TILE_HEIGHT}x{TILE_WIDTH}):")

    for row in range(TILE_HEIGHT):
        start = row * TILE_WIDTH
        row_values = tile[start : start + TILE_WIDTH]
        print(" ".join(format_cell(v) for v in row_values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Print signal as row-major 32x32 tiles")
    parser.add_argument(
        "--signal",
        required=True,
        help='Comma-separated fp32 values, e.g. "15,16,17,18"',
    )
    args = parser.parse_args()

    signal = parse_signal(args.signal)
    tiles = to_tiles_row_major(signal)

    print(f"Signal length: {len(signal)}")
    print(f"Tiles created: {len(tiles)}")
    print("Packing mode: row_major")
    print("Layout mode: row_major")
    print()

    for tile_index, tile in enumerate(tiles):
        print_tile(tile, tile_index)


if __name__ == "__main__":
    main()
