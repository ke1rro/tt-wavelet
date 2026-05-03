FACE_H = FACE_W = 16
TILE_H = TILE_W = 32
BLOCK_W = 16

signal_length = 512
signal = [i for i in range(signal_length)]


def read_block(signal, start, pad_value=0):
    block = []

    for i in range(start, start + BLOCK_W):
        if i < len(signal):
            block.append(signal[i])
        else:
            block.append(pad_value)

    return block


def signal_to_tiles(signal, tiles_count=2, pad_value=0):
    blocks_per_row = tiles_count * 2
    step_blocks_per_row = blocks_per_row - 1

    tiles = []

    for _ in range(tiles_count):
        tile = []

        for _ in range(TILE_H):
            tile.append([pad_value] * TILE_W)

        tiles.append(tile)

    for row in range(TILE_H):
        for block_id in range(blocks_per_row):
            signal_block_id = row * step_blocks_per_row + block_id
            signal_start = signal_block_id * BLOCK_W

            block = read_block(signal, signal_start, pad_value)

            tile_id = block_id // 2
            local_block_id = block_id % 2

            col_start = local_block_id * BLOCK_W
            col_end = col_start + BLOCK_W

            tiles[tile_id][row][col_start:col_end] = block

    return tiles


tiles = signal_to_tiles(signal, tiles_count=2)

tile0 = tiles[0]
tile1 = tiles[1]


rows = len(tile0)
for row in range(rows):
    print(f"Tile0 Row{row}" + " " * 128 + f"Tile1 Row{row}")
    print(tile0[row], " " * 10, tile1[row])
    if all(x == 0 for x in tile0[row]):
        break
