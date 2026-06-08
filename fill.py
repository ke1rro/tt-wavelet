signal = list(range(5054))


TILE_HEIGHT = 32
TILE_WIDTH = 32
FACE_WIDTH = 16
FACE_HEIGHT = 16


def get_tilized_idx(row, col):
    local_row = row % TILE_HEIGHT
    local_col = col % TILE_WIDTH
  
    offset = 0
  
    if (local_col >= FACE_WIDTH):
        local_col -= FACE_WIDTH;  
        offset += FACE_HEIGHT * FACE_WIDTH
  
    if (local_row >= FACE_WIDTH):
        local_row -= FACE_WIDTH;  
        offset += FACE_HEIGHT * TILE_WIDTH
  
    index = offset + local_row * FACE_WIDTH + local_col;  
    return index


def get_idx(row, col):
    tile = col // 32
    return tile * TILE_HEIGHT * TILE_WIDTH + get_tilized_idx(row, col)


def fill_first_tile(ptr, signal, pad, buffer):
    for i in range(pad, 16):
        if not signal:
            break

        ptr[i] = signal.pop(0)

    for i in range(32*48):
        if not signal:
            break

        row = i // 48
        col = (i % 48) + 16

        value = signal.pop(0)
        ptr[get_idx(row, col)] = value
        if col >= 48 and row < 31:
            ptr[get_idx(row + 1, col - 48)] = value
        elif col >= 48 and row == 31:
            buffer[col - 48] = value


def fill_next_tile(ptr, signal, buffer):
    for i in range(16):
        ptr[i] = buffer[i]

    for i in range(32*48):
        if not signal:
            break

        row = i // 48
        col = (i % 48) + 16

        value = signal.pop(0)
        ptr[get_idx(row, col)] = value
        if col >= 48 and row < 31:
            ptr[get_idx(row + 1, col - 48)] = value
        elif col >= 48 and row == 31:
            buffer[col - 48] = value






def get_row_col_from_tilized_idx(index):
    row_block = 0
    col_block = 0

    if index >= FACE_HEIGHT * TILE_WIDTH:
        row_block = 1
        index -= FACE_HEIGHT * TILE_WIDTH

    if index >= FACE_HEIGHT * FACE_WIDTH:
        col_block = 1
        index -= FACE_HEIGHT * FACE_WIDTH

    local_row = index // FACE_WIDTH
    local_col = index % FACE_WIDTH

    if row_block:
        local_row += FACE_WIDTH

    if col_block:
        local_col += FACE_WIDTH

    return local_row, local_col

def invert_ptr(ptr, title=""):
    tile = [[-1] * TILE_WIDTH for _ in range(TILE_HEIGHT)]

    for idx in range(1024):
        r, c = get_row_col_from_tilized_idx(idx)
        tile[r][c] = ptr[idx]

    if title:
        print(f"\n{title}")

    # Column header
    print("     ", end="")
    for c in range(TILE_WIDTH):
        print(f"{c:5}", end="")
    print()

    print("    " + "-" * (TILE_WIDTH * 5))

    for r in range(TILE_HEIGHT):
        print(f"{r:2} | ", end="")
        for c in range(TILE_WIDTH):
            v = tile[r][c]
            if v == -1:
                print("    .", end="")
            else:
                print(f"{v:5}", end="")
        print()


buffer = list(range(16))
pad = 4

is_first = True
while signal:
    ptr = [-1]*2048

    if is_first:
        fill_first_tile(ptr, signal, pad, buffer)
    else:
        fill_next_tile(ptr, signal, buffer)

    is_first = False

    print("NEXT SPLICE")
    print("Tile 1:")
    invert_ptr(ptr)
    print("Tile 2:")
    invert_ptr(ptr[1024:])
    print()
