# Tile-native оптимізація LWT ConeStreamed на Wormhole N150

Цей документ описує перехід cone backend із padded `32x32` FP32 tiles на native
`32x16` tiles, два layouts локального L1 workspace, host-side вибір layout і фактично
виміряний результат. Загальна математика lifting scheme, dependency-cone planner і
порівняння `resident`/`cone` описані в [LWT_MEMORY_MODES.md](LWT_MEMORY_MODES.md).

## 1. Результат коротко

Зміна дала гарантований виграш у footprint і traffic granularity, але не універсальне
прискорення кожної wavelet scheme:

| Показник | До | Після | Зміна |
| --- | ---: | ---: | ---: |
| Cone lifting CBs/core | 49,824 B | 41,632 B | **-8,192 B, -16.44%** |
| CB page payload/group | 24,576 B | 20,480 B | **-4,096 B, -16.67%** |
| Intermediate output packets/group, row-major | 96 | 96 | без змін |
| Intermediate output packets/group, tile-native | 96 | 3 | **32x менше packets** |
| FP32 Dst logical occupancy | 4 full tiles | 7 half tiles | 16,384 -> 14,336 B logical |
| Intermediate DRAM passes | 0 | 0 | dependency cone збережено |

На довжині 5,000,000 із `warmup=3`, `repeats=10`:

- `bior3.9`: tile-native **1.041x**, або приблизно **4.1% швидше** за row-major;
- `db7`: tile-native приблизно **1.8% повільніше**, тому auto policy обирає row-major;
- `db7` auto проти попереднього full-tile historical measurement: близько **1.6% швидше**,
  але historical і current runs мають різну кількість repeats, тому це directional result.

Отже, головний результат — не один global speedup, а hybrid backend: narrow compute завжди,
а persistent tile-native workspace використовується лише там, де його page-locality окупає
logical-to-physical remap.

## 2. Що було до зміни

Один LWT output group містить:

```text
32 rows * 3 blocks/row * 16 FP32 = 1536 FP32 = 6144 bytes
```

Старий cone compute транспортував group через full `32x32` pages:

```text
source: 2 * 32x32 = 2048 elements = 8192 B
base:   2 * 32x32 = 2048 elements = 8192 B
output: 2 * 32x32 = 2048 elements = 8192 B
```

У base і output другий tile використовував лише 512 із 1024 FP32 values. На кожному group
зайві 512 base і 512 output positions проходили через CB/unpack/pack page geometry.

Intermediate workspace уже знаходився у local sharded L1. Writer materialize-ив compact
output у row-major layout через 96 окремих 64-byte writes:

```text
32 rows * 3 half-stick blocks = 96 packets/group
```

## 3. Native `32x16` compute geometry

FP32 tile `32x16` містить рівно один 16-element block для кожного з 32 rows:

```text
32 * 16 * 4 bytes = 2048 bytes/page
```

Тому один group тепер має точне представлення:

```text
source: 4 narrow pages = 2048 elements
base:   3 narrow pages = 1536 elements
output: 3 narrow pages = 1536 elements
```

Source лишається 2048-element transport window, бо stencil потребує 1536 output positions,
до 16 halo positions і зручне four-block Dst mapping. Base та output більше не мають
напівпорожніх full tiles.

### 3.1. Dst register mapping

Сім narrow tiles розміщені в чотирьох full FP32 Dst slots:

```text
narrow Dst index: 0      1       2      3       4      5       6
role:             base0  src0    base1  src1    base2  src2    src3

pack W index:     0              1              2
```

Narrow Dst address:

```text
dst_base = tile_index * 32 + face_index * 16
```

На Wormhole unpack MOP уже читає two-face geometry із `32x16` CB metadata. Проблема була
лише в generic FP32 direct-copy math path: він використовує full-tile Dst stride. Тому
`copy_narrow_tile()` не є новою copy instruction; це стандартний unpack плюс явний
`DstTileShape::Tile32x16` math-side address setup.

Фізично `tile_regs_acquire()` все ще резервує чотири full Dst slots. Значення 14,336 B у
таблиці означає logical occupancy, а не додаткові 2 KiB вільного L1.

## 4. Circular-buffer пам'ять

CBs мають double buffering на два groups.

### 4.1. До оптимізації

| CB | Pages | Bytes |
| --- | ---: | ---: |
| `c0`, source tile 0 | `2 * 4096` | 8,192 |
| `c1`, source tile 1 | `2 * 4096` | 8,192 |
| `c2`, base | `4 * 4096` | 16,384 |
| `c16`, output | `4 * 4096` | 16,384 |
| `c3`, input stick cache | `4 * 128` | 512 |
| `c5`, sync token | 1 | 32 |
| `c6`, reader config | 1 | 64 |
| `c7`, writer config | 1 | 64 |
| **Разом/core** |  | **49,824 B** |

### 4.2. Після оптимізації

| CB | Pages | Bytes |
| --- | ---: | ---: |
| `c0`, source narrow 0/1 | `4 * 2048` | 8,192 |
| `c1`, source narrow 2/3 | `4 * 2048` | 8,192 |
| `c2`, base narrow | `6 * 2048` | 12,288 |
| `c16`, output narrow | `6 * 2048` | 12,288 |
| `c3`, input stick cache | `4 * 128` | 512 |
| `c5`, sync token | 1 | 32 |
| `c6`, reader config | 1 | 64 |
| `c7`, writer config | 1 | 64 |
| **Разом/core** |  | **41,632 B** |

Економія 8,192 B/core виникає з base та output CB. Source CB footprint не змінився, бо
чотири half pages займають стільки ж, скільки два full pages.

## 5. Два layouts intermediate workspace

Compute CBs завжди `32x16`. Відмінність стосується лише трьох persistent L1 slots
`A/B/Scratch`.

### 5.1. Row-major

```text
physical_index = logical_index
workspace_alignment = 32 elements
```

Reader збирає narrow pages із linear FP32 stream. Writer розсіює output у 96 contiguous
half-stick blocks. Цей layout вигідний, коли lifting routes часто мають shifted base/source
intervals.

### 5.2. Tile-native

Кожна 1536-element group зберігається як три послідовні narrow pages:

```text
logical  = row * 48 + block * 16 + lane
physical = block * 512 + row * 16 + lane
workspace_alignment = 1536 elements
```

Переваги:

- intermediate writer робить 3 page writes замість 96 half-stick writes;
- aligned base group читається трьома direct 2048-byte page transfers;
- output CB і workspace мають однакове physical представлення.

Ціна:

- shifted source/base interval треба переставити назад у stencil geometry;
- capacity округлюється до 1536, а не 32 elements;
- direct-page path можливий лише для aligned group.

Приклад для 5M benchmark:

| Layout | Slot capacity/core | Три slots/core |
| --- | ---: | ---: |
| row-major | 39,968 FP32 | 479,616 B |
| tile-native | 41,472 FP32 | 497,664 B |
| різниця | +1,504 FP32 | **+18,048 B, +3.76%** |

Тобто tile-native зменшує CB і packet overhead, але може трохи збільшити persistent L1
workspace через сильніше alignment.

## 6. Reader, compute і writer dataflow

### 6.1. Reader

Row-major path має окремі dense і bounds-checked loops. Tile-native path використовує:

- direct page reads для aligned source/base/output groups;
- `WorkspaceIndexCursor` для shifted intervals;
- копіювання 16-element logical block максимум двома contiguous L1 segments;
- один cursor на tile/group замість division/modulo для кожного scalar.

Initialization dependency cone одразу записує initial even/odd у вибраний physical layout.
Policy branch відсутній у scalar hot loop: layout передається як compile-time kernel argument.

### 6.2. Compute

Predict/update route:

```text
4 source narrow CB pages
       + 3 base narrow CB pages
       -> 7 narrow Dst tiles
       -> SFPU stencil-plus-base
       -> optional fused terminal scale
       -> 3 output narrow CB pages
```

Коефіцієнти, order операцій, route offsets і FP32 accumulation не змінені.

### 6.3. Writer

| Output destination | Transfer pattern |
| --- | --- |
| final DRAM | 96 valid 64-byte half-stick writes/group |
| local row-major L1 | 96 stateful 64-byte writes/group |
| local tile-native L1 | 3 stateful 2048-byte writes/group |

Payload intermediate output однаковий — 6144 B/group. Tile-native зменшує packet/address
overhead, а не кількість корисних output bytes.

`noc_async_writes_flushed()` захищає lifetime output CB pages. Route-level
`noc_async_write_barrier()` гарантує visibility перед наступним lifting route. Кількість і
семантика route barriers не змінені.

## 7. Host-side auto selection

Layout обирається один раз під час створення executable:

```text
1. Побудувати row-major cone plan.
2. Порахувати predict/update routes з base_offset_elements == 0.
3. Якщо aligned routes >= 50%, перебудувати plan як tile-native.
4. Скомпілювати відповідний reader/writer variant.
```

Перевизначення для A/B benchmark:

```bash
export TT_WAVELET_LWT_CONE_WORKSPACE_LAYOUT=auto
export TT_WAVELET_LWT_CONE_WORKSPACE_LAYOUT=row-major
export TT_WAVELET_LWT_CONE_WORKSPACE_LAYOUT=tile-native
```

Telemetry:

```text
lwt_tile_native_workspace: 0  # row-major
lwt_tile_native_workspace: 1  # tile-native
```

`compare_timings.py` зберігає це поле як `lwt_tile_native_workspace` у CSV.

## 8. DRAM і L1 traffic

Оптимізація не повертає route-by-route DRAM loopback.

```text
input DRAM
  -> exact initial dependency cone
  -> three local L1 slots
  -> усі predict/update/scale routes у local L1 + CB + Dst
  -> final even/odd interval у DRAM
```

Тому:

- initial input dependency читається з DRAM один раз на chunk, з можливим overlap halo;
- intermediate output не пишеться у DRAM;
- final coefficient пишеться у DRAM один раз;
- tile-native змінює local L1/CB geometry, а не asymptotic DRAM traffic.

## 9. Виміряна продуктивність

### 9.1. Методика

```text
Device: Wormhole N150
Build: Release, ./update.sh Release lwt
Mode: cone
Input: generated FP32 ramp із lwt CLI
Timing: device execution + queue Finish; allocation/program creation поза timer
5M A/B: warmup=3, repeats=10
100k/1M controls: warmup=5, repeats=20
```

### 9.2. Пряме A/B workspace layout

Це найчистіше порівняння, бо binary, narrow compute, device і timing policy однакові.

| Scheme, N | Row-major median | Tile-native median | Auto | Результат |
| --- | ---: | ---: | --- | ---: |
| `db7`, 5M | **21.767 ms** | 22.165 ms | row-major | row-major **1.8% швидше** |
| `bior3.9`, 5M | 13.861 ms | **13.310 ms** | tile-native | tile-native **4.1% швидше** |

Variability:

| Scheme/layout | p10 | p90 | stddev |
| --- | ---: | ---: | ---: |
| `db7` row-major | 21.717 ms | 21.945 ms | 0.136 ms |
| `db7` tile-native | 22.103 ms | 22.240 ms | 0.054 ms |
| `bior3.9` tile-native | 13.240 ms | 13.346 ms | 0.114 ms |
| `bior3.9` row-major | 13.816 ms | 14.204 ms | 0.194 ms |

Effective input throughput:

```text
db7 auto:     5,000,000 / 0.021767 s = ~229.7 M samples/s
bior3.9 auto: 5,000,000 / 0.013310 s = ~375.7 M samples/s
```

### 9.3. Cone проти resident

Цей результат показує виграш dependency-cone architecture загалом, а не лише narrow tiles:

| `db7`, N | Resident median | Cone auto median | Speedup | Lower latency |
| --- | ---: | ---: | ---: | ---: |
| 100k | 3.152 ms | 2.525 ms | **1.25x** | 19.9% |
| 1M | 11.220 ms | 6.389 ms | **1.76x** | 43.1% |

### 9.4. Historical full-tile baseline

`tt_wavelet_timings.csv`, знятий до narrow-tile workspace, має для `db7`:

| N | Historical median | Current median | Directional delta |
| --- | ---: | ---: | ---: |
| 100k | 2.078 ms | 2.525 ms | current приблизно 21.5% повільніше |
| 1M | 6.221 ms | 6.389 ms | current приблизно 2.7% повільніше |
| 5M | ~22.112 ms | 21.767 ms | current приблизно 1.6% швидше |

Це не strict A/B: historical CSV використовував 3 repeats і був знятий в інший момент.
Тому з нього коректно робити лише висновок про crossover: на коротких signals додатковий
narrow page orchestration не окупається, на великих signal/chunks memory locality починає
давати невеликий виграш. Не можна заявляти загальний speedup для всіх `N`.

## 10. Correctness

Виконані перевірки:

- усі 106 scheme JSON у примусових `row-major` і `tile-native` layouts;
- length mismatches: 0;
- `max_abs(row_major, tile_native) = 0` для виведених FP32 coefficients;
- representative `haar`, `db7`, `bior3.9` проти PyWavelets;
- final runtime smoke tests обох auto variants на N150;
- `./update.sh Release lwt`, `git diff --check`, Python syntax check.

Для 36 high-order schemes похибка проти PyWavelets перевищує `1e-2`, але cone output
біт-у-біт збігається з resident. Це існуюча numerical stability межа FP32 lifting
factorization, а не layout regression.

## 11. Змінені компоненти

| Файл | Зміна |
| --- | --- |
| `kernels/compute/lwt_cone_compute.cpp` | 7 narrow Dst tiles, narrow copy wrapper, 3-page output |
| `kernels/sfpi/horizontal_stencil_sfpi.h` | `32x16` Dst addressing, narrow stencil і scale |
| `kernels/dataflow/lwt_cone_reader.cpp` | row-major/native readers, cursor і direct-page paths |
| `kernels/dataflow/lwt_cone_writer.cpp` | 3-page native writer та 96-block row-major/final writers |
| `tt_wavelet/include/device_protocol/lwt_config.hpp` | narrow tile constants |
| `tt_wavelet/include/lifting/cone_plan.hpp` | workspace layout enum і layout-specific alignment |
| `tt_wavelet/src/lifting/cone_device.cpp` | narrow CB allocation, auto policy, compile-time variants |
| `tt_wavelet/include/lifting/device.hpp` | scheduler telemetry field |
| `main.cpp`, `compare_timings.py` | telemetry print, parse і CSV column |

## 12. Як повторити benchmark

```bash
./update.sh Release lwt
source ./scripts/set_env.sh

TT_WAVELET_LWT_CONE_WORKSPACE_LAYOUT=auto \
./build/lwt \
  --benchmark \
  --memory-mode cone \
  --repeats 10 \
  --warmup-runs 3 \
  --length 5000000 \
  bior3.9
```

Для прямого A/B повторіть команду з `row-major` і `tile-native`. Не запускайте два device
benchmarks паралельно й не використовуйте cold run без warmup як performance result.

## 13. Що оптимізувати далі

Найбільший залишковий cost не в DRAM loopback і не в padded output tile. Це:

1. remap shifted source intervals у tile-native reader;
2. 96 final DRAM half-stick packets/group;
3. SFPU shuffle/MAD instruction schedule;
4. launch/config overhead для коротких signals;
5. route-level synchronization, який необхідний для local dependency visibility.

Наступний крок варто вибирати після per-kernel profiler breakdown. Для коротких сигналів
доцільнішим може бути окремий low-launch-overhead path, а не подальше ускладнення
tile-native remap.
