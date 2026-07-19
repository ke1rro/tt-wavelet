# LWT на Wormhole: `ResidentSharded` і `ConeStreamed`

Цей документ описує поточну forward Lifting Wavelet Transform у `tt-wavelet`: математику,
host-side планування, розміщення даних у DRAM/L1, роботу reader/compute/writer kernels на
Wormhole, синхронізацію, формули місткості та практичні обмеження.

Окремий виміряний звіт про перехід на `32x16`, точний CB/workspace footprint і performance
A/B наведено в [LWT_TILE_NATIVE_OPTIMIZATION.md](LWT_TILE_NATIVE_OPTIMIZATION.md).

Основні файли реалізації:

- [forward planner](../tt-wavelet/tt_wavelet/include/lifting/plan.hpp);
- [dependency-cone planner](../tt-wavelet/tt_wavelet/include/lifting/cone_plan.hpp);
- [Resident host runtime](../tt-wavelet/tt_wavelet/src/lifting/device.cpp);
- [Cone host runtime](../tt-wavelet/tt_wavelet/src/lifting/cone_device.cpp);
- [Resident reader](../tt-wavelet/kernels/dataflow/lwt_reader.cpp),
  [compute](../tt-wavelet/kernels/compute/lwt_compute.cpp) і
  [writer](../tt-wavelet/kernels/dataflow/lwt_writer.cpp);
- [Cone reader](../tt-wavelet/kernels/dataflow/lwt_cone_reader.cpp),
  [compute](../tt-wavelet/kernels/compute/lwt_cone_compute.cpp) і
  [writer](../tt-wavelet/kernels/dataflow/lwt_cone_writer.cpp);
- [SFPU horizontal stencil](../tt-wavelet/kernels/sfpi/horizontal_stencil_sfpi.h).

## 1. Коротко про два режими

Обидва режими використовують однакову lifting-схему, FP32 SFPU compute kernel і три логічні
слоти `A`, `B`, `Scratch`. Відмінність полягає в тому, скільки сигналу одночасно живе в L1 і
як робота розподіляється між Tensix cores.

| Властивість | `ResidentSharded` | `ConeStreamed` |
| --- | --- | --- |
| CLI-значення | `resident` | `cone` |
| Початкові even/odd | Увесь padded signal одразу split-иться в L1 shards | Кожен core читає з DRAM лише залежності свого chunk |
| Intermediate streams | Height-sharded між cores | Локальні у L1 конкретного core |
| Взаємодія cores | Remote L1 halo reads і global barrier після route | Chunks незалежні; між cores немає route barrier |
| Повторне читання input | Один повний pad/split pass | Halo сусідніх chunks може читатися повторно |
| L1 requirement | Три повнорозмірні sharded slots | Три slots розміру найбільшого локального cone |
| Коли не влазить | Завершується `TT_FATAL` | Збільшує кількість chunks, доки workspace не вміститься |
| Final output | Дві terminal scale routes пишуть у DRAM | Planner може злити scale одного stream з останнім predict/update; кожен chunk пише свій interval |
| Compute tiles | `32x32` FP32 | Native `32x16` FP32 |
| Intermediate physical layout | Row-major sticks у sharded L1 | `row-major` або persistent `tile-native`, вибір один раз на host |

Поточний CLI за замовчуванням вибирає `cone`. Автоматичного перемикання
`resident -> cone` при нестачі L1 зараз немає.

## 2. Математична модель

### 2.1. Boundary padding і even/odd split

Нехай вхідний сигнал має довжину `N`, а `T = tap_size` wavelet-схеми. Planner використовує

```text
p = T - 1
padded_length = N + 2p
```

Forward ConeStreamed та ConeStreamed ILWT приймають вісім non-periodization modes; default —
`symmetric`:

```bash
./build/lwt --memory-mode cone --boundary-mode symmetric ...
./build/lwt --memory-mode cone --boundary-mode zero ...
./build/lwt --memory-mode cone --boundary-mode constant ...
./build/lwt --memory-mode cone --boundary-mode periodic ...
./build/lwt --memory-mode cone --boundary-mode antisymmetric ...
./build/lwt --memory-mode cone --boundary-mode smooth ...
./build/lwt --memory-mode cone --boundary-mode reflect ...
./build/lwt --memory-mode cone --boundary-mode antireflect ...
```

Для довільного логічного індексу `q = t - p`:

```text
zero:      x_pad[t] = 0,                         якщо q < 0 або q >= N
constant:  x_pad[t] = x[clamp(q, 0, N - 1)]
periodic:  x_pad[t] = x[q mod N]
```

Half-sample symmetric extension має період `2N`:

```text
r = q mod 2N,  0 <= r < 2N
rho_N(q) = r,                 якщо r < N
           2N - 1 - r,       якщо r >= N
x_pad[t] = x[rho_N(t - p)]
```

Наприклад, крайні samples повторюються при віддзеркаленні. Для whole-sample `reflect` вони не
повторюються; при `N > 1` mapping має період `2(N-1)`:

```text
r = q mod 2(N-1),  0 <= r < 2(N-1)
sigma_N(q) = r,                  якщо r <= N-1
             2(N-1) - r,        якщо r > N-1
x_pad[t] = x[sigma_N(t - p)]
```

Як і PyWavelets DWT, `reflect` вимагає `N > 1`.

`antisymmetric` використовує той самий half-sample reflected source index, але знак
чергується для кожного наступного reflected segment довжини `N`:

```text
... -x[1] -x[0] | x[0] x[1] ... x[N-1] | -x[N-1] -x[N-2] ...
```

Для `smooth`, коли `N > 1`, продовжується перша edge difference:

```text
x[-k]      = x[0]   + k * (x[0]   - x[1])       для k >= 1
x[N-1+k]   = x[N-1] + k * (x[N-1] - x[N-2])     для k >= 1
```

Для `N=1` DWT не має визначеної edge difference, тому використовується zero slope, тобто
constant extension. Це відповідає PyWavelets DWT (public `pywt.pad` для цього degenerate case
має іншу поведінку).

У першому reflected segment `antireflect` є whole-sample odd reflection навколо значення edge:

```text
x[-k]      = 2*x[0]   - x[k]
x[N-1+k]   = 2*x[N-1] - x[N-1-k]
```

Device accessor використовує closed-form repeated odd reflection також коли padding ширший за
сигнал. Як і PyWavelets DWT, цей mode вимагає `N > 1`.

У Cone mode boundary policy є compile-time specialization reader kernel. Усі вісім modes
мають спільний прямий interior path; mode-specific робота виконується лише для prefix/suffix
padding. Повний padded signal не матеріалізується. `ResidentSharded` наразі підтримує тільки
`symmetric`.

ILWT не екстраполює original signal повторно. Він інвертує canonical coefficients у потрібні
інтервали padded even/odd streams і crop-ить `[p, p+N)`. Тому inverse arithmetic і crop geometry
однакові для всіх восьми modes; mode в ILWT API фіксує контракт походження coefficients і
перевіряє допустиму комбінацію length/mode без додаткової роботи у device hot path.

Blackhole P150b validation початкових `zero/constant` включає:

- 36 short-signal cases: `db1`, `db2`, `db7`, `bior3.9`, усі три modes, `N=1,2,3`;
- 54 odd/even і group-boundary cases: `db2`, `db7`, `bior3.9`, усі три modes,
  `N=17,32,33,3071,3072,3073`;
- 12 forced-layout cases для `db7` із `zero/constant`, `N=33,3073`, layouts
  `row-major`, `tile-native`, `auto`;
- 212 runtime-stability cases: усі 106 schemes, `zero/constant`, `N=33`, із перевіркою
  output shape та finite values.

Перші три групи порівнювали coefficients з PyWavelets; найбільша absolute error була
`2.98143605e-05`. All-scheme runtime sweep навмисно не застосовує єдиний PyWavelets
tolerance: відома похибка high-order FP32 lifting factorization відстежується окремо від
boundary geometry. Відтворити перевірку можна через
`scripts/validate_lwt_boundaries.py`; для stability sweep використовуйте
`--all-schemes --modes zero constant --lengths 33 --runtime-only`.

Для `periodic/antisymmetric/smooth/antireflect` додатково виконано:

- 72 forward PyWavelets cases: `db2`, `db7`, `bior3.9`, чотири modes,
  `N=17,32,33,3071,3072,3073`, max absolute error `2.98143605e-05`;
- 24 forced-layout forward cases: `db7`, `N=33,3073`, три layouts;
- 424 forward runtime-stability cases: 106 schemes, чотири modes, `N=33`;
- 168 ILWT/PyWavelets cases: `db1`, `db7`, `bior3.9`, сім раніше підтримуваних modes,
  `N=2,3,17,32,33,3071,3072,3073`, max absolute error `3.29005435e-05`;
- 42 forced-layout ILWT cases, bitwise identical між `row-major`, `tile-native`, `auto`;
- 424 ILWT runtime/JIT cases: 106 schemes, чотири нові modes, `N=33`.

Для `reflect` окремо виконано:

- 36 forward PyWavelets cases: `db1`, `db2`, `db7`, `bior3.9`,
  `N=2,3,17,31,32,33,3071,3072,3073`, max absolute error `2.98143605e-05`;
- 6 forced-layout forward cases для `db7`, `N=33,3073`;
- 106/106 forward runtime/JIT cases на `N=33`;
- 36 ILWT/PyWavelets cases на тій самій representative matrix, max absolute error
  `3.29005435e-05`;
- 6 forced-layout ILWT cases, bitwise identical між `row-major`, `tile-native`, `auto`;
- 106/106 ILWT runtime/JIT cases на `N=33`.

Host mapping також перевірено проти `pywt.pad(reflect)` у 6300 комбінаціях `N=2..64` та
padding width `1..100`, включно з padding, ширшим за сигнал.

Wormhole N150 L revalidation 2026-07-19 (`TT-KMD 2.3.0`, firmware bundle `19.11.0.0`,
project `6738a33305aa`, TT-Metal `f87c34a93ee4`) дала:

- 312/312 representative forward PyWavelets cases для всіх восьми modes, включно з short,
  odd/even і 3072-group boundaries; max absolute error `2.98143605e-05`;
- 848/848 runtime/JIT cases (`106 schemes * 8 modes`) на `N=33`;
- 106/106 schemes bitwise identical між production ConeStreamed і ResidentSharded на
  однаковому symmetric input;
- current PyWavelets count `70/106` при absolute tolerance `1e-2`, тобто точно історичний
  Wormhole baseline; 36 high-order failures лишаються FP32 factorization питанням;
- усі 106 JSON schemes точно відповідають згенерованим static headers.

Під час цієї перевірки tile-native `antireflect` reader перевищив 16 KiB Wormhole NCRISC
instruction region на 280 bytes. Boundary-only extension і source-cache lookup винесені в один
callable slow path лише для `ARCH_WORMHOLE`; direct interior path лишився inline, а Blackhole
inline path не змінений. Після cleanup reader `.text` становить 15,004 bytes, тобто має
1,380 bytes запасу, і наведені вище matrices пройшли повторно.

Device-only spot A/B для найбільш arithmetic-heavy boundary mode (`db7`, auto layout,
5 warmups, 30 repeats) не показав material regression від `smooth`: median на `N=1,000,000`
становила `2.723318 ms` проти `2.710964 ms` для `symmetric`, а на `N=5,000,000` —
`7.391035 ms` проти `7.381310 ms`. Різниця менша за run-to-run spread цього вимірювання;
це не окрема speedup claim. Архітектурно кількість full DRAM passes, L1 workspace і route
barriers не змінилась; додаткова affine arithmetic виконується лише для bounded edges.

Для `reflect` окремий `db7`, `N=5,000,000`, auto-layout A/B із 50 warmups і 100 repeats дав
median `7.448676 ms` проти `7.373625 ms` для `symmetric` (приблизно `+1.0%`); p10/p90
частково перекриваються. Active cores (`110`), workspace (`23072` elements), dependency
overhead (`0.000287`), DRAM passes, L1 footprint і barriers однакові.

Після padding сигнал ділиться на дві polyphase-послідовності:

```text
e0[i] = x_pad[2i]
o0[i] = x_pad[2i + 1]
```

Тому

```text
Le = ceil(padded_length / 2)
Lo = floor(padded_length / 2)
```

Початкові Laurent shifts беруться зі scheme JSON:

```text
even_state = (delay_even, Le)
odd_state  = (delay_odd,  Lo)
```

### 2.2. Valid convolution і route geometry

Для source stream `s`, коефіцієнтів `h[0..k-1]` і kernel shift `sigma` SFPU обчислює valid
convolution:

```text
C_h(s)[i] = sum(j=0..k-1) h[j] * s[i + k - 1 - j]
```

Кожен результат залежить від contiguous source interval довжини `k`. Якщо source має стан
`(delta_s, Ls)`, тоді для нормального випадку `Ls >= k`:

```text
delta_conv = delta_s + sigma + k - 1
Lconv      = Ls - k + 1
```

У коді використано `min(Ls, k)` і нульову довжину convolution для `Ls < k`, щоб geometry
лишалася визначеною і для дуже коротких сигналів.

Результат lifting step є перетином valid convolution та base stream `(delta_b, Lb)`:

```text
delta_out = max(delta_b, delta_conv)
end_out   = min(delta_b + Lb, delta_conv + Lconv)
Lout      = max(0, end_out - delta_out)

source_offset = delta_out - delta_conv
base_offset   = delta_out - delta_b
```

Device route тому виконує:

```text
y[i] = base[base_offset + i]
     + C_h(source)[source_offset + i],    0 <= i < Lout
```

Це важлива властивість: kernels не намагаються відновлювати Laurent geometry. Host planner
один раз перетворює її на компактні `length/offset` поля.

### 2.3. Lifting steps

Підтримуються п'ять типів кроків:

```text
Predict:    odd  <- odd  + P(even)
Update:     even <- even + U(odd)
ScaleEven:  even <- a * even
ScaleOdd:   odd  <- d * odd
Swap:       swap(even, odd)
```

`Predict` і `Update` використовують route geometry вище. `Scale` множить кожен sample на
compile-time FP32 scalar. `Swap` не запускає device kernel: planner лише міняє metadata
активних streams.

Схема компілюється статично. Типи steps, shifts, кількість і FP32 bit patterns коефіцієнтів
вбудовуються в compute kernel; runtime config містить лише адреси, довжини, offsets і work
ranges.

### 2.4. Три storage slots

Замість чотирьох `even/odd x ping/pong` buffers використовується:

```text
A       — початковий even
B       — початковий odd
Scratch — вільний output slot
```

Для кожного `Predict`/`Update` результат пишеться у вільний slot, а старий target стає
вільним:

```text
initial: active_even=A, active_odd=B, free=Scratch

Predict:
    free = active_odd + P(active_even)
    released old active_odd becomes the new free slot

Update:
    free = active_even + U(active_odd)
    released old active_even becomes the new free slot
```

Source і base ніколи не перезаписуються під час того самого route. Через це compute можна
паралелити по output samples без in-route data hazards.

Схема мусить завершуватись рівно двома різними scale routes — однією для even і однією для
odd. У resident їх output перенаправляється прямо у final DRAM buffers. У cone planner може
злити scale stream-а, який змінює останній predict/update, у цей самий SFPU route; друга
scale route залишається окремою. В обох випадках четвертий повнорозмірний L1 slot не
потрібний.

### 2.5. Canonical output

Внутрішні final streams можуть мати scheme-specific shifts і довжини. Публічний output
канонізується на host до

```text
canonical_length = (N + T - 1) / 2
canonical_start  = T / 2
```

з використанням `final_even_shift` і `final_odd_shift`. Device buffers зберігають повні
внутрішні final streams; canonical slicing відбувається після readback і не входить у
benchmark execution time.

### 2.6. Короткий приклад для `db7`

Для `db7`: `T=14`, тому `p=13`, `delay_even=3`, `delay_odd=4`. Якщо `N` парне, initial
even та odd мають однакову довжину

```text
L = (N + 26) / 2 = N/2 + 13.
```

Перший step — `Predict`, `k=1`, `sigma=0`, coefficient `5.093498...`. Planner отримує:

```text
source_offset = 1
base_offset   = 0
output shift  = 4
output length = L - 1

odd1[i] = odd0[i] + 5.093498... * even0[i + 1]
```

Другий step — `Update`, `k=2`, `sigma=0`. Для coefficients `u0`, `u1`:

```text
source_offset = 0
base_offset   = 2
output shift  = 5
output length = L - 2

even1[i] = even0[i + 2] + u0 * odd1[i + 1] + u1 * odd1[i]
```

Саме ці offsets, а не delays чи Laurent degrees, потрапляють у 64-byte runtime route
config. Наступні steps повторюють той самий механізм.

## 3. Одиниці layout на Wormhole

Кожен Wormhole Tensix має близько 1.5 MiB фізичного L1 SRAM, але цей простір спільний для
даних, circular buffers, kernel code, stack та runtime infrastructure. Поточна реалізація
має такі фіксовані layout constants:

| Одиниця | Розмір |
| --- | ---: |
| Scalar | FP32, 4 bytes |
| Stick | 32 FP32 = 128 bytes |
| Full tile | `32 x 32` FP32 = 4096 bytes |
| Narrow tile | `32 x 16` FP32 = 2048 bytes |
| Half-stick block | 16 FP32 = 64 bytes |
| LWT output group | `32 rows x 3 blocks x 16` = 1536 elements = 6144 bytes |
| Route config page | 16 `uint32_t` = 64 bytes |
| Cone chunk config page | 16 `uint32_t` = 64 bytes |

Один group навмисно формує 1536 output elements. Resident зберігає старе представлення у
двох full tiles, тоді як cone pipeline тепер використовує тільки потрібні `32x16` pages:

```text
Cone source:  4 narrow tiles = 2048 scalar positions
Cone base:    3 narrow tiles = 1536 scalar positions
Cone output:  3 narrow tiles = 1536 scalar positions
```

Source потребує 1552 логічні значення для 1536 outputs плюс максимальний 16-element halo;
решта narrow source capacity є padding. На base/output більше немає 512-element
напівпорожнього full tile.

`source_left_pad = 17 - k` — це внутрішнє вирівнювання SFPU stencil до його 17-tap
capacity. Воно не є boundary padding початкового сигналу.

## 4. Як Tensix виконує route

На кожному активному Wormhole Tensix одночасно працюють три частини pipeline:

```text
Reader data-movement RISC-V
    -> формує source/base tiles у circular buffers
Compute engine / SFPU
    -> stencil-plus-base або scale
Writer data-movement RISC-V
    -> записує compact output у L1 або final DRAM
```

Compute kernel розгортає static scheme через C++ templates. Resident для одного
predict/update group:

1. забирає два source tiles і два base tiles з CB;
2. копіює їх у destination registers;
3. виконує `hstencil_plus_base_tile<k>`;
4. pack-ить два output tiles у output CB.

Cone використовує сім narrow tiles у тих самих чотирьох FP32 Dst slots:

```text
Dst narrow indices: base={0,2,4}, source={1,3,5,6}
Pack W indices:     output={0,1,2}
```

Його route копіює 4 source і 3 base pages, виконує
`hstencil_plus_base_narrow_tiles<k>` і pack-ить рівно 3 output pages. Окремої Metalium
операції `copy_tile` для `32x16` не потрібно: unpack MOP уже бере two-face geometry з CB
metadata. Але generic Wormhole FP32 direct-copy math path жорстко використовує stride
`32x32`, тому cone має малий math-side wrapper, який задає
`DstTileShape::Tile32x16` перед unpack-to-Dst. Це workaround адресації Dst, а не новий
формат копіювання.

CBs мають buffering на два groups, тому reader, compute і writer можуть перекривати роботу.
Однак lifting routes самі по собі послідовні: наступний route може використовувати output
попереднього.

## 5. `ResidentSharded`

### 5.1. Host-side shard planner

Planner спочатку знаходить максимальну кількість groups серед initial even/odd та всіх
executable routes:

```text
Gmax = max(ceil(initial_max_length / 1536),
           ceil(route_output_length / 1536),
           1)
```

Нехай `C_limit` — менше з кількості hardware worker cores та
`TT_WAVELET_LWT_MAX_CORES`. Тоді:

```text
groups_per_shard = ceil(Gmax / C_limit)
active_cores     = ceil(Gmax / groups_per_shard)
shard_elements   = groups_per_shard * 1536
```

Кожен із трьох slots є `BufferType::L1`, `HEIGHT_SHARDED`, row-major buffer з:

```text
page shape  = [1, 32]
shard shape = [shard_elements / 32, 32]
```

Границя shard завжди кратна цілому LWT group, тому output group не розривається між cores.

### 5.2. Resident L1 capacity

Signal storage на одному core:

```text
B_resident_per_core = 3 * shard_elements * 4 bytes
                    = 3 * groups_per_shard * 1536 * 4
```

Значення порівнюється з `TT_WAVELET_L1_SIGNAL_BUDGET_BYTES`. Default — `768 * 1024`
bytes/core. Це бюджет лише для трьох signal slots, а не декларація всього вільного L1.
Консервативне значення залишає простір для CBs, kernels, stack і TT-Metal runtime state.

За default budget максимально можливе:

```text
floor(768 KiB / (3 * 1536 * 4)) = 42 groups/core
slot capacity/core = 42 * 1536 = 64,512 FP32
three slots/core   = 774,144 bytes = 756 KiB
```

На поточному N150 runtime експонує 64 worker cores. Це дає аналітичну aggregate capacity
одного stream:

```text
64 * 64,512 = 4,128,768 FP32 elements
```

тобто приблизно `8,257,536` input elements до врахування `2*(tap_size-1)` padding та
округлення. Це не гарантія для кожної конфігурації: фактична кількість cores береться з
`compute_with_storage_grid_size()`, а env може її зменшити.

Якщо три slots перевищують budget, поточний `ResidentSharded` завершується помилкою. Він не
створює частковий resident cache і не перемикається автоматично у `ConeStreamed`.

### 5.3. Resident preprocess dataflow

Перед lifting запускається окрема pad/split program:

```text
input DRAM
  -> reader: 8-stick read cache + symmetric mapping
  -> CB even / CB odd
  -> writer: 128-byte stick writes
  -> L1 slot A / L1 slot B, height-sharded
```

Work partition відповідає shard ownership. Кожен core формує діапазон padded input, який
належить його even/odd shard. Partial final stick доповнюється фізично, але logical length
лишається окремим metadata.

Pad/split CB footprint на активному core:

| CB | Призначення | Bytes |
| --- | --- | ---: |
| `c0` | even output, 2 sticks | 256 |
| `c1` | odd output, 2 sticks | 256 |
| `c2` | input cache, 8 sticks | 1024 |
|  | Разом | 1536 |

Це окрема program і вона завершується до запуску lifting program.

### 5.4. Resident route dataflow

Для route `r` core `c` отримує статичний group range:

```text
group_begin = c * groups_per_shard
group_count = intersection(core shard groups, route output groups)
```

Тому output writes потрапляють у shard того самого core. Коли streams після valid
convolution стають коротшими, cores у кінці списку можуть мати `group_count=0`, але вони все
одно беруть участь у route barrier.

Повний шлях одного route:

```text
64-byte route config із DRAM
  -> source/base TensorAccessor для height-sharded L1
  -> 4-stick source і base read caches
  -> source/base tile CBs
  -> SFPU stencil-plus-base або scale
  -> output tile CB
  -> local owner L1 shard або terminal DRAM
  -> NoC write barrier
  -> global route barrier
  -> local c5 token дозволяє reader почати наступний route
```

Основна частина source/base зазвичай розташована на тому самому core. На межі shards reader
може читати сусідній L1 через NoC. `TensorAccessor` виконує physical mapping, тому kernel
працює з логічними stick indices, не обчислюючи NoC coordinates вручну.

### 5.5. Resident synchronization

Після кожного route, крім останнього, writer:

1. виконує `noc_async_write_barrier()`, щоб output став видимим;
2. входить у global semaphore barrier усіх active cores;
3. core 0 збирає `active_core_count - 1` arrivals і розсилає release;
4. writer push-ить локальний token у `c5`;
5. reader наступного route чекає цей token.

Global barrier потрібний через remote halo reads: core не може почати читати shard сусіда,
поки сусід не завершив попередній route. Це також означає, що надмірна кількість cores може
погіршити час на коротких або сильно cropped routes.

### 5.6. Resident lifting CB footprint

Для FP32 tile size 4096 bytes:

| CB | Призначення | Allocation |
| --- | --- | ---: |
| `c0` | source tile 0 | `2 x 4096 = 8192` B |
| `c1` | source tile 1 | `2 x 4096 = 8192` B |
| `c2` | base/scale tiles | `4 x 4096 = 16384` B |
| `c16` | output tiles | `4 x 4096 = 16384` B |
| `c3` | source stick cache | `4 x 128 = 512` B |
| `c4` | base stick cache | `4 x 128 = 512` B |
| `c5` | route sync token | 32 B |
| `c6` | reader config page | 64 B |
| `c7` | writer config page | 64 B |
|  | Разом | **50,336 B/core** |

CB footprint не входить у `TT_WAVELET_L1_SIGNAL_BUDGET_BYTES`.

## 6. `ConeStreamed` / dependency cone

### 6.1. Основна ідея

Замість зберігання повних streams planner бере interval фінального output — chunk — і йде
lifting-схемою назад. Він знаходить мінімальні contiguous intervals initial even/odd, від
яких залежить chunk:

```text
final output interval Q
  <- backward dependency propagation
initial even interval + initial odd interval
  -> завантаження лише цих samples із input DRAM
  -> усі lifting routes локально в L1
  -> лише Q записується у final DRAM
```

Сусідні chunks мають overlapping halo, тому частина input може читатися і обчислюватися
повторно. Натомість жоден intermediate stream не матеріалізується у DRAM.

### 6.2. Interval algebra

Використовуються half-open intervals `I = [begin, end)`. Визначимо:

```text
translate(I, a, r) = [I.begin + a, I.end + a + r)
hull(I, J)          = найменший interval, що містить I та J
```

`r = k - 1` додає stencil halo: output interval довжини `m` потребує source interval
довжини `m + k - 1`.

Нехай `after.even/odd` — потрібні intervals після route, а `s` і `b` — forward
`source_offset` та `base_offset`.

Для `Predict`:

```text
before.even = hull(after.even,
                   translate(after.odd, s, k - 1))
before.odd  = translate(after.odd, b, 0)
```

Для `Update`:

```text
before.even = translate(after.even, b, 0)
before.odd  = hull(after.odd,
                  translate(after.even, s, k - 1))
```

Для scale потрібний той самий interval відповідного stream. Для `Swap` requirements even
і odd міняються місцями.

`hull` важливий: незмінений stream може бути потрібний майбутнім route ширше, ніж interval,
необхідний лише для поточного stencil. Backward pass зберігає обидві вимоги.

Після проходу всіх routes planner отримує `initial_even` та `initial_odd`. Для кожного
forward route він переводить global intervals у offsets відносно локального storage цього
chunk.

### 6.3. Як вибираються chunks

Фінальний простір ділиться лише по границях 1536-element groups:

```text
final_group_count = ceil(max(final_even_length, final_odd_length) / 1536)
initial_chunk_count = min(final_group_count, core_limit)
```

Groups розподіляються між chunks максимально рівномірно. Для кожного кандидата planner
обчислює точний backward cone і найбільший local stream на будь-якому route:

```text
workspace_elements = round_up(max_chunk_max_workspace,
                              layout == tile-native ? 1536 : 32)
B_cone_per_core      = 3 * workspace_elements * 4 bytes
```

Якщо workspace не вкладається у L1 signal budget, `chunk_count` подвоюється, не перевищуючи
`final_group_count`. Менший chunk зменшує body workspace, але фіксований halo лишається,
тому overhead відносно output зростає.

Мінімальна гранулярність — один output group. Якщо навіть one-group cone не вкладається,
planner завершується помилкою.

`active_core_count = min(chunk_count, core_limit)`. Якщо chunks більше за cores, кожен core
послідовно обробляє contiguous range chunks, повторно використовуючи ті самі три local L1
slots.

### 6.4. Dependency overhead

Для chunk telemetry обчислює element-level metric:

```text
final_elements      = len(final_even) + len(final_odd)
dependency_elements = len(initial_even) + len(initial_odd)

dependency_overhead = max(dependency_elements - final_elements, 0)
                      / final_elements
```

Це аналітичний overhead у кількості split-stream elements. Він не дорівнює точному DRAM
traffic overhead, бо 128-byte sticks, 4-stick cache, boundary clipping і повторні config
reads змінюють фактичну кількість NoC transactions.

Сканування всіх 106 scheme JSON у цьому repository показує:

| Показник | Значення |
| --- | ---: |
| Максимальна кількість coefficients в одному predict/update | 9 |
| Device coefficient capacity | 17 |
| Найбільший `tap_size` | 102 (`coif17`) |
| Найбільша кількість scheme steps | 55 (`coif17`) |
| Schemes зі `Swap` | 58 із 106 |
| Порушення terminal-scale contract | 0 |

Для одного внутрішнього chunk розміром 1536 elements у кожному final stream точний backward
planner дає:

| Scheme | Initial dependency | Extra elements | Element overhead |
| --- | ---: | ---: | ---: |
| `db7` | `1542 + 1542` | 12 total | 0.390625% |
| `coif17` | `1586 + 1586` | 100 total | 3.255208% |

Ці числа є аналітичними для внутрішнього one-group interval великого сигналу. Boundary
clipping може зменшити абсолютний cone, але короткий partial tail chunk іноді має більший
відносний overhead через менший знаменник. Тому runtime
`lwt_max_dependency_overhead` може бути вищим за значення цієї таблиці. Multi-group chunks
зазвичай мають меншу відносну частку halo.

### 6.5. Cone workspace layouts

Кожен із трьох workspace slots також створений як `L1 HEIGHT_SHARDED` MeshBuffer, але
semantics інша, ніж у resident:

```text
shard shape/core = [workspace_elements / 32, 32]
```

Кожен core використовує тільки власний shard як локальні `A/B/Scratch`. Device routes
містять local L1 base addresses, а kernels читають і пишуть intermediate data прямими
`tt_l1_ptr` accesses. Cone ніколи не читає intermediate shard іншого core.

Всередині shard підтримуються два physical layouts:

- `row-major`: логічний index дорівнює фізичному scalar index; reader збирає narrow tiles,
  writer розсіює кожен 16-element block назад у linear workspace;
- `tile-native`: кожна 1536-element group зберігається як три послідовні `32x16` pages;
  output writer робить три 2048-byte writes без transpose/scatter, а вирівняний base reader
  читає ті самі три pages напряму.

У native group logical порядок є row-major по трьох 16-element blocks, тоді як physical
порядок є block-major:

```text
logical index = row * 48 + block * 16 + lane
physical      = block * 512 + row * 16 + lane
```

Host приймає layout-рішення один раз при створенні executable. `auto` спочатку будує
row-major plan і вибирає `tile-native`, якщо принаймні половина predict/update routes має
`base_offset_elements == 0`. Саме ці routes отримують direct three-page base read; shifted
base все одно потребує remap. Для native layout capacity округлюється до 1536, щоб жодна
physical group не перетнула slot boundary.

### 6.6. Cone device dataflow

Для кожного local chunk reader виконує:

```text
chunk config із DRAM
  -> initial_even/initial_odd intervals
  -> input DRAM через 4-stick cache
  -> symmetric mapping + even/odd split on the fly
  -> local L1 slot A / slot B

для кожного route:
  route config із DRAM
  -> direct local L1 source/base reads
  -> source/base tile CBs
  -> SFPU compute
  -> output tile CB
  -> direct local L1 stores або final DRAM NoC writes
```

Reader має два compile-time варіанти packing loop і вибирає між ними один раз на group:

- dense body: усі `1536` base/output elements і `1552` source elements (output group плюс
  16-element stencil halo) гарантовано лежать у local interval, тому scalar bounds checks
  усередині гарячої петлі відсутні;
- boundary/tail: зберігається checked path із left padding, right clipping та zero fill.

Для tile-native workspace reader додатково має direct-page path для вирівняних groups і
`WorkspaceIndexCursor` для shifted source/base intervals. Cursor копіює 16-element logical
block максимум двома contiguous L1 segments; scalar division/modulo всередині hot loop не
виконується.

Це не змінює dependency cone, порядок lifting routes або tile layout. Оптимізація лише
виносить перевірку interval за межі scalar loop, тому крайові значення обчислюються так
само, а внутрішні multi-group chunks не платять за перевірку кожного L1 access.

На відміну від resident, окрема pad/split program не запускається. Initialization потрібного
cone входить у reader kernel тієї ж lifting program.

У row-major workspace intermediate writer переносить 96 вирівняних 16-element blocks з
output CB у local L1 stateful 64-byte NoC writes. У tile-native workspace той самий group
потребує лише трьох 2048-byte page writes. Після group `noc_async_writes_flushed()` чекає,
доки payloads покинуть source CB, і лише тоді writer звільняє його pages. Один повний NoC
barrier наприкінці route гарантує завершення записів до того, як наступний lifting route
почне читати workspace. Діагностичний scalar-L1 variant можна ввімкнути environment
variable, але він не є default path.

Final writer пише такі самі 64-byte half-stick blocks у global DRAM interval
`output_offset + local_index`. Ознака final DRAM є явним route flag, а не наслідком
`StepType`, тому fused predict/update також може бути final writer. Після кожного DRAM
output group виконується NoC write barrier, щоб обмежити кількість outstanding writes і не
звільнити CB page завчасно.

Terminal-scale fusion є compile-time спеціалізацією kernel, без runtime branch у SFPU loop.
Planner в auto mode вмикає її, коли кількість final output groups більша за доступну кількість
cores — тобто коли хоча б частина chunks обробляє кілька groups. Scale множиться на FP32
результат останнього stencil до pack у L1, відповідна окрема scale route вилучається, а
результат одразу йде у final DRAM. `Swap` після останнього predict/update враховується як
metadata permutation при виборі even/odd scale.

### 6.7. Cone synchronization

Chunks незалежні за побудовою. Тому:

- між cores немає global semaphore barrier;
- немає remote L1 halo reads;
- `c5` використовується лише як локальний writer-to-reader token між послідовними routes і
  chunks того самого core;
- різні cores можуть одночасно обробляти різні final intervals.

Це одна з причин, чому `cone` іноді швидший навіть тоді, коли весь сигнал формально влазить
у resident L1: він обмінює невелике дублювання halo на відсутність global route barriers і
remote intermediate reads. Це потрібно перевіряти benchmark-ом для конкретної scheme та
довжини, а не вважати універсальним правилом.

### 6.8. Cone lifting CB footprint

Cone не потребує окремого base stick cache, бо intermediate source/base є прямими local L1
arrays:

| CB | Призначення | Allocation |
| --- | --- | ---: |
| `c0` | source narrow pages 0/1 | `4 x 2048 = 8192` B |
| `c1` | source narrow pages 2/3 | `4 x 2048 = 8192` B |
| `c2` | base/scale narrow pages | `6 x 2048 = 12288` B |
| `c16` | output narrow pages | `6 x 2048 = 12288` B |
| `c3` | input DRAM stick cache | `4 x 128 = 512` B |
| `c5` | local sync token | 32 B |
| `c6` | reader config page | 64 B |
| `c7` | writer config page | 64 B |
|  | Разом | **41,632 B/core** |

Як і в resident, цей footprint не входить у signal budget.

## 7. DRAM і NoC traffic

### 7.1. Resident

Основні transfers:

1. input DRAM -> L1 `A/B` під час pad/split;
2. intermediate routes: L1 shard -> CB -> L1 shard;
3. remote L1 reads лише коли потрібні дані належать іншому shard;
4. terminal even/odd scale -> final DRAM;
5. reader і writer окремо читають по одній 64-byte config page на core для кожного
   executable route.

Resident не робить DRAM round-trip після кожного lifting step.

### 7.2. Cone

Основні transfers:

1. input DRAM -> local `A/B` для кожного dependency cone;
2. intermediate routes повністю local L1;
3. terminal even/odd intervals -> final DRAM; один із scale routes може бути fused з останнім
   predict/update;
4. 64-byte chunk config на chunk;
5. 64-byte route config на кожну пару `(chunk, executable route)` і на reader, і на writer.

Final elements записуються один раз, але initial halo може читатися кількома chunks.

## 8. Runtime config і telemetry

Route config page містить:

```text
type
source_addr, source_length
base_addr, base_length
output_addr, output_length
source_offset, base_offset
source_left_pad
output_offset       # використовується Cone final DRAM write
group_count         # використовується Cone
flags               # kRouteFlagFinalDram та майбутні route-level властивості
```

`lwt --benchmark` друкує scheduler telemetry:

| Поле | Resident | Cone |
| --- | --- | --- |
| `lwt_memory_mode` | `resident` | `cone` |
| `lwt_max_group_count` | Максимальний full-stream group count | Final group count |
| `lwt_groups_per_shard` | Groups у shard | 0 |
| `lwt_active_core_count` | Active shard owners | Chunk workers |
| `lwt_shard_elements` | Capacity slot/core | Те саме, що `workspace_elements` |
| `lwt_chunk_count` | 0 | Кількість final chunks |
| `lwt_groups_per_chunk` | 0 | Maximum groups/chunk |
| `lwt_workspace_elements` | 0 | Local slot capacity/core |
| `lwt_max_dependency_overhead` | 0 | Максимум серед chunks |
| `lwt_terminal_scale_fused` | 0 | `1`, якщо planner використав fused kernel variant |
| `lwt_tile_native_workspace` | 0 | `1` для persistent `32x16` workspace, `0` для row-major |
| `lwt_zero_work_cores_per_route` | Zero-work cores для кожного route | Порожньо |

## 9. Запуск і налаштування

### 9.0. Збірка

Для змін у `tt-wavelet` використовуйте локальну target-збірку:

```bash
./update.sh Release lwt
source ./scripts/set_env.sh
./build/lwt --help
```

`./build.sh` тут не потрібний: він перебудовує весь `tt-metal` і понад тисячу targets.

### 9.1. Прямий benchmark

```bash
source ./scripts/set_env.sh

./build/lwt \
  --benchmark \
  --memory-mode resident \
  --repeats 3 \
  --warmup-runs 1 \
  --length 1000000 \
  db7
```

Для dependency cone достатньо замінити:

```bash
--memory-mode cone
```

Якщо прапорець не вказано, current default — `cone`.

### 9.2. Sweep через `compare_timings.py`

Resident:

```bash
python3 compare_timings.py \
  --backend tt-wavelet \
  --tt-memory-mode resident \
  --wavelets db7 \
  --length-start 100000 \
  --length-stop 1000000 \
  --length-step 10000 \
  --tt-repeats 3 \
  --tt-warmup-runs 1
```

Cone:

```bash
python3 compare_timings.py \
  --backend tt-wavelet \
  --tt-memory-mode cone \
  --wavelets db7 \
  --length-start 100000 \
  --length-stop 1000000 \
  --length-step 10000 \
  --tt-repeats 3 \
  --tt-warmup-runs 1
```

`lwt_memory_mode` входить у CSV row key, тому обидва режими можуть зберігатися в одному
CSV. Не використовуйте `--overwrite` на другому запуску, якщо хочете зберегти перший.

Benchmark timer включає device execution:

- у resident — pad/split program і lifting program;
- у cone — cone initialization та всі lifting routes;
- в обох режимах — queue `Finish()`.

Створення buffers/programs, запис config pages та final coefficient readback перебувають
поза timed interval. У benchmark mode вмикається TT-Metal program cache.

### 9.3. Environment variables

| Variable | Default | Дія |
| --- | ---: | --- |
| `TT_WAVELET_LWT_MAX_CORES` | усі hardware worker cores | Обмежує resident shard owners і cone workers |
| `TT_WAVELET_L1_SIGNAL_BUDGET_BYTES` | `786432` | Максимум bytes/core лише для трьох signal/workspace slots |
| `TT_WAVELET_LWT_FUSE_TERMINAL_SCALE` | `auto` | Cone: `auto`, примусово вимкнути `0` або ввімкнути `1` terminal-scale fusion |
| `TT_WAVELET_LWT_CONE_NOC_LOCAL_WRITE` | `1` | Cone: `1` використовує 64-byte local NoC writes; `0` повертає scalar L1 writer для діагностики/A-B |
| `TT_WAVELET_LWT_CONE_WORKSPACE_LAYOUT` | `auto` | Cone: `auto`, `row-major` або `tile-native`; вибір compile-time variant при створенні executable |

Приклад:

```bash
export TT_WAVELET_LWT_MAX_CORES=40
export TT_WAVELET_L1_SIGNAL_BUDGET_BYTES=$((640 * 1024))
export TT_WAVELET_LWT_FUSE_TERMINAL_SCALE=auto
export TT_WAVELET_LWT_CONE_NOC_LOCAL_WRITE=1
export TT_WAVELET_LWT_CONE_WORKSPACE_LAYOUT=auto
```

Не задавайте signal budget близьким до повного фізичного L1. На Wormhole L1 спільний для
buffers, CBs, kernel binaries, stack і runtime infrastructure. Код не запитує реальний
free-L1 після всіх allocations; відповідальність за безпечне значення лишається на caller.

У pad/split runtime також існує `TT_WAVELET_PAD_SPLIT_CORES`, але current resident path
пише у вже sharded L1 buffers і бере owner cores з їх distribution spec. Тому цей variable
не змінює core count для описаного тут `ResidentSharded`; він стосується unsharded
pad/split output path.

## 10. Як обирати режим

Використовуйте `resident`, коли:

- три повні streams гарантовано вкладаються у signal budget;
- dependency overlap для схеми великий;
- довгі routes добре завантажують більшість cores;
- один pad/split pass вигідніший за повторне читання chunk halos.

Використовуйте `cone`, коли:

- повний resident signal не влазить у L1;
- global route barriers або zero-work cores домінують;
- важлива повна локальність intermediate streams;
- схема має невеликий dependency halo відносно chunk body.

Правильний вибір є емпіричним. Порівнюйте `tt_wavelet_median_s`, `p10/p90`,
`lwt_active_core_count`, `lwt_zero_work_cores_per_route` і
`lwt_max_dependency_overhead` для однакових signal/scheme/repeats.

### 10.1. Коли tile-native вигідний

Tile-native не є універсально швидшим. Він прибирає intermediate scatter writes і дає
zero-copy base pages, але shifted source windows все одно треба переставляти з block-major
workspace у stencil Dst geometry. Тому `auto` лишає `db7` у row-major, а `bior3.9` обирає
tile-native.

Локальний N150 A/B на довжині 5,000,000 (`warmup=3`, `repeats=10`) показав:

| Scheme | Auto layout | Auto | Примусово інший layout | Різниця |
| --- | --- | ---: | ---: | ---: |
| `db7` | row-major | 21.767 ms | tile-native 22.165 ms | auto приблизно 1.8% швидше |
| `bior3.9` | tile-native | 13.310 ms | row-major 13.861 ms | auto приблизно 4.1% швидше |

Повторний N150 L A/B 2026-07-19 (`warmup=5`, `repeats=20`, 64 active cores) підтвердив
forward policy на поточній revision:

| Scheme | N | row-major median | tile-native median | auto median/layout |
| --- | ---: | ---: | ---: | ---: |
| `db7` | 1,000,000 | 6.676 ms | **6.475 ms** | 6.607 ms / row-major |
| `db7` | 5,000,000 | 23.604 ms | **23.545 ms** | 23.617 ms / row-major |
| `db7` | 8,000,000 | **36.148 ms** | 36.251 ms | 36.119 ms / row-major |
| `bior3.9` | 1,000,000 | 4.399 ms | **4.146 ms** | 4.201 ms / tile-native |
| `bior3.9` | 5,000,000 | 15.554 ms | **14.802 ms** | 14.795 ms / tile-native |
| `bior3.9` | 8,000,000 | 23.974 ms | **22.702 ms** | 22.715 ms / tile-native |

Для `db7` різниця між layouts мала і змінює знак із довжиною; current auto row-major є
консервативним. Для `bior3.9` tile-native стабільно швидший приблизно на 5%. Ці дані
стосуються forward LWT: inverse має окремий architecture-dependent result у
[ILWT_1D.md](ILWT_1D.md).

Це не performance guarantee: для коротких signals launch/runtime noise близький до різниці
між layouts. Для A/B тесту задайте environment variable явно й використовуйте однакові
warmup/repeats.

Correctness перевірено примусовим запуском обох layouts для всіх 106 scheme JSON: довжини
збіглись, `max_abs(row_major, tile_native) = 0` для надрукованих FP32 coefficients. Окрема
перевірка проти PyWavelets на поточному 20-element test signal дала:

| Absolute tolerance | Passed | Failed |
| ---: | ---: | ---: |
| `1e-5` | 49 | 57 |
| `1e-2` | 70 | 36 |

Усі 36 schemes, які перевищують `1e-2`, так само біт-у-біт збігаються з resident backend.
Отже narrow-tile layout, ConeStreamed scheduling і remap не додають числової похибки.

### 10.2. Чому lifting result може відрізнятися від PyWavelets

Ці відхилення є очікуваною властивістю деяких довгих lifting factorizations у FP32, а не
ознакою layout corruption. TT-Wavelet виконує послідовність factorized predict/update
steps. Кожен step округлює проміжний FP32 результат, а наступні steps використовують уже
округлене значення. У high-order schemes із багатьма steps або великими/alternating
lifting coefficients попередня похибка може підсилюватися по всьому ланцюжку.

PyWavelets обчислює той самий wavelet через інший filter-bank execution path, тому порядок
операцій і точки округлення не збігаються з lifting chain. Біт-у-біт збіг із PyWavelets для
FP32 lifting не є коректною універсальною вимогою; tolerance має визначатися numerical
contract конкретного application і властивостями scheme.

Водночас це не означає, що будь-яка похибка автоматично прийнятна. Наведені counts залежать
від test signal, довжини та absolute tolerance. Schemes із великим amplification повинні
мати окремо визначений accuracy envelope, перевірятися на representative/random inputs або
бути виключені з production support matrix. Для перевірки layout/backend regression
правильним reference залишається поточний FP32 resident result: `row-major`, `tile-native`,
cone і resident мають відтворювати однаковий порядок arithmetic і однаковий результат.

## 11. Поточні обмеження

1. Реалізовані one-level forward LWT і ConeStreamed ILWT. Автоматична multi-level
   decomposition не виконується.
2. Підтримується лише FP32.
3. Cone forward та ILWT підтримують `symmetric`, `zero`, `constant`, `periodic`,
   `antisymmetric`, `smooth`, `reflect`, `antireflect`. Resident forward — лише `symmetric`;
   `periodization` ще не реалізований. Він не є alias для `periodic`: він змінює canonical
   coefficient length і потребує окремої geometry роботи.
4. Один predict/update step може мати максимум 17 coefficients. У поточних 106 schemes
   фактичний максимум — 9.
5. Scheme повинна закінчуватися однією `ScaleEven` і однією `ScaleOdd`; інших scale routes
   до них бути не може.
6. `Swap` є тільки metadata operation і не має coefficients.
7. Input і більшість route fields передаються як `uint32_t`. Cone консервативно вимагає
   `padded_length <= INT32_MAX`; symmetric boundary mapping використовує signed logical
   indexing. Resident host check зараз дозволяє `uint32_t`, але той самий device indexing
   робить практично безпечним обмеження `INT32_MAX` і для resident.
8. Resident не має автоматичного fallback у cone.
9. Cone не може дробити chunk менше одного 1536-element group.
10. Cone config storage росте як `chunks * executable_routes * 64 bytes`; дуже велика
    кількість chunks збільшує config traffic і host planning cost.
11. Resident global barrier включає також cores із нульовою роботою на коротшому route.
12. L1 budget враховує тільки три slots, а не CB/runtime allocations.
13. Logical tails доповнюються до 32-element sticks. Дані за logical length не є частиною
    результату, навіть якщо фізично присутні у buffer.
14. Cone compute native `32x16`, але це не означає zero-copy для кожного read: shifted
    source/base intervals потребують logical-to-physical remap.
15. Деякі high-order lifting factorizations чисельно менш стабільні у FP32 через накопичення
    й підсилення rounding error між predict/update steps. Це scheme/factorization property,
    а не layout regression: row-major, tile-native, cone і resident відтворюють однаковий
    FP32 результат.

## 12. Головні invariants для змін у kernels

При модифікації backend потрібно зберігати такі умови:

- source і base не alias-ять output поточного predict/update route;
- output group належить рівно одному writer;
- `noc_async_write_barrier()` завершується до повторного використання output CB pages;
- resident наступний route стартує лише після global visibility попереднього;
- cone chunk містить усі backward-propagated dependencies;
- cone cores ніколи не залежать від intermediate workspace іншого core;
- offsets і lengths лишаються у logical elements; physical address обчислюється відповідно
  до row-major stick або tile-native `block * 512 + row * 16 + lane` layout;
- final DRAM пишуть лише routes з `kRouteFlagFinalDram`; це terminal scale або fused
  останній predict/update.

Ці invariants є важливішими за конкретний розмір chunk або кількість cores: їх порушення
може дати тихі числові помилки, навіть якщо програма не зависає і buffers формально
вміщуються у L1.
