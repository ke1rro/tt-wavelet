# ConeStreamed ILWT 1D

## Статус

1D ILWT реалізовано поверх наявного FP32 ConeStreamed/narrow-tile backend. Перевірка виконана
2026-07-17 на Blackhole P150b (`TT-KMD 2.9.0`), project revision `ada4f55` і TT-Metal revision
`f87c34a93ee` з локальними незакоміченими змінами. Wormhole hardware у цій сесії недоступний,
тому Wormhole equivalence ще треба зняти на окремій машині.

Device dataflow:

```text
canonical approximation/detail DRAM
        -> per-chunk dual coefficient reader
        -> local A / B / Scratch
        -> reciprocal scale + reversed P/U routes in FP32 SFPU
        -> crop padded even/odd streams + interleave
        -> reconstructed signal DRAM
```

Між inverse routes немає DRAM loopback. Metadata `swap` не запускає compute route. Кінцевий
writer зараз спершу materialize-ить останні even/odd intervals у L1, потім interleave-ить їх у
128-byte DRAM sticks. Це коректний Phase 2 варіант; fusion останнього P/U route з interleave ще не
реалізований.

## Arithmetic і geometry

Generator створює поруч із кожною forward scheme її compile-time inverse scheme:

- кроки виконуються у зворотному порядку;
- `predict`/`update` зберігають type, shift і порядок coefficients, але кожен FP32 sign bit
  інвертується;
- scale замінюється на один FP32-rounded reciprocal;
- `swap` залишається metadata-only;
- нульовий scale відхиляється generator-ом.

Inverse arithmetic береться з reversed scheme, але geometry завжди береться з forward route
trace. Для output chunk `[a,b)` і forward left padding `P=T-1` початкові split targets:

```text
E0 = [ceil((a+P)/2), ceil((b+P)/2))
O0 = [floor((a+P)/2), floor((b+P)/2))
```

Ці intervals проходять вперед через original forward trace, щоб отримати мінімальні required
final internal streams. Перехід у canonical coefficient index:

```text
canonical_index = internal_index + final_stream_shift - T/2
```

Після цього routes будуються у зворотному порядку. Для forward P/U route з `source_offset=s`,
`base_offset=b0`, coefficient count `K` та target old base interval `R` inverse route читає:

```text
target = R - b0
source_required = [target.begin+s, target.end+s+K-1)
base_required = target
```

Це не еквівалентно запуску forward planner на synthetically reversed scheme: такий planner не
знає, які samples були відкинуті forward valid cropping.

Natural ILWT group дорівнює 3072 original samples: приблизно 1536 even і 1536 odd values. Chunk
config зберігає canonical A/D intervals, final local stream addresses/offsets і output interval.
Route config використовує той самий 128-byte protocol, що й forward ConeStreamed.

## Blackhole SFPI lane-mask correction

Під час ILWT boundary validation було знайдено окрему forward regression. Blackhole halo injection
використовувала `SFPSETCC(LTILEID, EQ0)` з припущенням, що `LTILEID` повторює `0..7` у кожному
subvector. Насправді `LReg[15]` містить `lane*2`, тобто `0,2,...,62`; EQ0 вибирає лише physical
lane 0. Це визначено в
`tt-isa-documentation/WormholeB0/TensixTile/TensixCoprocessor/LReg.md`, а Blackhole використовує
ту саму register model.

Blackhole predicate тепер обчислюється як:

```text
(LREG15 & 0xF) == 0
```

Він точно вибирає lanes 0, 8, 16 і 24 перед masked halo move. Після виправлення `db7`, N=255,
змінився з max detail error приблизно `85.47` до `2.99e-5` проти PyWavelets. Wormhole erratum
fast path не змінювався.

## API і CLI

Public C++ API знаходиться у `tt_wavelet/include/lifting/device.hpp`:

```cpp
auto executable = ttwv::create_cone_streamed_ilwt_executable<Scheme>(
    kernel_root,
    mesh_device,
    approximation_buffer,
    detail_buffer,
    coefficient_length,
    original_length);
ttwv::prepare_cone_streamed_ilwt(command_queue, executable);
ttwv::execute_cone_streamed_ilwt(mesh_device, command_queue, executable);
```

`original_length` обов'язковий. Canonical approximation/detail buffers мають однакову довжину.

Внутрішній TT round-trip:

```bash
source scripts/set_env.sh
build/lwt --inverse --length 3073 db7
```

ILWT із зовнішніх canonical coefficient files:

```bash
build/lwt --inverse \
  --original-length 3073 \
  --approximation-file approximation.txt \
  --detail-file detail.txt \
  db7
```

Synthetic maximum-capacity SFPI test доступний лише як CLI acceptance scheme:

```bash
build/lwt --inverse --length 3073 synthetic-k17
```

Він не входить до production registry 106 wavelets.

## Correctness results

Незалежна Python geometry model перевірила 106 schemes, 59,042 chunk cases, довжини `1..129`,
`255/256/257`, `1023/1024/1025`, `3071/3072/3073` і `10000`. Найбільший dependency overhead
очікувано виник для `coif17`, `N=1`: 102 coefficient values на один output sample.

PyWavelets interoperability використовувала FP32-cast canonical PyWavelets coefficients як прямий
TT ILWT input. Результати для bounded sine/cosine signal:

| Scheme | Lengths | max abs TT ILWT vs PyWavelets ILWT |
|---|---:|---:|
| `db1` | 17, 32, 33, 3071, 3072, 3073 | `1.17e-7` |
| `db7` | 17, 32, 33, 3071, 3072, 3073 | `3.29e-5` |
| `bior3.9` (shipped K=9) | 17, 32, 33, 3071, 3072, 3073 | `3.62e-7` |

Для всіх цих cases `row-major`, `tile-native` та `auto` outputs були bitwise identical
(`max_abs_between_layouts = 0`). Додатково N=1,2,3,4,5,15,16,17,18 пройшли для всіх трьох
representative schemes. Synthetic K=17 пройшов N=33,255,3073 у row-major і tile-native з max
round-trip error `7.15e-7`. Runtime-stability sweep: 106/106 schemes, failures 0.

Команди:

```bash
.venv/bin/python scripts/validate_ilwt_geometry.py
.venv/bin/python scripts/validate_ilwt.py --layouts row-major tile-native auto
.venv/bin/python scripts/validate_ilwt_stability.py --length 33
```

## Blackhole device-only performance

`auto` workspace, 5 measured repetitions, 2 warmups. Forward coefficient generation, program
construction, config upload і readback не входять до timed interval.

Для однакового sweep LWT та ILWT проти відповідних PyWavelets `dwt`/`idwt`:

```bash
.venv/bin/python compare_timings.py \
  --transform both \
  --backend both \
  --tt-memory-mode cone \
  --wavelets db7 bior3.9 \
  --lengths 100000 1000000 5000000 8000000 \
  --pywt-repeats 10 \
  --tt-repeats 10 \
  --tt-warmup-runs 3 \
  --csv lwt_ilwt_timings.csv \
  --overwrite
```

CSV column `transform` розділяє `lwt` та `ilwt`, тому результати не перезаписують один одного.
Для ILWT обидва backends готують approximation/detail coefficients поза timed interval.

| Scheme | N | median, ms | min, ms | p10, ms | p90, ms | active cores | workspace/core | layout |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `db7` | 100,000 | 1.583 | 1.416 | 1.474 | 1.816 | 33 | 1,568 FP32 | row-major |
| `db7` | 1,000,000 | 2.753 | 2.723 | 2.730 | 2.800 | 110 | 4,640 FP32 | row-major |
| `db7` | 5,000,000 | 8.167 | 8.123 | 8.138 | 8.215 | 110 | 23,072 FP32 | row-major |
| `db7` | 8,000,000 | 12.028 | 11.967 | 11.986 | 12.069 | 110 | 36,896 FP32 | row-major |
| `bior3.9` | 1,000,000 | 2.404 | 2.348 | 2.367 | 2.561 | 110 | 6,144 FP32 | tile-native |
| `bior3.9` | 5,000,000 | 7.356 | 7.307 | 7.324 | 7.390 | 110 | 24,576 FP32 | tile-native |

Це перший ILWT backend, тому old/new ILWT speedup відсутній. Наступні performance кроки мають
порівнювати: (1) materialized-final-stream baseline, (2) fused final interleave, (3) inverse scale
fusion. Також потрібні окремі Wormhole timings з ідентичним timing boundary.

## Відомі обмеження і наступні кроки

- Тільки symmetric boundary mode і FP32 storage/SFPU arithmetic.
- Реалізовано 1D single-level ILWT; multi-level і separable 2D ще не додані.
- Final interleave ще не fused з останнім inverse route.
- Inverse terminal scales не fused; це свідомо відкладено до профілювання.
- 106-scheme sweep доводить runtime stability, а не однакову малу numerical error для
  ill-conditioned high-order FP32 factorizations.
- Потрібне Wormhole hardware A/B для architecture equivalence; compile-time Wormhole path
  збережений, але на Blackhole server його неможливо апаратно перевірити.
