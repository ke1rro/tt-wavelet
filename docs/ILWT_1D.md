# ConeStreamed ILWT 1D

## Статус

1D ILWT реалізовано поверх наявного FP32 ConeStreamed/narrow-tile backend. Перевірка виконана
2026-07-18 на Blackhole P150b (`TT-KMD 2.9.0`), project revision `1f6eae385f0e` і TT-Metal revision
`f87c34a93ee4` з локальними незакоміченими змінами. Wormhole hardware у цій сесії недоступний,
тому Wormhole equivalence ще треба зняти на окремій машині.

Device dataflow:

```text
canonical approximation/detail DRAM
        -> per-chunk dual coefficient reader
        -> local A / B / Scratch
        -> lazy reciprocal-scale fusion + reversed P/U routes in FP32 SFPU
        -> final-route/interleave fusion + crop
        -> reconstructed signal DRAM
```

Між inverse routes немає DRAM loopback. Metadata `swap` не запускає compute route. Обидва
початкові reciprocal-scale routes fused у predict/update chain, а кінцевий writer може споживати
три output-CB pages останнього route без його проміжної materialization у L1. Незмінений final
stream читається з resident workspace, після чого writer interleave-ить/crop-ить результат у
128-byte DRAM sticks.

ConeStreamed ILWT приймає ті самі boundary modes, що й forward:
`symmetric`, `zero`, `constant`, `periodic`, `antisymmetric`, `smooth`, `reflect`,
`antireflect`.
Boundary samples не обчислюються вдруге в inverse kernel. Mode визначає, як були отримані
canonical coefficients; inverse відновлює потрібні padded polyphase intervals і для кожного
mode однаково crop-ить original interval `[T-1, T-1+N)`. Через це mode не додає branch або
арифметику в ILWT device hot path. `reflect` і `antireflect` вимагають `N > 1`.

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

## Реалізовані fusion

Inverse scale fusion не перемножує lifting coefficients зі scale, бо це змінило б FP32 rounding.
Замість цього compute compile-time відстежує `needs_scale` і scalar окремо для поточних even/odd
streams. `swap` переставляє metadata разом зі stream, а перший predict/update, який замінює stream,
виконує його reciprocal multiply у SFPU registers перед тим самим MAD stencil. Тільки після цього
lazy-scale стан stream скидається. Обидва окремі scale routes, їх L1 outputs і наступні L1 reads
прибрані. Compile-time assertion не дозволяє завершити схему з невикористаним non-identity scale.

Final route/interleave fusion позначається route flag. Writer не записує final P/U output у
workspace, а interleave-ить updated parity прямо з трьох narrow output pages разом з другим
resident stream. Останній неповний 3072-element group підтримує також випадок, коли updated parity
не має другого CB group.

Виміряна default policy:

- inverse scale fusion — `on`;
- final interleave fusion — `auto`: `on` для tile-native workspace і `off` для row-major;
- примусовий A/B: `TT_WAVELET_ILWT_FUSE_INVERSE_SCALE=0|1` і
  `TT_WAVELET_ILWT_FUSE_FINAL_INTERLEAVE=auto|0|1`.

Telemetry і `compare_timings.py` CSV містять `lwt_inverse_scale_fused` та
`lwt_inverse_final_interleave_fused`.

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
    original_length,
    boundary_mode);
ttwv::prepare_cone_streamed_ilwt(command_queue, executable);
ttwv::execute_cone_streamed_ilwt(mesh_device, command_queue, executable);
```

`original_length` обов'язковий. Canonical approximation/detail buffers мають однакову довжину.
`boundary_mode` має збігатися з mode forward transform, що створив coefficients; default —
`BoundaryMode::kSymmetric`.

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
round-trip error `7.15e-7`.

Після обох fusion runtime/JIT sweep пройшов 106/106 production inverse schemes без failures;
forced-on telemetry підтвердила обидві fusion для 106/106. Окремий fused/unfused Blackhole A/B
для `db7` дав однаковий max round-trip error на довжинах
`1,2,31,32,33,1535,1536,1537,3071,3072,3073,6143,6144,6145`. Для `db7` і `bior3.9` результати
також збіглися між fused/unfused у forced `row-major` та `tile-native` на 3073 і 6145. Synthetic
K=17 пройшов odd/even 64/65 з нульовою round-trip похибкою.

Boundary-mode extension перевірено окремо canonical PyWavelets coefficients: 168 cases для
`db1`, `db7`, `bior3.9`, семи раніше підтримуваних modes і довжин
`2,3,17,32,33,3071,3072,3073`. Максимальна absolute ILWT error становила
`3.29005435e-05`. Ще 42 `db7` cases дали bitwise-identical output у `row-major`,
`tile-native`, `auto`. Runtime/JIT sweep нових чотирьох modes пройшов 424/424 cases
(`106 schemes * 4 modes`) на `N=33`.

Для `reflect` додатково пройдено 36 PyWavelets cases для `db1`, `db2`, `db7`, `bior3.9`
на `N=2,3,17,31,32,33,3071,3072,3073`; max absolute ILWT error — `3.29005435e-05`.
Ще 6 forced-layout `db7` cases були bitwise identical, а all-scheme runtime/JIT sweep
пройшов 106/106 на `N=33`.

Команди:

```bash
.venv/bin/python scripts/validate_ilwt_geometry.py
.venv/bin/python scripts/validate_ilwt.py --layouts row-major tile-native auto
.venv/bin/python scripts/validate_ilwt_stability.py --length 33
```

## Blackhole device-only performance

Pre-fusion reference: `auto` workspace, 5 measured repetitions, 2 warmups. Forward coefficient
generation, program construction, config upload і readback не входять до timed interval.

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

Fusion A/B на тому самому Blackhole, N=5,000,000, 20 measured repetitions і 5 warmups:

| Scheme/layout | Mode | median, ms | min, ms | p10, ms | p90, ms |
|---|---|---:|---:|---:|---:|
| `db7` / row-major | none | 8.226 | 7.980 | 8.123 | 8.320 |
| `db7` / row-major | interleave only | 8.644 | 8.550 | 8.559 | 8.700 |
| `db7` / row-major | scale only (auto default) | 7.695 | 7.490 | 7.548 | 7.781 |
| `db7` / row-major | both | 8.256 | 8.145 | 8.203 | 8.308 |
| `bior3.9` / tile-native | none | 7.463 | 7.034 | 7.059 | 7.586 |
| `bior3.9` / tile-native | interleave only | 6.906 | 6.745 | 6.827 | 7.052 |
| `bior3.9` / tile-native | scale only | 7.370 | 7.231 | 7.254 | 7.436 |
| `bior3.9` / tile-native | both (auto default) | 6.819 | 6.725 | 6.793 | 6.866 |

Виміряна auto policy дає приблизно 6.5% для `db7` і 8.6% для `bior3.9` проти повністю unfused
baseline. Окремі Wormhole timings все ще потрібні з ідентичним timing boundary.

## Відомі обмеження і наступні кроки

- Вісім non-periodization boundary modes і FP32 storage/SFPU arithmetic. `periodization`
  ще не реалізований і потребує іншої coefficient geometry.
- Реалізовано 1D single-level ILWT; multi-level і separable 2D ще не додані.
- 106-scheme sweep доводить runtime stability, а не однакову малу numerical error для
  ill-conditioned high-order FP32 factorizations. Високопорядкові FP32 factorization errors
  відокремлені від ILWT geometry regression.
- Потрібне Wormhole hardware A/B для architecture equivalence; compile-time Wormhole path
  збережений, але на Blackhole server його неможливо апаратно перевірити.
