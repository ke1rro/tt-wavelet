



## Мій висновок

Я б **не будував dynamic cache**, де кожен `InputStream::pop()` вирішує: читати з L1 чи DRAM. Це додасть багато станів, coherence, CB ownership і runtime branching.

Краще зробити один backend із двома режимами:

```text
1. ResidentSharded
   увесь even/odd signal живе в L1 shards;
   між lifting steps немає DRAM.

2. ConeStreamed
   якщо весь signal не влазить:
   беремо великий chunk фінального output,
   backward planner знаходить його dependency cone,
   завантажуємо cone з DRAM,
   виконуємо всі lifting steps у L1,
   записуємо лише фінальний chunk.
```

Тобто DRAM/L1 decision робиться **один раз на host-side planner**, а не всередині кожного читання.

---

# 1. Що зараз відбувається

Поточна схема фактично така:

```text
DRAM source/base
    ↓
lwt_reader
    ↓
CB tiles
    ↓
SFPU stencil
    ↓
CB output
    ↓
lwt_writer
    ↓
DRAM ping/pong
    ↓
global route barrier
    ↓
наступний lifting step знов читає DRAM
```

`create_signal_mesh_buffer()` створює `even.ping`, `even.pong`, `odd.ping`, `odd.pong` саме як `BufferType::DRAM`. Кожен route містить окремі `source_addr`, `base_addr`, `output_addr`, а writer після кожного lifting step матеріалізує весь output назад у DRAM. fileciteturn2file1

Сам SFPU kernel уже має нормальну структуру:

```text
output = base + stencil(source)
```

Тому `horizontal_stencil_sfpi.h` і `lwt_compute.cpp` переписувати з нуля не потрібно. Основна проблема — **storage/runtime architecture**, а не stencil math.

---

# 2. Не робити чотири L1 ping/pong buffers

Очевидна заміна:

```text
even_ping DRAM → even_ping L1 sharded
even_pong DRAM → even_pong L1 sharded
odd_ping  DRAM → odd_ping  L1 sharded
odd_pong  DRAM → odd_pong  L1 sharded
```

вже прибере DRAM loopback. Але це все ще чотири повнорозмірні buffers.

Натомість я рекомендую **три універсальні storage slots**:

```text
slot A
slot B
slot S — scratch/free
```

Початковий стан:

```text
active_even = A
active_odd  = B
free        = S
```

Для predict:

```text
S = odd + P(even)

active_even залишається A
active_odd  стає S
старий odd-buffer B стає free
```

Для update:

```text
B = even + U(odd)

active_even стає B
active_odd  залишається S
старий even-buffer A стає free
```

У загальному вигляді:

```cpp
route.source = active_source;
route.base   = active_target;
route.output = free_slot;

free_slot    = active_target;
active_target = route.output;
```

Для `Swap` лише міняються metadata:

```cpp
std::swap(active_even, active_odd);
```

## Чому три buffers краще

Поточна схема має:

```text
4 × приблизно N/2 FP32 = 2N FP32 elements = 8N bytes
```

Три slots:

```text
3 × приблизно N/2 FP32 = 1.5N FP32 elements = 6N bytes
```

Для сигналу `N = 10,000,000`:

```text
4-buffer scheme ≈ 80 MB aggregate L1
3-buffer scheme ≈ 60 MB aggregate L1
```

При 80 cores це приблизно:

```text
60 MB / 80 ≈ 750 KB signal storage per core
```

Поточні CB займають приблизно ще 50 KB/core. На Wormhole є 1.5 MB SRAM на Tensix, хоча частина зарезервована runtime/kernel infrastructure. Тому 10M FP32 signal із трьома slots має значно реалістичніший шанс повністю поміститися в aggregate L1. SRAM і sharded allocation якраз призначені для зберігання проміжних tensors без DRAM round-trip. fileciteturn1file10

---

# 3. Нова структура planner

Зараз `StreamRef` містить:

```cpp
struct StreamRef {
    LogicalStream family;
    StreamSlot slot;  // ping/pong
};
```

і `append_forward_route()` постійно toggle-ить ping/pong. fileciteturn1file0turn1file1

Я б замінив це на:

```cpp
enum class StorageSlot : uint8_t {
    kA = 0,
    kB = 1,
    kScratch = 2,
};

struct StreamRef {
    StorageSlot slot;
};

struct LiftingActiveStreams {
    StreamRef even{StorageSlot::kA};
    StreamRef odd{StorageSlot::kB};
    StorageSlot free{StorageSlot::kScratch};
};
```

Route геометрія залишається майже без змін:

```cpp
struct LiftingStepRoute {
    StepType type;

    StorageSlot source;
    StorageSlot base;
    StorageSlot output;

    size_t source_length;
    size_t base_length;
    size_t source_offset;
    size_t base_offset;
    uint32_t source_left_pad;
    size_t output_length;
};
```

Твій `compute_step_geometry()` уже правильно обчислює:

```text
output shift
output length
source offset
base offset
```

Його треба зберегти. fileciteturn1file8

---

# 4. ResidentSharded mode

## Memory layout

Кожен із трьох slots — це:

```text
FP32
row-major
stick width = 32
height-sharded
BufferType::L1
```

Логічно tensor можна представити як:

```text
[num_sticks, 32]
```

і shard-ити по `num_sticks`.

Бажано зробити shard capacity кратним:

```cpp
kLwtGroupOutputElements = 32 * 3 * 16 = 1536 elements
```

тобто:

```text
1536 elements = 48 sticks
```

Тоді один output group ніколи не буде розділений між двома cores.

```cpp
shard_elements =
    round_up(ceil(max_stream_length / core_count), 1536);
```

Перевірка capacity:

```cpp
signal_bytes_per_core =
    3 * shard_elements * sizeof(float);

fits =
    signal_bytes_per_core +
    fixed_cb_bytes +
    l1_safety_margin
    <= usable_l1_bytes_per_core;
```

Не потрібно жорстко припускати, що доступні всі 1.5 MB. Краще мати configurable safety budget, наприклад:

```cpp
TT_WAVELET_L1_SIGNAL_BUDGET_BYTES
```

або визначати conservative budget на host.

---

## Pad/split

Зараз `pad_split` записує even/odd у DRAM buffers.

У resident mode він повинен одразу записувати:

```text
even → slot A, L1 sharded
odd  → slot B, L1 sharded
```

`pad_split_1d_writer.cpp` концептуально не потрібно міняти: `TensorAccessor` працює і для DRAM, і для sharded L1, якщо правильний layout переданий compile-time. Sharded memory також доступна через `TensorAccessor` на page granularity. fileciteturn0file4

---

## LWT reader

Reader отримує три L1-sharded accessors:

```cpp
slot_a
slot_b
slot_scratch
```

На кожному route він вибирає:

```cpp
const auto& src  = slots[route.source];
const auto& base = slots[route.base];
```

Поточний `StickReadCache` можна лишити.

У більшості випадків source/base sticks будуть локальними. Лише stencil halo біля shard boundary потрапить на сусідній core через NoC.

Тобто замість:

```text
кожен sample через NoC із DRAM bank
```

буде:

```text
основний body: local core L1
boundary halo: сусідній core L1
```

---

## Compute

`lwt_compute.cpp` можна практично не змінювати:

```cpp
run_predict_update_step(
    cb_input0,
    cb_input1,
    cb_base,
    cb_output,
    Step::coeff_bits,
    group_count);
```

Твоя LWT abstraction уже зведена до generic stencil:

```text
predict: odd  = odd  + P(even)
update:  even = even + U(odd)
scale:   stream *= scalar
swap:    metadata swap
```

Це саме правильна boundary для нового memory backend. fileciteturn1file6

---

## Writer

Writer записує output у sharded L1 slot:

```cpp
dst = slots[route.output];
```

Зберігається compact layout:

```text
output[0 ... output_length)
```

Тому нинішні 16-element block writes залишаються aligned, а route geometry не треба ускладнювати фізичними origins.

Після route:

```text
global route barrier
```

залишається потрібним, тому що наступний step може читати щойно записаний stream, включно з remote halo іншого core.

Важливо: ми прибираємо **DRAM materialization**, але не порядок lifting steps. Lifting steps послідовні за визначенням; всередині одного step усі output samples паралельні. Це також відповідає стандартній in-place/parallel властивості lifting scheme. fileciteturn1file15

---

# 5. Не використовувати поточні `InputStream` / `OutputStream`

Я б їх не розвивав як основу нового backend.

В активному LWT path вони взагалі не використовуються. Reader працює через `StickReadCache`, а writer — напряму через tiled output CB.

Також у `InputStream` зараз видно незавершену source-state abstraction:

```cpp
reset(...) {
    _source = new_source;
}
```

але `_next_blob()` перевіряє:

```cpp
if (_read_source == kSrcDRAM)
```

тобто `_source` та `_read_source` не узгоджені. L1 path ще й передбачає, що якийсь writer уже залишив потрібний blob у тому самому CB, що сильно зв’язує producer/consumer state. fileciteturn2file0

Dynamic blob cache створить додаткові проблеми:

- хто володіє CB page;
- чи актуальна L1 copy;
- чи потрібно spill-ити dirty page;
- коли DRAM backing стає authoritative;
- як працює route barrier;
- що робити, якщо source і output посилаються на той самий cached blob.

Для LWT це зайва складність.

---

# 6. Що робити, коли весь сигнал не влазить

Тут я б **не повертався до route-by-route DRAM spilling**.

Інакше для великого сигналу знов отримаємо:

```text
step 0: DRAM → L1 → DRAM
step 1: DRAM → L1 → DRAM
step 2: DRAM → L1 → DRAM
...
```

Замість цього потрібен `ConeStreamed` mode.

## Ідея

Беремо chunk фінального output:

```text
final output interval Q = [q0, q1)
```

Йдемо lifting scheme у зворотному порядку і визначаємо, які intervals потрібні до кожного step.

Для predict:

```text
odd_new[i] = odd_old[i] + Σ c[j] even_old[i + shift + j]
```

Якщо потрібен interval `I` нового odd, тоді потрібні:

```text
old odd:  I
old even: I + [shift, shift + K - 1]
```

Для update аналогічно.

Для scale:

```text
required_old = required_new
```

Для swap:

```text
swap required_even / required_odd
```

Після backward pass отримаємо:

```text
required initial even interval
required initial odd interval
```

Далі:

```text
1. Load required even/odd cone from DRAM.
2. Розмістити їх у slot A/B.
3. Виконати всі lifting steps через той самий 3-slot engine.
4. Взяти лише потрібний final interior.
5. Записати final coefficients у DRAM.
6. Перейти до наступного chunk.
```

Тобто intermediate results **ніколи не потрапляють у DRAM**.

```text
DRAM initial signal
       ↓
large dependency cone
       ↓
all lifting steps in L1
       ↓
final chunk to DRAM
```

Є невелике повторне обчислення halo між сусідніми chunks, але для великого chunk overhead буде:

```text
approximately 1 + cone_halo / chunk_size
```

і буде набагато дешевшим за повний DRAM pass на кожному step.

---

# 7. Unified architecture

Насправді resident і streamed modes можуть використовувати той самий execution engine:

```text
LwtWindowEngine
├── slot A
├── slot B
├── scratch slot
├── active_even
├── active_odd
├── free_slot
└── static lifting-step sequence
```

Різниця лише у window planner:

```cpp
enum class LwtMemoryMode {
    kResidentSharded,
    kConeStreamed,
};

struct LwtExecutionPlan {
    LwtMemoryMode mode;
    uint32_t core_count;
    uint32_t shard_elements;
    uint32_t final_chunk_elements;
    std::vector<StepGeometry> steps;
};
```

Selection:

```cpp
LwtExecutionPlan make_execution_plan(...) {
    if (full_signal_fits_in_three_sharded_slots()) {
        return make_resident_plan();
    }

    return make_dependency_cone_plan();
}
```

Не потрібно мати runtime switch у кожному read.

Крім того, `TensorAccessorArgs` кодує memory layout на compile time. Тому host-side вибір між двома kernel variants чистіший:

```text
lwt_reader_resident_l1.cpp
lwt_reader_dram_window.cpp
```

або один kernel із двома compile-time accessors, але не один абстрактний accessor, де лише змінюється address.

---

# 8. Які конкретно частини переписати

## `lifting/plan.hpp`

Прибрати:

```cpp
LogicalStream family;
StreamSlot ping/pong;
with_toggled_slot();
```

Додати:

```cpp
StorageSlot A/B/Scratch;
active_even;
active_odd;
free_slot;
```

`compute_step_geometry()` лишити.

---

## `lifting/device.hpp`

Замість:

```cpp
MeshBufferPair even;
MeshBufferPair odd;
```

зробити:

```cpp
struct LiftingWorkingBuffers {
    std::array<std::shared_ptr<MeshBuffer>, 3> slots;
    std::shared_ptr<MeshBuffer> route_config;
};
```

---

## `lifting/device.cpp`

Замінити:

```cpp
create_signal_mesh_buffer(... BufferType::DRAM)
```

на створення L1 height-sharded buffer.

`build_route_config_words()` повинен resolve-ити:

```cpp
slots[route.source]
slots[route.base]
slots[route.output]
```

---

## `build_core_work()`

Зараз кожен route заново рівномірно ділиться між cores.

Для sharded layout краще, щоб core завжди обробляв groups, які фізично належать його shard:

```cpp
core_group_begin = core_index * groups_per_shard;
core_group_end   = min(
    core_group_begin + groups_per_shard,
    route_group_count);
```

Тоді:

- output writes локальні;
- source/base body переважно локальні;
- remote reads залишаються лише для halo.

---

## `pad_split`

Додати resident variant, який пише початкові even/odd прямо в slot A/B.

---

## Final drain

Після останнього step окремий kernel:

```text
active even L1 → final even DRAM
active odd  L1 → final odd DRAM
```

Це буде єдиний повний L1→DRAM write після transform.

Terminal scale steps пізніше можна застосувати прямо у final drain, але спочатку я б лишив їх звичайними L1 routes.

---

# 9. Рекомендований порядок реалізації

### Етап 1 — повністю resident backend

1. Три L1-sharded slots.
2. Pad/split одразу в L1.
3. Поточний reader/compute/writer, але source/output — L1.
4. Static shard ownership.
5. Один final DRAM drain.
6. Benchmark.

Це вже має прибрати основний bottleneck без dependency-cone складності.

### Етап 2 — memory planner

Додати перевірку:

```text
чи влазять 3 slots + CB + safety margin
```

### Етап 3 — dependency-cone fallback

Для сигналів, які не влазять:

```text
final chunk → backward cone → DRAM load → all LWT steps → final write
```

---

## Підсумкова схема

```text
                    ┌──────────────────────────┐
                    │ Host memory planner      │
                    └────────────┬─────────────┘
                                 │
                 ┌───────────────┴────────────────┐
                 │                                │
         signal fits                       signal does not fit
                 │                                │
                 ▼                                ▼
       ResidentSharded                    ConeStreamed
                 │                                │
      pad/split → L1 A/B              load cone from DRAM
                 │                                │
                 └──────────┬─────────────────────┘
                            ▼
                  Three-slot LWT engine
                   A / B / scratch
                            │
                    all lifting steps
                            │
                            ▼
                   one final DRAM write
```

Це зберігає твою нинішню LWT structure, static scheme і SFPU stencil, але повністю міняє неправильну частину — route-by-route DRAM materialization. Найкращою першою реалізацією я вважаю саме **three-slot L1-sharded resident backend**, а не складний L1/DRAM blob cache.