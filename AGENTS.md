Firstly check docs and limitations of the project

-[text](AGENTS.md)
-[text](docs/DEFINITIONS.md)
-[text](docs/HORIZONTAL_STENCIL.md)
-[text](docs/LWT.md)
-[text](docs/PROJECT_LIMITATIONs.md)
-[text](docs/VERTICAL_STENCIL.md)


Be logical validate the idea be precise write greate performance code and check try to refer to docs.

the Code of chunked kernel in [chunked kernel](chunk_kernel.txt)

Так, я посидів і тут головний висновок такий:

> **Dependency cone не треба реалізовувати як “один chunk kernel з scalar масивами”.**
> Це майже гарантовано повільніше, бо ти вбиваєш головну перевагу старого backend — **SFPU/tile compute**.
> Правильна модель: **dependency cone має змінити schedule і materialization boundaries**, але **залишити наявний `reader → compute/SFPU → writer` execution style**.

Тобто переписувати треба не “математику LWT у scalar kernel”, а **план виконання існуючого tile/SFPU pipeline**.

Нижче доводжу модель до кінця.

---

# 1. Що в source вже є хорошого

У твоєму старому backend уже є правильна high-performance основа:

```text
reader:
    читає source/base stream
    формує tiles для compute

compute:
    hstencil_plus_base_tile<K>()
    тобто SFPU horizontal stencil

writer:
    пише output tiles назад у row-major sticks
```

Це видно в snapshot: `lwt_compute.cpp` бере `cb_input0`, `cb_input1`, `cb_base`, викликає `hstencil_plus_base_tile<K>()`, пакує два output tiles у `cb_output` . Reader формує source/base tiles по `kLwtGroupOutputElements`, `kLwtRowsPerGroup`, `kLwtOutputBlocksPerRow`, `kLwtHalfStickElements` . Writer потім розкладає output tiles назад у row-major sticks і після route робить route barrier/sync .

І host уже створює нормальний TT-Metal program з трьома kernels:

```cpp
ReaderDataMovementConfig
ComputeConfig
WriterDataMovementConfig
```

з `fp32_dest_acc_en = true` і `UnpackToDestFp32` для input/base CBs .

Тобто старий backend **не треба викидати як low-level основу**. Треба змінити тільки те, **які діапазони він обробляє і куди матеріалізує intermediate streams**.

---

# 2. Чому Codex-версія повільна

Те, що він написав, фактично робить:

```text
one dataflow kernel:
    read original input
    fill local float arrays
    run all LWT steps using scalar loops
    write final output
```

У snapshot видно scalar chunk kernel: він бере `get_write_ptr(cb_even)`, `get_write_ptr(cb_odd)`, тримає `local_capacity`, будує `LocalStream`, і якщо capacity не влазить, пише empty output .

Це математично схоже на dependency cone, але performance-wise погано:

```text
старий route backend:
    більше memory/materialization
    але math = SFPU/tile

chunk scalar backend:
    менше global materialization
    але math = scalar dataflow core
```

Тому воно й повільніше.

**Головний принцип:** dependency cone має прибрати глобальні проміжні materialization, але не має переносити арифметику з SFPU у scalar RISC-V/dataflow kernel.

---

# 3. Правильна модель: tiled dependency cone, не scalar dependency cone

Поточний `lwt_reader/compute/writer` працює не по окремих scalar elements, а по **LWT output groups**.

У protocol:

```cpp
kLwtRowsPerGroup = 32
kLwtOutputBlocksPerRow = 3
kLwtHalfStickElements = 16
```

Тобто один output group містить:

[
G = 32 \cdot 3 \cdot 16 = 1536
]

output scalar elements.

Це важливо.

**Dependency cone треба рахувати не для chunk size = 128 elements, а для chunk size у group units.**

Наприклад:

```text
chunk = 1 group  = 1536 elements
chunk = 2 groups = 3072 elements
chunk = 4 groups = 6144 elements
chunk = 8 groups = 12288 elements
```

Тобто базова одиниця high-perf schedule:

```text
final output group interval
```

а не arbitrary scalar interval.

---

# 4. Формальна модель

## 4.1. LWT graph

Node:

[
E_i^{(r)}, O_i^{(r)}
]

де:

* (i) — logical stream index;
* (r) — step/layer;
* (E) — even stream;
* (O) — odd stream.

Predict:

[
O^{(r+1)}[i]
============

O^{(r)}[i]
+
\sum_{j=0}^{K-1} c_j E^{(r)}[i - \text{shift} - (K-1) + j]
]

У current convention source span:

```cpp
source_span_lo = -shift - K + 1;
source_span_hi = -shift;
```

Тому для output interval:

[
I = [a,b]
]

source interval:

[
I + [source_span_lo, source_span_hi]
]

---

## 4.2. Backward dependency cone

Для кожного layer (r):

[
D_E^{(r)} = \text{required even interval before step } r
]

[
D_O^{(r)} = \text{required odd interval before step } r
]

Final chunk:

[
D_E^{(R)} = I_m
]

[
D_O^{(R)} = I_m
]

Predict backward:

[
D_O^{(r)} = D_O^{(r+1)}
]

[
D_E^{(r)}
=========

D_E^{(r+1)}
\cup
\left(D_O^{(r+1)} + [source_span_lo, source_span_hi]\right)
]

Update backward:

[
D_E^{(r)} = D_E^{(r+1)}
]

[
D_O^{(r)}
=========

D_O^{(r+1)}
\cup
\left(D_E^{(r+1)} + [source_span_lo, source_span_hi]\right)
]

Swap:

[
D_E^{(r)} = D_O^{(r+1)}
]

[
D_O^{(r)} = D_E^{(r+1)}
]

Scale:

```text
does not expand interval
```

Це ми вже розуміємо.

Але тепер важливе продовження.

---

# 5. Потрібна не просто final cone, а interval for every layer

Для high-perf schedule треба зберігати не тільки:

```text
initial needed E/O
```

а всю таблицю:

```text
layer 0:
    E interval
    O interval

layer 1:
    E interval
    O interval

layer 2:
    E interval
    O interval

...

layer R:
    E final interval
    O final interval
```

Тобто host має побудувати:

```cpp
struct ConeLayer {
    Interval even;
    Interval odd;
};

std::array<ConeLayer, R + 1> cone;
```

Чому?

Бо forward execution step (r) має знати:

```text
який target interval треба materialize після step r
```

Наприклад predict:

```text
O_new = O_old + stencil(E_old)
```

Треба обчислити не весь `O_new`, а тільки:

[
D_O^{(r+1)}
]

Even не змінюється. Його не треба копіювати.

Update:

[
D_E^{(r+1)}
]

Scale terminal — не route.

Це дуже важливо для performance.

---

# 6. Forward execution після backward cone

Після backward planning actual compute йде forward.

Для chunk (m):

```text
1. backward cone gives all D_E[r], D_O[r]
2. build initial E/O local slices for D_E[0], D_O[0]
3. for r = 0..R-1:
       execute only target interval D_target[r+1]
4. write final owned output interval
```

Для predict:

```text
source = E current over D_E[r]
base   = O current over D_O[r]
target = O next over D_O[r+1]

compute:
    O_next[D_O[r+1]] =
        O_current[D_O[r+1]] + stencil(E_current)

E_current is unchanged.
No copy of E.
```

Для update:

```text
source = O current over D_O[r]
base   = E current over D_E[r]
target = E next over D_E[r+1]

compute:
    E_next[D_E[r+1]] =
        E_current[D_E[r+1]] + stencil(O_current)

O_current is unchanged.
No copy of O.
```

Це означає:

```text
ping-pong only updated stream
```

а не:

```text
copy unchanged stream every step
```

---

# 7. Чому це не має бути scalar local arrays

Наївний dependency cone каже:

```text
read input + halo into float arrays
for step:
    for i:
        scalar stencil
```

Правильний TT dependency cone має казати:

```text
read input + halo into local tiled stream representation
for step:
    emit source/base tiles
    run hstencil_plus_base_tile on SFPU
    write target tiles to local scratch
```

Тобто actual compute unit лишається:

```cpp
hstencil_plus_base_tile<K>()
```

а dependency cone тільки обмежує:

```text
які groups/tiles треба подати в compute
і де вони лежать
```

---

# 8. Нова high-perf architecture

Назвемо її:

```text
Chunk-major tiled route fusion
```

Це не “one scalar chunk kernel”.

Це:

```text
for each output chunk:
    run all routes locally
    but each route still uses reader/compute/writer SFPU pipeline
```

Візуально:

```text
old schedule:

route 0 over whole signal
    reader -> compute -> writer -> global materialization

route 1 over whole signal
    reader -> compute -> writer -> global materialization

route 2 over whole signal
    reader -> compute -> writer -> global materialization


new schedule:

chunk 0:
    route 0 over cone slice
        reader -> compute/SFPU -> local writer
    route 1 over cone slice
        reader -> compute/SFPU -> local writer
    route 2 over cone slice
        reader -> compute/SFPU -> final writer

chunk 1:
    same

chunk 2:
    same
```

Тобто ми міняємо outer loop:

```cpp
// old
for route:
    for global groups:
        compute

// new
for chunk:
    for route:
        compute local groups
```

Але всередині route лишається старий fast tile compute.

---

# 9. Storage model: local stream slices

Для кожного core/chunk треба мати local stream descriptors:

```cpp
struct LocalStreamSlice {
    LogicalStream family;       // even or odd
    int32_t logical_lo;         // logical index of first stored value
    uint32_t length;            // stored logical length
    uint32_t stick_count;       // row-major stick storage size
    uint32_t base_l1_addr;      // local scratch base
};
```

А для ping-pong:

```cpp
struct LocalParityStorage {
    LocalStreamSlice a;
    LocalStreamSlice b;
    bool cur_is_a;
};
```

Predict:

```text
odd.free receives O_next[D_O[r+1]]
odd.cur = odd.free
old odd.cur becomes free
even.cur unchanged
```

Update:

```text
even.free receives E_next[D_E[r+1]]
even.cur = even.free
old even.cur becomes free
odd.cur unchanged
```

Swap:

```text
swap even.cur with odd.cur
swap even.free with odd.free
swap interval metadata
```

Scale:

```text
metadata only
```

---

# 10. Як local scratch має бути layouted

Найкращий варіант: **row-major stick layout такий самий, як поточні stream buffers**.

Чому?

Бо тоді можна reuse-нути майже всю логіку з `lwt_reader.cpp` і `lwt_writer.cpp`.

Поточний reader already knows how to read source/base row-major streams and emit:

```text
cb_src_tile0
cb_src_tile1
cb_base_tile
```

для `hstencil_plus_base_tile` .

Поточний writer already knows how to take `cb_output` and write row-major half-blocks/sticks .

Тобто local scratch має виглядати як звичайний stream:

```text
stick 0: values [0..31]
stick 1: values [32..63]
...
```

але це не full global stream, а тільки local slice:

```text
logical [lo..hi]
stored as local [0..length-1]
```

Тоді access formula:

```cpp
local_index = logical_index - logical_lo;
stick = local_index / 32;
lane  = local_index % 32;
```

---

# 11. Initial reader: direct input + symmetric + split

Для layer 0 треба побудувати:

```text
E^0[D_E[0]]
O^0[D_O[0]]
```

З original input.

Формули:

```text
E0[p] = x_pad[2p]
O0[p] = x_pad[2p + 1]
```

А padded index переводиться в original:

```text
original_index = symmetric_index(padded_index - left_pad, input_length)
```

Тут можна reuse-нути existing `StickReadCache` / `read_padded_symmetric_value`, бо pad_split reader уже це робить через stick cache, а не через random per-float NoC reads .

Але результат initial reader має писати не в global even/odd buffers, а в **local scratch stream slice** для цього chunk.

---

# 12. Local route reader

Після layer 0 всі source/base streams уже в local scratch.

Для кожного predict/update route (r), local reader має emit tiles from local slice:

```text
source slice = current E or O
base slice   = current O or E
target interval = D_target[r+1]
```

Він має сформувати ті самі CBs, що старий reader:

```text
cb_src_tile0
cb_src_tile1
cb_base_tile
```

Тобто логіка майже така сама як `emit_predict_update_tiles`, але замість global route config і global buffer адрес він використовує:

```text
local source base address
local source logical_lo
local base base address
local base logical_lo
target output group interval
```

Ключ:

```text
reader still emits exactly the tile format expected by hstencil_plus_base_tile
```

---

# 13. Local compute

Compute kernel майже не треба перепридумувати.

Він вже робить:

```cpp
run_predict_update_step<K>(
    cb_input0,
    cb_input1,
    cb_base,
    cb_output,
    Step::coeff_bits,
    output_group_count
);
```

і всередині використовує `hstencil_plus_base_tile<K>()` .

Зміна:

```text
output_group_count now means local target groups for this chunk/step
```

а не global whole-route groups.

Scale routes бажано прибрати:

```text
terminal scale is fused at final output
```

---

# 14. Local writer

Writer після compute має два режими.

## 14.1. Intermediate route writer

Для не-final predict/update:

```text
write cb_output into local scratch target slice
```

Тобто:

```text
O_next local slice
or
E_next local slice
```

Він використовує ту саму tile-to-stick логіку, що старий writer, але destination — local scratch, не global output stream.

## 14.2. Final writer

Після останнього useful predict/update:

```text
write only owned final output interval
apply final scale
do not write halo output
```

Тобто якщо local final slice більший через halo:

```text
local final slice: [a-H, b+H]
owned output:      [a, b]
```

writer пише тільки `[a,b]`.

---

# 15. Sync model

У старому backend є global route barrier між routes. Writer після route робить `route_barrier`, потім пушить `cb_sync`, а reader наступного route чекає `cb_sync` .

У dependency-cone chunk backend **global barrier не потрібен**, бо кожен core/chunk незалежний.

Потрібен тільки local order:

```text
writer route r finished local scratch
reader route r+1 may read local scratch
```

Це можна зробити через local `cb_sync`, без cross-core semaphore.

Тобто:

```text
remove cross-core route_barrier
keep local reader/writer sync
```

Це має дати speedup саме на multi-route schemes.

---

# 16. Host-side planner

Host має робити не просто:

```text
chunk_size + 8*tap_size + 512
```

Це погано.

Host має точно рахувати cone.

Для кожного chunk:

```cpp
struct ChunkPlan {
    uint32_t core_id;

    Interval final_even_owned;
    Interval final_odd_owned;

    std::array<ConeLayer, R + 1> cone;

    uint32_t max_even_sticks;
    uint32_t max_odd_sticks;

    uint32_t local_scratch_bytes;

    std::array<StepLocalPlan, R> steps;
};
```

Для кожного step:

```cpp
struct StepLocalPlan {
    StepType type;

    Interval source_interval;
    Interval base_interval;
    Interval target_interval;

    uint32_t source_local_offset;
    uint32_t base_local_offset;
    uint32_t target_local_offset;

    uint32_t target_group_begin;
    uint32_t target_group_count;
};
```

І тільки якщо:

```text
local_scratch_bytes <= available_l1_budget
```

chunk може запускатися.

Інакше треба збільшити/зменшити chunk group count.

---

# 17. Quantization to groups

Оскільки compute unit — group/tile-based, target intervals треба quantize.

Нехай:

[
G = 1536
]

Тоді:

```cpp
group_lo = floor_div(interval.lo, G);
group_hi = ceil_div(interval.hi + 1, G) - 1;
```

А target materialized interval:

```cpp
materialized_lo = group_lo * G;
materialized_hi = (group_hi + 1) * G - 1;
```

Але writer пише тільки valid logical interval.

Це дає трохи extra local work, але зберігає tile/SFPU efficiency.

---

# 18. Основний performance model

Старий backend:

```text
for each route r:
    read full source/base route ranges
    SFPU compute
    write full output route range
    route barrier
```

Навіть якщо working buffers default L1, все одно є:

```text
full stream pass per route
reader tile construction per route over entire stream
writer pass per route over entire stream
cross-core route sync
terminal scale routes
```

До речі, у source default working buffer type — L1, якщо env не задано, тобто bottleneck не завжди “DRAM”, а ширше: **full-stream route materialization/pass/sync** .

Новий tiled dependency cone:

```text
for each chunk:
    read initial input + halo once
    for each route:
        process only local cone slice
        write intermediate only to local scratch
        no cross-core barrier
    final write only owned output
```

Перевага:

```text
global full-route passes disappear
cross-core route barriers disappear
terminal scale routes disappear
SFPU compute remains
```

Це вже справжня high-perf модель.

---

# 19. Чому не треба “all steps in one compute kernel only”

Можна подумати:

```text
reader reads initial E/O
compute runs all steps
writer writes final
```

Але compute kernel сам не дуже зручно читає arbitrary local scratch і робить tile layout. Поточна архітектура вже розділяє:

```text
reader constructs tiles
compute does SFPU
writer stores tiles
```

Тому більш реалістична high-perf модель:

```text
within one chunk:
    route-like micro-pipeline
```

Тобто:

```text
reader/compute/writer still exist,
but they process local cone slices instead of global routes.
```

Це важливий компроміс:

```text
ми не scalarize compute
і не намагаємось запхати весь memory layout у compute kernel
```

---

# 20. Що саме треба сказати Codex

Ось стисла, але повна постановка задачі:

```text
The previous dependency-cone implementation is slower because it implements the cone as a scalar dataflow kernel.

Do not continue that direction.

The correct optimization is a tiled dependency-cone schedule over the existing reader/compute/writer SFPU pipeline.

Use the current LWT group/tile representation as the execution unit:
- kLwtRowsPerGroup = 32
- kLwtOutputBlocksPerRow = 3
- kLwtHalfStickElements = 16
- group_output_elements = 1536

Compute dependency cones on host per final output group chunk.

For each chunk, keep the full cone table:
    D_E[r], D_O[r] for every layer r.

Run forward over the cone:
    initial reader builds E0/O0 local slices from original input + symmetric boundary;
    for each predict/update step:
        reader emits source/base tiles from local stream slices;
        compute uses the existing hstencil_plus_base_tile<K>();
        writer stores the updated target stream to local scratch;
    terminal scale is fused into final output;
    final writer writes only the owned final output interval.

Do not use scalar LocalStream arithmetic for predict/update.

Do not use chunk size in arbitrary scalar elements; use group/tile-aligned chunks.

Do not estimate local capacity with chunk_size + 8*tap_size + 512.
Compute exact cone intervals and exact local scratch requirements.

Avoid global route barriers.
Only local writer-to-reader sync is needed between micro-routes inside the same chunk/core.

Keep the existing source span convention:
    source_span_lo = -shift - K + 1
    source_span_hi = -shift

Preserve existing coefficient ordering used by hstencil_plus_base_tile.

The goal is:
    same math,
    same SFPU stencil compute,
    different schedule,
    local materialization only,
    no full-stream intermediate route passes.
```

---

# 21. Найкоротший verdict

Те, що він написав, правильно як **proof of correctness**, але не як optimization.

Правильна dependency-cone optimization для твого codebase:

```text
не:
    scalar chunk_lwt.cpp

а:
    chunk-major schedule over existing SFPU route micro-pipeline
```

Тобто:

```text
Dependency cone = host-side plan of local stream intervals.
Reader = builds tiles for those intervals.
Compute = existing hstencil_plus_base_tile.
Writer = local scratch or final output.
```

І ключова фраза:

> **Dependency cone should restrict and reorder the existing tile/SFPU computation. It should not replace SFPU computation with scalar dataflow arithmetic.**


але от ти кажеш reuse все але можливо воно не дуже підійде і ти ліпше переписати це навколо самого Dependency cone
Виконано пошук з посиланням на завантажені файли

Так, тут ти правий. Я би уточнив свою позицію:

Reuse старого backend не означає “взяти lwt_reader.cpp/lwt_writer.cpp і трохи підкрутити”.
Скоріше: старий backend треба використати як reference для tile layout, SFPU primitive, coefficient convention і correctness, але саму архітектуру, ймовірно, краще переписати навколо Dependency Cone IR.

Бо поточний route backend фундаментально побудований навколо іншого schedule:

route 0 globally
route 1 globally
route 2 globally
...

А dependency cone хоче:

chunk 0: all steps locally
chunk 1: all steps locally
chunk 2: all steps locally
...

Це не маленький patch. Це новий backend.

1. Чому старий lwt_reader/writer може не підійти напряму

Старий backend має хороші частини, але його структура route-oriented.

Що в ньому добре

У тебе вже є:

reader -> compute -> writer

і compute kernel реально використовує:

hstencil_plus_base_tile<K>()

через SFPU/tile path. Це видно в lwt_compute.cpp: compute читає source/base tiles з CB, викликає hstencil_plus_base_tile<K>(), пакує output у cb_output .

Також host уже створює нормальну TT-Metal програму з окремими reader, compute, writer kernels через ReaderDataMovementConfig, ComputeConfig, WriterDataMovementConfig .

Це цінно.

Що в ньому погано для dependency cone

Поточний reader читає global route config і для кожного route генерує source/base tiles з глобальних stream buffers. Поточний writer пише output route назад у global row-major buffer, а потім робить route barrier перед наступним route .

Тобто він мислить так:

route input = global stream
route output = global stream
after route: synchronize all cores

Dependency cone хоче інакше:

chunk input = original input + halo
intermediate streams = local per-core scratch
no global route barrier
final output = owned chunk only

Тому старий reader/writer можна брати як приклад tile packing/unpacking, але не як готову архітектуру.

2. Правильна абстракція: не routes, а ConePlan

Я б переписував backend навколо нового host-side IR:

struct Interval {
    int32_t lo;
    int32_t hi; // inclusive
};

struct ConeLayer {
    Interval even;
    Interval odd;
};

struct ConeStep {
    StepType type;

    Interval source;
    Interval base;
    Interval target;

    LogicalStream source_family;
    LogicalStream base_family;
    LogicalStream target_family;

    int32_t source_span_lo;
    int32_t source_span_hi;

    uint32_t target_group_begin;
    uint32_t target_group_count;
};

struct ConeChunkPlan {
    uint32_t chunk_id;
    Interval final_even_owned;
    Interval final_odd_owned;

    std::vector<ConeLayer> layers; // size Scheme::num_steps + 1
    std::vector<ConeStep> steps;

    uint32_t local_even_capacity_elements;
    uint32_t local_odd_capacity_elements;
    uint32_t local_scratch_bytes;

    uint32_t final_even_scale_bits;
    uint32_t final_odd_scale_bits;
};

Суть:

routes = глобальний schedule старого backend
ConePlan = локальний schedule нового dependency-cone backend

Це принципово інший план.

3. Що саме рахує host

Host має для кожного output chunk побудувати повну таблицю:

layer 0: D_E[0], D_O[0]
layer 1: D_E[1], D_O[1]
layer 2: D_E[2], D_O[2]
...
layer R: D_E[R], D_O[R]

Не тільки initial cone.

Чому? Бо forward execution має знати, який interval materialize після кожного step.

Наприклад predict:

O_next[D_O[r+1]] = O_cur[D_O[r+1]] + stencil(E_cur)

Тут target interval — саме D_O[r+1], а не весь odd stream.

Update:

E_next[D_E[r+1]] = E_cur[D_E[r+1]] + stencil(O_cur)

Тобто кожен step має локальний target interval.

4. Dependency cone у правильній convention

У твоєму коді source span convention уже фактично такий:

source_span_lo = -shift - K + 1
source_span_hi = -shift

Це треба зберегти. Поточний route planner теж через compute_step_geometry рахує convolution geometry з source.shift, kernel_shift, k, base.shift .

Backward rules:

Predict
O_new = O_old + stencil(E_old)

Тоді:

D_O_before = D_O_after

D_E_before = union(
    D_E_after,
    D_O_after + [source_span_lo, source_span_hi]
)
Update
E_new = E_old + stencil(O_old)

Тоді:

D_E_before = D_E_after

D_O_before = union(
    D_O_after,
    D_E_after + [source_span_lo, source_span_hi]
)
Swap
D_E_before = D_O_after
D_O_before = D_E_after
Scale
no spatial expansion

Scale в усіх схемах terminal, тому це final multiplier, не compute route.

5. Найважливіший design decision: local scratch, не FIFO-only CB

Оце прям центральна проблема.

Поточний lwt_compute — streaming kernel:

cb_wait_front(source)
copy_tile(source)
cb_pop_front(source)

cb_wait_front(base)
copy_tile(base)
cb_pop_front(base)

compute

pack output
cb_push_back(output)

Це добре для одного route, але dependency cone потребує, щоб intermediate stream після step r можна було знову читати на step r+1.

FIFO CB не дуже природно підходить як random-access local storage.

Тому новий backend має мати поняття:

local stream scratch

а не тільки transient CB pipe.

Модель:

E_A local slice
E_B local slice
O_A local slice
O_B local slice

Кожен slice — row-major stick/tile-compatible storage у L1 для поточного chunk.

Predict:

source = E_cur
base   = O_cur
target = O_free

O_free = O_cur + stencil(E_cur)
O_cur <-> O_free
E_cur unchanged

Update:

source = O_cur
base   = E_cur
target = E_free

E_free = E_cur + stencil(O_cur)
E_cur <-> E_free
O_cur unchanged

Це і є dependency-cone native storage model.

6. Тобто архітектура з нуля має бути така

Не:

reuse old route config
reuse old global route reader
reuse old global route writer

А:

ConePlan host side
    ↓
chunk reader
    builds local E0/O0 slices from original input + halo
    ↓
chunk compute
    uses SFPU stencil on local slices
    ↓
chunk writer
    writes final owned interval

Але compute все одно має бути SFPU/tile.

Тобто новий backend:

DependencyConeLwtProgram
    reader:  data movement into local scratch
    compute: SFPU over local scratch
    writer:  final output

А не scalar chunk_lwt.cpp.

7. Як зробити compute навколо dependency cone

Тут є два варіанти.

Варіант A — micro-pipeline всередині chunk

Це ближче до старої моделі, але без global materialization.

Для кожного chunk:

initial_reader:
    original input -> local E/O scratch

for each predict/update step:
    local_reader:
        local source/base scratch -> source/base tiles CB

    compute:
        hstencil_plus_base_tile<K>()

    local_writer:
        output tiles CB -> local target scratch

final_writer:
    final active local scratch -> global output

Тут reader/compute/writer все ще існують, але працюють по local scratch, не по global route buffers.

Плюс:

легше reuse tile construction logic

Мінус:

більше локальних phase transitions
Варіант B — compute kernel сам читає local scratch tiles

Compute kernel напряму бере source/base tiles з local scratch і пише target scratch.

Тоді dataflow reader тільки будує initial scratch, writer тільки final output.

Плюс:

менше kernel-level pipeline overhead

Мінус:

складніше, бо compute kernel має вміти адресувати local scratch/tile pages

Я б для “нормальної optimized версії з нуля” думав саме про B, але треба бути чесним: на TT-Metal compute kernels найзручніше працюють через CB/unpack/pack. Тому practical design може бути B концептуально, але технічно мати CB windows over scratch.

8. Чому не варто мислити “chunk_size = 128”

Це scalar-thinking.

У твоєму current protocol є:

kLwtRowsPerGroup = 32
kLwtOutputBlocksPerRow = 3
kLwtHalfStickElements = 16

Отже один LWT group:

32 * 3 * 16 = 1536 scalar values

Це natural compute unit старого SFPU pipeline.

Dependency cone backend має бути group/tile-aligned:

chunk_groups = 1, 2, 4, 8
chunk_elements = chunk_groups * 1536

Не arbitrary 128.

Чому:

SFPU primitive очікує tile/group geometry
writer пише half-blocks
reader формує 2 source tiles + 2 base tiles

Якщо зробити 128, ти отримаєш багато overhead і partial edge cases.

9. Повна модель нового backend
9.1 Host

Host робить:

for each output chunk:
    1. choose final owned interval in group units
    2. compute full dependency cone table D_E[r], D_O[r]
    3. quantize intervals to group/tile boundaries
    4. compute local scratch capacity
    5. emit compact ChunkPlan

Важливо:

capacity = exact max over cone layers
not chunk_size + 8*tap_size + 512
9.2 Initial reader

Reader читає original input + halo.

Для E0[p]:

padded_index = 2p
original_index = symmetric_index(padded_index - left_pad, input_length)

Для O0[p]:

padded_index = 2p + 1
original_index = symmetric_index(padded_index - left_pad, input_length)

Воно має використовувати stick cache, як pad_split reader. У source вже є StickReadCache і read_padded_symmetric_value .

Але результат іде не в global even/odd buffers, а в local scratch:

E_A = E0[D_E[0]]
O_A = O0[D_O[0]]
9.3 Compute

Для кожного step:

Predict
source slice = E_cur
base slice   = O_cur
target slice = O_free for interval D_O[r+1]

target = base + stencil(source)

Compute має використовувати:

hstencil_plus_base_tile<K>()

а не scalar loops. Сам primitive уже є в stencil_sfpi.h .

Update
source slice = O_cur
base slice   = E_cur
target slice = E_free for interval D_E[r+1]

target = base + stencil(source)
Swap
swap stream roles
Scale
do nothing until final output
9.4 Writer

Final writer:

read final active E/O local scratch
apply final scale
write only owned final output interval

Не писати halo.

Не застосовувати old canonicalization повторно, якщо writer уже пише canonical output.

10. Що саме можна “reuse”, а що переписати
Reuse as-is або майже as-is
hstencil_plus_base_tile<K>()
scale_tile()
coefficient bit representation
StepType / StaticStep / generated schemes
symmetric_index
StickReadCache ideas
tile_row_major offset/write helpers
Reuse as reference, але не напряму
lwt_reader.cpp
lwt_writer.cpp
route_config
route barriers
build_core_work by route groups

Бо вони route-global.

Переписати з нуля
host LWT scheduling
chunk planning
local scratch descriptor
dependency cone table
chunk-level runtime args
initial input+halo reader
local scratch writer
final writer ownership/canonical mapping
11. Мені здається, найкраща нова назва

Я б не називав це просто chunk_lwt.

Краще:

cone_lwt

або:

fused_cone_lwt

Бо chunk_lwt вже асоціюється зі scalar slow prototype.

Наприклад файли:

kernels/dataflow/cone_lwt_reader.cpp
kernels/compute/cone_lwt_compute.cpp
kernels/dataflow/cone_lwt_writer.cpp

Host:

ConeLwtProgram
ConeChunkPlan
ConeLayer
ConeStep
ConeScratchLayout
12. Що сказати Codex як новий conceptual prompt

Ось короткий текст, який я б йому дав:

The previous chunk_lwt implementation is not the target architecture. It implemented dependency cones as scalar LocalStream arithmetic in a dataflow kernel, which is slower.

Design a new backend around Dependency Cone as the central IR.

Do not try to patch the old route scheduler directly. Use the old route backend only as a reference for:
- coefficient order,
- source span convention,
- SFPU stencil primitive,
- tile/group layout,
- correctness.

The new backend should be organized around ConePlan:

For each final output chunk:
    compute a full dependency cone table:
        D_E[r], D_O[r] for every step layer r.
    Use this table to define exact local stream slices for every step.

Execution:
    initial reader:
        original input + symmetric boundary -> local E0/O0 scratch
    compute:
        for each step:
            predict/update using SFPU hstencil_plus_base_tile
            store only the target interval required by the next cone layer
            ping-pong only the updated parity stream
        swap = metadata
        scale = terminal metadata
    final writer:
        apply terminal scale
        write only owned final output interval

Do not use scalar LocalStream loops for predict/update.
Do not use arbitrary scalar chunk size 128.
Use group/tile-aligned chunks based on kLwtGroupOutputElements = 1536.
Do not estimate local capacity from tap_size heuristics.
Compute exact max local scratch requirement from the cone table.

The old lwt_reader/lwt_writer are route-global and may not fit directly. It is acceptable to rewrite reader/writer around local cone scratch, but preserve their tile layout semantics and SFPU compute convention.

Main invariant:
    same mathematical LWT graph,
    different schedule,
    local materialization per dependency cone,
    no global route materialization,
    no scalar arithmetic replacement of SFPU.
13. Найкоротший висновок

Так: можливо, краще переписати backend навколо dependency cone з нуля.

Але не з нуля в сенсі “забути старий код”.

Правильне розділення:

старий код = reference + low-level primitives
новий код = новий scheduler/storage model around dependency cone

Суть нового backend:

ConePlan on host
local scratch per chunk
SFPU predict/update over local slices
terminal scale fused
final owned output only

І головна заборона:

dependency cone не має означати scalar LocalStream kernel

Dependency cone має означати:

новий graph schedule + local materialization boundaries

а не:

новий повільний scalar алгоритм