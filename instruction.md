Так, ідея правильна. Я б зробив це не як “JSON → runtime scheme”, а як:

```cpp
ttwv::lwt(signal, "bior2.2")
        ↓
runtime string lookup
        ↓
compile-time scheme type
        ↓
specialized fused TT-Metal program
```

Тобто public API лишається зручним, але `"bior2.2"` всередині просто вибирає вже згенерований C++ тип.

---

## 1. Що зараз не так

Зараз у тебе схема реально вантажиться з JSON у `RuntimeLiftingScheme`: парситься `tap_size`, `delay_even`, `delay_odd`, `steps`, `shift`, `coefficients` і потім пакується в device descriptors. Це видно у твоєму `load_runtime_lifting_scheme(...)` і `pack_device_step_desc(...)`.

А в `lwt_compute.cpp` уже є частково правильна ідея: `k_stencil_width` береться як compile-time arg, а коефіцієнти кроку читаються через runtime args у `run_step<K>()`. Тобто stencil width уже compile-time, але сама схема ще runtime.

Окрема проблема: scale зараз винесений у `lwt_scale_compute.cpp`, тобто predict/update і scale живуть як різні device paths.  Якщо хочеш “одну програму”, треба scale інтегрувати в той самий step loop.

---

## 2. Головна архітектура

Я б ввів **два рівні API**:

```cpp
ttwv::lwt(signal, "bior2.2");
```

і внутрішній compile-time API:

```cpp
ttwv::lwt<ttwv::schemes::bior2_2>(signal);
```

Тоді runtime string не заважає compile-time оптимізації:

```cpp
Tensor lwt(const Tensor& signal, std::string_view wavelet_name) {
    switch (scheme_id(wavelet_name)) {
        case SchemeId::kBior2_2:
            return lwt<schemes::bior2_2>(signal);

        case SchemeId::kCoif2:
            return lwt<schemes::coif2>(signal);

        default:
            TT_THROW("Unsupported wavelet scheme: {}", wavelet_name);
    }
}
```

Важлива думка: **runtime string не може напряму стати template parameter**. Але runtime string може вибрати одну з уже скомпільованих template specializations.

---

## 3. Як представити схему compile-time

Не `std::vector`, а fixed-size структура.

Наприклад:

```cpp
#pragma once

#include <array>
#include <bit>
#include <cstdint>

namespace ttwv {

enum class StepType : uint8_t {
    kPredict,
    kUpdate,
    kScaleEven,
    kScaleOdd,
    kSwap,
};

template <size_t MaxCoeffs>
struct StaticStep {
    StepType type;
    int32_t shift;
    uint32_t k;
    std::array<uint32_t, MaxCoeffs> coeffs;
};

template <size_t NumSteps, size_t MaxCoeffs>
struct StaticScheme {
    uint32_t tap_size;
    int32_t delay_even;
    int32_t delay_odd;
    std::array<StaticStep<MaxCoeffs>, NumSteps> steps;
};

consteval uint32_t f32(float value) {
    return std::bit_cast<uint32_t>(value);
}

}
```

Приклад згенерованої схеми:

```cpp
#pragma once

#include "tt_wavelet/include/lifting/static_scheme.hpp"

namespace ttwv::schemes {

struct bior2_2 {
    static constexpr uint32_t tap_size = 6;
    static constexpr int32_t delay_even = 1;
    static constexpr int32_t delay_odd = 2;
    static constexpr uint32_t num_steps = 4;
    static constexpr uint32_t max_coeffs = 3;

    static constexpr std::array<StaticStep<max_coeffs>, num_steps> steps = {{
        {
            StepType::kPredict,
            0,
            1,
            {f32(-1.0f), 0, 0}
        },
        {
            StepType::kUpdate,
            -1,
            3,
            {f32(0.0625f), f32(0.5f), f32(-0.0625f)}
        },
        {
            StepType::kScaleEven,
            0,
            1,
            {f32(1.41421356237f), 0, 0}
        },
        {
            StepType::kScaleOdd,
            0,
            1,
            {f32(0.70710678118f), 0, 0}
        }
    }};
};

}
```

Це треба **не писати руками**, а генерувати з JSON build-time скриптом:

```text
lifting_schemes/*.json
        ↓
tools/generate_schemes.py
        ↓
include/tt_wavelet/schemes/generated/bior2_2.hpp
include/tt_wavelet/schemes/generated/coif2.hpp
include/tt_wavelet/schemes/generated/registry.hpp
```

Тобто JSON лишається source of truth, але не використовується під час запуску програми.

---

## 4. Як зробити “одну програму”

Тут треба розділити два значення слова “програма”.

У TT-Metal нормальна operator program все одно має:

```text
reader kernel
compute kernel
writer kernel
```

Тому “одна програма” має означати:

```text
one Program launch
one reader
one compute
one writer
internal loop over all lifting steps
```

а не:

```text
one C++ function без reader/writer/compute
```

Бо на Tensix reader/compute/writer — це нормальна модель виконання.

---

## 5. Device-side fused loop

Зараз compute приблизно робить:

```cpp
for step in num_steps:
    read coeffs from runtime args
    run_step<K>(...)
```

Треба перейти до:

```cpp
template <typename Scheme>
void kernel_main_static() {
    run_all_steps<Scheme>();
}
```

Ідея:

```cpp
template <typename Scheme, uint32_t StepIndex>
inline void run_one_step(...) {
    constexpr auto step = Scheme::steps[StepIndex];

    if constexpr (step.type == StepType::kPredict) {
        run_predict_update_step<Scheme, StepIndex>(...);
    }

    if constexpr (step.type == StepType::kUpdate) {
        run_predict_update_step<Scheme, StepIndex>(...);
    }

    if constexpr (step.type == StepType::kScaleEven) {
        run_scale_step<Scheme, StepIndex>(...);
    }

    if constexpr (step.type == StepType::kScaleOdd) {
        run_scale_step<Scheme, StepIndex>(...);
    }

    if constexpr (step.type == StepType::kSwap) {
        swap_stream_refs(...);
    }
}
```

Тобто predict, update, scale, swap стають **branches at compile time**, а не runtime `switch`.

---

## 6. Але є важлива пастка: різні `k` у різних steps

У схемах один step може мати 1 coefficient, інший 3, інший 5. Поточний compute kernel бере один `k_stencil_width` як compile-time arg і викликає `run_step<k_stencil_width>()`.

Тому є два варіанти.

### Варіант

Кожен step має власне compile-time `k`:

```cpp
run_step<step.k>(...);
```

Але `step.k` має бути справжнім `constexpr`, і для цього краще робити template-recursion / `index_sequence`.

Концептуально:

```cpp
template <typename Scheme, size_t I>
inline void run_static_step(...) {
    constexpr auto step = Scheme::steps[I];

    if constexpr (step.type == StepType::kPredict || step.type == StepType::kUpdate) {
        run_stencil_step<step.k>(step.coeffs, ...);
    } else if constexpr (step.type == StepType::kScaleEven || step.type == StepType::kScaleOdd) {
        run_scale_step(step.coeffs[0], ...);
    }
}
```

Це краще для performance, але складніше для TT kernel compiler.

---

## 7. Що робити з `delay`, `shift`, geometry

`delay_even` / `delay_odd` — це не device padding, а початковий logical origin для even/odd stream. У твоєму TT_LWT.md це вже правильно описано: delay стає initial `StreamState.shift`, а потім кожен step комбінується зі своїм `shift` у step geometry.

Тому в compile-time схемі мають бути:

```cpp
static constexpr int32_t delay_even;
static constexpr int32_t delay_odd;
```

А от geometry залежить від runtime signal length, тому вона не вся compile-time.

Я б зробив так:

```cpp
template <typename Scheme>
LiftingForwardPlan make_forward_lifting_plan_static(...) {
    StreamState even_state{.shift = Scheme::delay_even, .length = even_ping.length};
    StreamState odd_state{.shift = Scheme::delay_odd, .length = odd_ping.length};

    static_for<Scheme::num_steps>([&](auto I) {
        constexpr auto step = Scheme::steps[I];
        ...
    });
}
```

Тобто **step metadata compile-time**, але lengths/addresses/runtime buffers лишаються runtime.

---

## 8. Як прибрати DRAM loopback

Тут найважливіша частина.

Якщо зараз кожен крок пише результат у DRAM, а наступний крок знову читає з DRAM, то “одна програма” має замінити це на:

```text
input DRAM
   ↓
pad/split
   ↓
even_ping / odd_ping in L1
   ↓
predict/update/scale loop in L1 ping-pong buffers
   ↓
final export to DRAM
```

Тобто треба мати:

```text
even_ping
even_pong
odd_ping
odd_pong
```

і кожен step міняє тільки stream references:

```text
Predict:
    odd_out = odd_in + stencil(even_in)
    odd slot toggles
    even unchanged

Update:
    even_out = even_in + stencil(odd_in)
    even slot toggles
    odd unchanged

ScaleEven:
    even_out = scalar * even_in
    even slot toggles

ScaleOdd:
    odd_out = scalar * odd_in
    odd slot toggles

Swap:
    swap logical even/odd refs
```

Це гарно лягає на твою вже існуючу `ping/pong` логіку: у `plan.cpp` є `toggle_slot(...)` і `with_toggled_slot(...)`.

---

## 9. Важливе обмеження

Повністю без DRAM loopback просто так вийде тільки якщо intermediate streams влазять у L1.


---

## 10. Найкращий практичний дизайн

Я б зробив таку структуру:

```text
include/tt_wavelet/
  lifting/
    static_scheme.hpp
    static_registry.hpp
    static_lwt.hpp

  schemes/generated/
    bior1_3.hpp
    bior2_2.hpp
    coif2.hpp
    registry.hpp

kernels/
  compute/
    lwt_fused_compute_template.hpp
    generated/
      lwt_fused_bior2_2_compute.cpp
      lwt_fused_coif2_compute.cpp

  dataflow/
    lwt_fused_reader.cpp
    lwt_fused_writer.cpp
```

Per-scheme compute wrapper:

```cpp
#include "kernels/compute/lwt_fused_compute_template.hpp"
#include "tt_wavelet/schemes/generated/bior2_2.hpp"

void kernel_main() {
    ttwv::kernels::lwt_fused_compute<ttwv::schemes::bior2_2>();
}
```

Host side:

```cpp
template <typename Scheme>
Tensor lwt(const Tensor& signal) {
    auto plan = make_forward_lifting_plan_static<Scheme>(signal);
    auto program = create_fused_lwt_program<Scheme>(plan);
    run(program);
    return read_result(...);
}
```

Runtime API:

```cpp
Tensor lwt(const Tensor& signal, std::string_view name) {
    switch (scheme_id(name)) {
        case SchemeId::kBior2_2:
            return lwt<schemes::bior2_2>(signal);
        case SchemeId::kCoif2:
            return lwt<schemes::coif2>(signal);
    }

    TT_THROW("Unknown wavelet scheme: {}", name);
}
```

---

## 11. Що я б робив поетапно

### Step 1

Прибрати runtime JSON з normal path.

Зробити generated constexpr scheme headers.

JSON loader залишити тільки для tests / debug / regeneration.

---

### Step 2

Зробити public registry:

```cpp
ttwv::available_wavelets();
ttwv::lwt(signal, "bior2.2");
ttwv::lwt<ttwv::schemes::bior2_2>(signal);
```

---

### Step 3

Злити scale у той самий step loop.

Навіть якщо спочатку коефіцієнти ще runtime, прибери окремий `lwt_scale_compute.cpp` path.

---

### Step 4

Зробити fused Program:

```text
one reader
one compute
one writer
one launch
all steps inside
```

На першій версії можна ще використовувати DRAM ping-pong, але без окремих program launches.

---

### Step 5

Перенести ping-pong intermediate buffers у L1.

Це вже буде справжнє “без DRAM loopback”.

---


For documentation you may check

- [text](docs/DEFINITIONS.md)

- [text](docs/HORIZONTAL_STENCIL.md)

- [text](docs/LWT.md)

- [text](docs/RECOVER1.md)

- [text](docs/RECOVER2.md)

- [text](docs/SHIFT.md)

- [text](docs/SPLICE.md)

- [text](docs/VERTICAL_STENCIL.md)


The math model is described in this code

- [text](dtypes.py)
- [text](lifting2.py)
- [text](test.py)

There is also a tt-isa documentation

- [text](tt-isa-documentation)


Also about pad_split it is uses sharded tensor but you may remove it and keep fetching by sticks of 32 elements

To run the code
build:
```
./update.sh
```

Compare test

```
python3 compare.py
```

The lifting schemes in this dir [schemes](ttnn-wavelet/lifting_schemes)

So the whole plan with json should be removed(deleted from code)
and instead of it we should have compile time structures with the same data as in json but in c++ code and then we can generate these structures from json by python script