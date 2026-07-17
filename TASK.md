Так, **найправильніший шлях для 1D ILWT — використати той самий ConeStreamed + three-slot + narrow-tile engine**, але є критичне уточнення:

> Недостатньо просто розвернути JSON і передати його в `make_forward_lifting_plan()`.

Математичні steps справді можна інвертувати, але **inverse geometry треба будувати з forward trace**, тому що поточний forward використовує valid convolution, змінює shifts/lengths і наприкінці canonicalize-ить лише потрібну частину внутрішніх streams. Поточний forward planner явно зберігає `source_offset`, `base_offset`, `output_length`, shifts і lengths, а cone planner уже вміє оперувати інтервалами та трьома `A/B/Scratch` slots. 

# 1. Математична модель inverse lifting

Для будь-якого predict/update route поточний forward математично має вигляд:

[
Y[t]
====

B[b+t]
+
\sum_{j=0}^{K-1}h_jS[s+t+j],
]

де:

* (S) — source stream;
* (B) — base/target stream до update;
* (Y) — target stream після update;
* (s=\text{source_offset});
* (b=\text{base_offset}).

Для predict:

[
S=E,\qquad B=O,\qquad Y=O'.
]

Для update:

[
S=O,\qquad B=E,\qquad Y=E'.
]

Інверсія:

[
\boxed{
B[b+t]
======

## Y[t]

\sum_{j=0}^{K-1}h_jS[s+t+j]
}
]

або:

[
B[b+t]
======

Y[t]
+
\sum_{j=0}^{K-1}(-h_j)S[s+t+j].
]

Тобто твій існуючий SFPU kernel:

```text
output = base + convolution(source, coefficients)
```

можна використати без зміни arithmetic structure:

```text
inverse output = current updated stream
               + convolution(unchanged stream, -coefficients)
```

## Що змінюється в steps

| Forward step        | Inverse step         |
| ------------------- | -------------------- |
| `predict(h, shift)` | `predict(-h, shift)` |
| `update(h, shift)`  | `update(-h, shift)`  |
| `scale-even(a)`     | `scale-even(1/a)`    |
| `scale-odd(a)`      | `scale-odd(1/a)`     |
| `swap`              | `swap`               |

І steps виконуються у зворотному порядку.

Важливо:

* **порядок coefficients усередині stencil не розвертається**;
* `shift` залишається тим самим;
* змінюється лише знак predict/update coefficients;
* обертається порядок самих steps.

Це тому, що inverse використовує той самий оператор (P) або (U), а не convolution із reversed filter.

---

# 2. Твій приклад `bior`

Позначимо три lifting operators:

[
P_1(E)
======

-\frac13E[\cdot+\tau_1],
]

[
U(O)
====

-1.125O[\cdot]
-0.375O[\cdot+1],
]

і дев’ятитаповий оператор (P_9(E)).

## Forward

[
O_1=O_0+P_1(E_0),
]

[
E_1=E_0+U(O_1),
]

[
O_2=O_1+P_9(E_1).
]

Після swap:

[
F_e=O_2,
\qquad
F_o=E_1.
]

Після scale:

[
A=2.121320343559643,F_e,
]

[
D=-0.4714045207910317,F_o.
]

## Inverse

Спочатку reciprocal scales:

[
F_e=\frac{A}{2.121320343559643},
]

[
F_o=\frac{D}{-0.4714045207910317}.
]

Undo swap:

[
E_1=F_o,
\qquad
O_2=F_e.
]

Потім steps у зворотному порядку:

[
O_1=O_2-P_9(E_1),
]

[
E_0=E_1-U(O_1),
]

[
O_0=O_1-P_1(E_0).
]

І наприкінці:

[
x_{\text{padded}}[2j]=E_0[j],
]

[
x_{\text{padded}}[2j+1]=O_0[j].
]

Після цього відкидається physical padding.

Для `tap_size=20`:

[
P=\text{tap_size}-1=19.
]

Тому reconstructed original signal:

[
\boxed{
x[n]
====

\begin{cases}
E_0\left[\dfrac{n+19}{2}\right],
& n+19\text{ парне},[6pt]
O_0\left[\dfrac{n+18}{2}\right],
& n+19\text{ непарне}.
\end{cases}
}
]

---

# 3. Чому не можна просто викликати forward planner для inverse scheme

Поточний forward geometry визначається через:

[
\text{conv_shift}
=================

\text{source.shift}
+
\text{kernel.shift}
+
K-1,
]

і valid convolution:

[
L_{\text{conv}}
===============

L_{\text{source}}-K+1.
]

Після цього перетинаються valid regions source convolution і base stream. Через це проміжні streams можуть ставати коротшими.

Inverse, навпаки, повинен відновлювати попередній stream region. Якщо просто передати reversed scheme у `compute_step_geometry()`, він знову виконуватиме valid contraction:

```text
final coefficients
→ ще коротший stream
→ ще коротший stream
```

замість reconstruction.

Тому:

[
\boxed{
\text{inverse coefficients можна генерувати зі scheme,}
\quad
\text{але inverse geometry треба отримувати з forward route trace.}
}
]

---

# 4. Найважливіша проблема: canonical coefficients

Forward не повертає весь internal final stream. Він робить canonicalization через:

[
\delta
======

## \text{canonical_start}

\text{stream_shift},
]

де:

[
\text{canonical_start}
======================

\left\lfloor\frac{\text{tap_size}}2\right\rfloor.
]

Поточний код копіює лише canonical interval довжини `output_length`, використовуючи `final_even_shift` і `final_odd_shift`. 

Тому inverse reader не повинен вважати:

```text
approximation[0] == internal_final_even[0]
detail[0]        == internal_final_odd[0]
```

Загальне відображення:

[
c=i+\text{stream_shift}-\text{canonical_start},
]

де:

* (i) — internal final stream index;
* (c) — canonical coefficient index.

Тобто inverse planner спочатку знаходить необхідний internal interval, а потім переводить його в canonical A/D coordinates.

Для interval:

[
I=[i_0,i_1)
]

canonical interval:

[
\boxed{
C=
[
i_0+s-c_0,;
i_1+s-c_0
)
}
]

де:

[
s=\text{final stream shift},
\qquad
c_0=\text{tap_size}/2.
]

Planner повинен перевірити:

[
C\subseteq[0,M),
]

де (M) — canonical coefficient length.

---

# 5. Хороша новина: discarded internal boundaries не обов’язково відновлювати

Це було моє головне побоювання: forward valid steps відкидають частину internal streams, а canonicalization обрізає їх ще раз.

Але для inverse не треба реконструювати весь padded stream. Треба лише ту частину initial even/odd streams, яка відповідає original signal.

## Приклад (N=100)

Для `tap_size=20`:

[
P=19.
]

Original signal займає padded interval:

[
[19,119).
]

Необхідні initial split intervals:

[
E_0=[10,60),
]

[
O_0=[9,59).
]

Якщо провести inverse dependency analysis через твою `bior` scheme, потрібні final internal intervals:

[
F_e=[4,55),
]

[
F_o=[4,63).
]

Final shifts:

[
s_e=10,
\qquad
s_o=6,
]

а:

[
c_0=10.
]

Тому canonical coefficient intervals:

[
A=[4,55),
]

[
D=[0,59).
]

Canonical coefficient length:

[
M=
\left\lfloor
\frac{N+\text{tap_size}-1}{2}
\right\rfloor
=============

59.

]

Отже всі необхідні coefficients лежать у доступному canonical output:

[
A\subseteq[0,59),
\qquad
D=[0,59).
]

Тобто **не треба вигадувати значення discarded internal boundaries**. Треба лише правильно побудувати inverse cone від original output region.

Це обов’язково треба перевірити для всіх 106 schemes:

```cpp
TT_FATAL(
    contains(canonical_approx_interval, required_final_even),
    "Inverse cone requires non-canonical approximation coefficients");

TT_FATAL(
    contains(canonical_detail_interval, required_final_odd),
    "Inverse cone requires non-canonical detail coefficients");
```

---

# 6. Exact inverse dependency cone

Нехай ми хочемо reconstructed signal chunk:

[
X=[a,b).
]

## 6.1. Переведення у padded split coordinates

[
p_0=a+P,
\qquad
p_1=b+P,
]

де:

[
P=\text{tap_size}-1.
]

Тоді target even interval:

[
\boxed{
R_E^{(0)}
=========

\left[
\left\lceil\frac{p_0}{2}\right\rceil,
\left\lceil\frac{p_1}{2}\right\rceil
\right)
}
]

Target odd interval:

[
\boxed{
R_O^{(0)}
=========

\left[
\left\lceil\frac{p_0-1}{2}\right\rceil,
\left\lceil\frac{p_1-1}{2}\right\rceil
\right)
}
]

Це та частина initial split streams, яку реально треба reconstructed.

---

## 6.2. Прохід через forward predict

Forward route:

[
O_{\text{after}}[t]
===================

O_{\text{before}}[b+t]
+
P(E_{\text{before}})[s+t].
]

Маємо required intervals **до** forward route:

[
R_E^{\text{before}},
\qquad
R_O^{\text{before}}.
]

Для потрібної частини old odd:

[
T=
R_O^{\text{before}}-b.
]

Тоді після route потрібні:

[
\boxed{
R_O^{\text{after}}=T
}
]

[
\boxed{
R_E^{\text{after}}
==================

R_E^{\text{before}}
\cup
\left(
T+s+[0,K)
\right)
}
]

Обов’язкова перевірка:

[
R_O^{\text{before}}
\subseteq
[b,b+L_{\text{output}}).
]

Інакше forward route відкинув значення, яке inverse намагається відновити.

---

## 6.3. Прохід через forward update

Forward:

[
E_{\text{after}}[t]
===================

E_{\text{before}}[b+t]
+
U(O_{\text{before}})[s+t].
]

Тоді:

[
T=
R_E^{\text{before}}-b,
]

[
\boxed{
R_E^{\text{after}}=T
}
]

[
\boxed{
R_O^{\text{after}}
==================

R_O^{\text{before}}
\cup
\left(
T+s+[0,K)
\right)
}
]

---

## 6.4. Scale і swap

Scale:

[
R^{\text{after}}=R^{\text{before}}.
]

Swap:

[
R_E^{\text{after}}=R_O^{\text{before}},
]

[
R_O^{\text{after}}=R_E^{\text{before}}.
]

---

# 7. Напрямок проходу planner-а

Це може трохи плутати.

## Forward cone

Ми починаємо з final coefficients і йдемо **назад через forward routes**:

```text
final coefficient chunk
→ required initial even/odd cone
```

## Inverse cone

Ми починаємо з reconstructed original chunk і питаємо, які final coefficients потрібні.

Inverse execution має reversed routes, тому dependency planner може пройти **forward routes у звичайному порядку**:

```text
required initial E/O output
→ route 0
→ route 1
→ ...
→ required final internal streams
→ canonical A/D intervals
```

Тобто можна мати:

```cpp
std::vector<RequiredStreams> propagate_inverse_requirements(
    const LiftingForwardPlan& forward_plan,
    IndexInterval target_initial_even,
    IndexInterval target_initial_odd);
```

Це майже дзеркальна версія поточного `backpropagate_requirements()`. Поточний cone planner уже має interval hull, translations, storage tracking і local offsets, тому більшість host-side інфраструктури можна reuse. 

---

# 8. Побудова inverse routes

Після propagation матимемо:

```text
required[0] = reconstructed initial E/O interval
required[1] = state after forward route 0
...
required[m] = required final internal streams
```

Inverse execution починається зі стану:

[
required[m]
]

і виконує routes:

[
m-1,m-2,\ldots,0.
]

Для forward predict route:

```cpp
const auto target_before = required[i].odd;
const auto t = subtract(target_before, forward_route.base_offset);

source_required =
    translate(t, forward_route.source_offset, K - 1);

base_required = t;
output = target_before;
coefficients = negate(forward_coefficients);
```

Фізично:

```text
source = current even-after
base   = current odd-after
output = reconstructed odd-before
```

Для update:

```text
source = current odd-after
base   = current even-after
output = reconstructed even-before
```

Three-slot transition не змінюється:

```cpp
const StorageSlot released = active_target.slot;
active_target = StreamRef{.slot = free_slot};
free_slot = released;
```

Swap знову metadata-only.

---

# 9. Що можна reuse майже без змін

## Повністю reuse

### SFPU stencil

Поточний `horizontal_stencil_sfpi.h` уже обчислює:

[
g=base+\sum h_j source_j
]

у FP32. Для inverse потрібні лише negated coefficients. Внутрішня narrow-tile geometry і `17-K` packing залишаються такими самими, оскільки (K) не змінюється. 

### Narrow `32x16` compute

Поточна схема:

```text
4 source pages
+ 3 base pages
→ 7 narrow DST tiles
→ SFPU
→ 3 output pages
```

повністю підходить для inverse predict/update. 

### `A/B/Scratch`

Та сама three-slot state machine.

### Row-major і tile-native workspace

Обидва layouts можна reuse. Але auto policy треба рахувати окремо для inverse routes — не можна автоматично використовувати forward вибір. Поточний tile-native layout вигідний лише коли route intervals добре вирівняні; shifted intervals усе ще потребують remap. 

### Chunk scheduler

Та сама ідея:

```text
one core → one or several independent inverse output chunks
```

Без global shard staircase.

---

# 10. Що треба написати окремо

## 10.1. Inverse input reader

Forward reader приймає один raw signal і виконує:

* symmetric mapping;
* physical padding;
* even/odd split;
* cone initialization.

Inverse reader прийматиме два buffers:

```text
approximation DRAM
detail DRAM
```

І завантажуватиме canonical intervals:

```cpp
approx_begin;
approx_length;
detail_begin;
detail_length;
```

У локальні slots A/B.

Цей reader буде простішим:

* не потрібен `symmetric_index`;
* не потрібен pad/split;
* не потрібне чергування parity;
* лише contiguous coefficient interval → selected workspace layout.

---

## 10.2. Inverse final writer

Forward має два final coefficient buffers.

Inverse має один reconstructed output buffer.

Writer повинен:

1. взяти reconstructed (E_0/O_0);
2. відкинути padding;
3. interleave-ити;
4. записати contiguous FP32 sticks у DRAM.

Для кожного (n):

```cpp
const uint32_t padded_index = left_pad + output_index;

if ((padded_index & 1U) == 0) {
    value = even[padded_index / 2];
} else {
    value = odd[(padded_index - 1) / 2];
}
```

Краще формувати один 32-FP32 stick у staging CB і робити один 128-byte NoC write.

Це unavoidable final (O(N)) pass, але він виконується лише один раз.

---

# 11. Найкраща optimization для inverse

Forward отримав великий виграш від terminal direct-to-DRAM.

Для inverse аналогічна оптимізація буде іншою.

## Наївний inverse

```text
last inverse predict/update
→ materialize final reconstructed stream in L1
→ separate interleave writer
→ DRAM
```

## Кращий inverse

Останній inverse predict/update реконструює лише один із потоків:

* predict реконструює (O_0);
* update реконструює (E_0).

Другий final stream уже resident.

Тому можна зробити:

```text
last inverse route output CB ─────┐
                                 ├→ fused interleave/crop writer → DRAM
other final resident stream ─────┘
```

Наприклад, у твоєму `bior` остання inverse operation:

[
O_0=O_1-P_1(E_0).
]

`E0` уже resident, а `O0` щойно з’являється в output CB.

Writer може одразу interleave:

```text
resident E0
+
CB O0
→ original signal DRAM
```

Таким чином не буде:

* final O0 L1 materialization;
* додаткового route barrier;
* повторного читання O0 з workspace.

Це фактично inverse-аналог terminal direct-to-DRAM.

---

# 12. Що робити з inverse scales

Forward terminal scales після reversing стають першими steps:

```text
inverse scale odd
inverse scale even
swap
inverse predict/update...
```

Для першої правильної реалізації я б залишив їх як два SFPU scale routes. Це максимально reuse і зберігає FP32 semantics.

Пізніше можна дослідити fusion.

## Можливий affine-stencil fusion

Якщо перший inverse predict/update використовує streams, які ще scaled:

[
B_{\text{old}}
==============

\beta B_{\text{input}}
+
\sum_j\gamma_jS_{\text{input}},
]

де:

[
\beta=\frac1{s_B},
\qquad
\gamma_j=-\frac{h_j}{s_S}.
]

Тоді SFPI kernel можна розширити з:

[
g=base+\sum h_js_j
]

до:

[
g=\beta,base+\sum \gamma_js_j.
]

Це потенційно прибере обидва initial scale routes, але не варто починати з цього. Спочатку correct generic inverse.

---

# 13. Генерація inverse scheme

Не треба створювати окремі JSON вручну.

Оригінальний JSON залишається source of truth.

Generator повинен emit-ити:

```cpp
ForwardScheme
InverseScheme
```

Inverse steps:

```python
for step in reversed(forward_steps):
    if step.type in ("predict", "update"):
        inverse_coefficients = [-fp32(c) for c in step.coefficients]
        inverse_shift = step.shift

    elif step.type in ("scale-even", "scale-odd"):
        s = fp32(step.coefficients[0])
        inverse_coefficient = fp32(1.0 / s)

    elif step.type == "swap":
        unchanged
```

Важливо для roundtrip:

[
s_{\text{inv}}
==============

\operatorname{fp32}
\left(
\frac1{\operatorname{fp32}(s)}
\right),
]

а не reciprocal початкового JSON double.

Negation predict/update можна зробити точним sign-bit flip:

```cpp
inverse_bits = forward_bits ^ 0x80000000U;
```

---

# 14. Рекомендовані структури

```cpp
struct LiftingInversePlan {
    LiftingForwardPlan forward_trace;

    size_t original_length{0};
    size_t coefficient_length{0};

    std::vector<InverseStepRoute> routes;

    size_t initial_even_length{0};
    size_t initial_odd_length{0};
};
```

```cpp
struct InverseConeChunkPlan {
    IndexInterval output_signal;

    IndexInterval reconstructed_even;
    IndexInterval reconstructed_odd;

    IndexInterval canonical_approximation;
    IndexInterval canonical_detail;

    std::vector<ConeStepRoute> routes;

    size_t max_workspace_elements{0};
    double dependency_overhead{0.0};

    bool fused_final_interleave{false};
};
```

API:

```cpp
create_cone_streamed_ilwt_executable<Scheme>(
    approximation_buffer,
    detail_buffer,
    coefficient_length,
    original_length);
```

`original_length` має бути обов’язковим. За coefficient length не завжди можна однозначно визначити початковий (N).

---

# 15. Розмір chunk

Поточна група одного stream:

[
1536
]

елементів.

Оскільки reconstructed signal interleave-ить два streams, природний output chunk:

[
\boxed{
C_x
===

# 2\cdot1536\cdot G

3072G
}
]

original samples.

Це дає:

* приблизно 1536G even values;
* приблизно 1536G odd values;
* добру відповідність narrow-tile group geometry;
* максимум один елемент різниці через parity та odd (N).

---

# 16. Performance model

Для output chunk довжини (C):

[
|E_{\text{target}}|+|O_{\text{target}}|
\approx C.
]

Нехай inverse cone потребує:

[
|A_{\text{req}}|+|D_{\text{req}}|.
]

Dependency overhead:

[
\boxed{
\eta_{\text{ILWT}}
==================

\frac{
|A_{\text{req}}|
+
|D_{\text{req}}|
----------------

C
}{C}
}
]

External traffic приблизно:

[
4(|A_{\text{req}}|+|D_{\text{req}}|)
+
4C
]

bytes.

Усі inverse intermediates залишаються в local L1:

```text
A/D DRAM
→ required coefficient cone
→ local A/B/Scratch
→ all inverse steps
→ interleave/crop
→ reconstructed DRAM
```

Без full upsampled streams і без DRAM loopback.

---

# 17. Correctness invariants

Перед device implementation planner повинен пройти всі 106 schemes і довести:

1. Кожен target-before interval лежить у reconstructable base region:

[
R_{\text{target}}
\subseteq
[b,b+L_{\text{output}}).
]

2. Required final internal intervals повністю представлені canonical A/D buffers.

3. Output chunks без gaps і overlaps покривають:

[
[0,N).
]

4. Reconstructed E/O indices не виходять за initial padded split lengths.

5. Scale coefficients ненульові.

6. Inverse workspace fits L1 budget.

7. Row-major і tile-native дають bitwise-identical output.

---

# 18. Тести

Обов’язково розділити:

## Внутрішній roundtrip

[
x
\rightarrow
\operatorname{TT_LWT}
\rightarrow
\operatorname{TT_ILWT}
\rightarrow
\hat x.
]

Перевіряти:

[
|x-\hat x|_\infty.
]

## Interoperability

[
\operatorname{TT_ILWT}
(
\operatorname{PyWavelets_LWT}(x)
)
]

і:

[
\operatorname{PyWavelets_ILWT}
(
\operatorname{TT_LWT}(x)
).
]

## Edge cases

* even/odd (N);
* (N=1,2,\dots);
* (N<\text{tap_size});
* first/last chunk;
* chunk boundary;
* swaps;
* positive/negative shifts;
* (K=1,2,17);
* all 106 schemes.

Через FP32 lifting factorization високопорядкові schemes можуть мати більшу roundtrip похибку. Це треба відділяти від architecture correctness.

---

# Рекомендований порядок реалізації

## Фаза 1 — CPU inverse model

* Generate inverse steps.
* Build forward trace.
* Implement inverse interval propagation.
* Reconstruct signal на CPU через exact route geometry.
* Validate всі schemes і lengths.

## Фаза 2 — простий ConeStreamed ILWT

Reuse:

* narrow SFPU compute;
* `A/B/Scratch`;
* row-major/tile-native workspace;
* chunk scheduler;
* route config.

Додати:

* dual coefficient reader;
* inverse cone planner;
* final interleave writer.

Спочатку materialize обидва final E/O streams.

## Фаза 3 — fused final interleave

Останній inverse predict/update пише output у CB, а writer interleave-ить його з другим resident stream прямо в final DRAM.

## Фаза 4 — inverse scale fusion

Лише після профілювання розглядати affine-stencil fusion.

---

# Основний висновок

Твоя фінальна архітектура має виглядати так:

```text
canonical approximation/detail in DRAM
                ↓
      inverse coefficient cone reader
                ↓
       local A / B / Scratch
                ↓
 reciprocal scales in FP32 SFPU
                ↓
 reversed swap/predict/update chain
                ↓
 fused final interleave + crop
                ↓
       original signal in DRAM
```

І найбільш важливе архітектурне рішення:

[
\boxed{
\text{Inverse arithmetic береться з reversed scheme,}
\quad
\text{але inverse geometry будується з forward route trace.}
}
]

Саме це дозволить максимально використати поточний дуже швидкий ConeStreamed/narrow-tile backend і не зламатися на valid cropping, shifts та canonical coefficient layout.
