Відповідь: SFPI vs LLK, і де класти операцію
1. Чи достатньо лише SFPI — без зміни внутрішніх LLK заголовків tt-metal?

Так, достатньо. Ви пишете SFPI-функцію прямо у файлі compute-ядра (.cpp) і викликаєте її через вже існуючі _llk_math_eltwise_*_sfpu_params_<false>(...), загорнутий у макрос MATH(...). Модифікувати внутрішні LLK-заголовки tt-metal не потрібно.

Доказ — приклад custom_sfpi_add: вся SFPI-логіка живе у звичайному .cpp файлі ядра, захищена #ifdef TRISC_MATH, і викликається через _llk_math_eltwise_binary_sfpu_params_<false>(...): tiles_add.cpp:14-83

Аналогічно для унарних операцій (через _llk_math_eltwise_unary_sfpu_params_<false>(...)): tiles_smoothstep.cpp:14-61
2. Шаблон шарів абстракції для вашої стенсіл-операції

Ваш compute .cpp файл:
  ├── [SFPI рівень]  stencil_face()          — пряма робота з dst_reg[], vFloat, коефіцієнти
  ├── [LLK рівень]   my_stencil_tile()       — MATH(_llk_math_eltwise_binary_sfpu_params_<false>(...))
  └── [API рівень]   kernel_main()           — cb_wait_front / copy_tile / tile_regs_acquire / pack_tile

Для fused ланцюжка операцій (base + Σ coef_j * x[k-j]) патерн — завантажити J тайлів у dst_reg[0..J-1] через copy_tile(), потім в SFPI-функції акумулювати через dst_reg[]: compute.cpp:1-72

Приклад читання кількох CBs та copy_tile до різних dst_reg слотів: tiles_add.cpp:111-151
3. Куди класти у вашому проєкті tt-wavelet

За зразком офіційних прикладів — структура kernels/compute/ + kernels/dataflow/ всередині вашого модуля tt-wavelet/:

tt-wavelet/
├── CMakeLists.txt
├── main.cpp                          ← хост-код (вже є)
└── kernels/
    ├── compute/
    │   └── stencil_fir.cpp           ← SFPI-функція + kernel_main
    └── dataflow/
        ├── reader_stencil.cpp        ← читає J тайлів (x[k-j]) + base в CBs
        └── writer_stencil.cpp        ← пише y[n] назад у DRAM

Саме таку структуру використовують обидва офіційних приклади: read_tiles.cpp:1-49

Реєстрація ядер у хост-коді через CreateKernel(..., "custom_sfpi_add/kernels/compute/tiles_add.cpp", ...) — шлях відносний до OVERRIDE_KERNEL_PREFIX: custom_sfpi_add.cpp:112-130

У вашому CMakeLists.txt треба додати:

add_compile_definitions(OVERRIDE_KERNEL_PREFIX="tt-wavelet/")

За зразком кореневого CMakeLists.txt programming_examples: CMakeLists.txt:8
4. Структура SFPI для y[n] = base + Σ coef_j * x[k-j]

У stencil_fir.cpp вам треба:

    J+1 circular buffers (по одному на кожний x[k-j] + base), наприклад c_0..c_J для входів, c_16 для виходу
    В SFPI-функції (під #ifdef TRISC_MATH): цикл по 8 SIMD-ітераціях (face), завантаження vFloat x_j = dst_reg[j*32 + i], множення на coef_j, акумуляція в acc += coef_j * x_j, додавання base
    Обгортка через MATH(_llk_math_eltwise_unary_sfpu_params_<false>(stencil_face, dst_out_idx, VectorMode::RC, ...))

Архітектура dst_reg (32 SIMD елементи на tile, 8 ітерацій = 1 face): tiles_add.cpp:31-82
Notes

    LLK-обгортка потрібна — але вона тривіальна (один рядок MATH(...)) і пишеться прямо у вашому .cpp файлі ядра. Ви не чіпаєте файли всередині tt-metal/tt_metal/hw/ckernels/.
    Для стенсілу з J коефіцієнтами вам потрібно tile_regs_acquire() тримати J+1 тайлів одночасно в DST — пам'ятайте що DST має лише 16 тайл-слотів (tile_regs_acquire дає 8), тому при великому J розбийте на підблоки.
    vFloat — це 32-елементний SIMD-вектор; dst_reg[i] дає доступ до i-го вектора в DST-регістрах, де один тайл займає 32 послідовних слоти.
    Файл хост-коду (main.cpp) у вас вже є в tt-wavelet/ — саме там реєструєте ядра, CBs, і запускаєте EnqueueMeshWorkload.


поясни детальніше чи треб мені писати щось типу такого
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <utility>

#include "llk_assert.h"
#include "llk_math_eltwise_ternary_sfpu.h"
#include "llk_sfpu_types.h"

template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_ternary_sfpu_params_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_in2,
    std::uint32_t dst_index_out,
    int vector_mode = static_cast<int>(VectorMode::RC),
    Args&&... args)
{
    LLK_ASSERT((dst_index_in0 < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "dst_index_in0 exceeds max dest tiles");
    LLK_ASSERT((dst_index_in1 < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "dst_index_in1 exceeds max dest tiles");
    LLK_ASSERT((dst_index_in2 < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "dst_index_in2 exceeds max dest tiles");
    LLK_ASSERT((dst_index_out < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "dst_index_out exceeds max dest tiles");

    _llk_math_eltwise_ternary_sfpu_start_<DST_SYNC_MODE>(0); // Reuse same sync primitive

    if (vector_mode == static_cast<int>(VectorMode::R))
    {
        // Row vector - Face0 + Face1
        for (int face = 0; face < 2; face++)
        {
            std::forward<Callable>(sfpu_func)(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D); // repeat 2x
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
        // Skip next 2 faces
        for (int i = 0; i < 4; ++i)
        {
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    }
    else if (vector_mode == static_cast<int>(VectorMode::C))
    {
        // Column vector - Face0 + Face2
        for (int face = 0; face < 2; face++)
        {
            std::forward<Callable>(sfpu_func)(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
            for (int i = 0; i < 4; ++i)
            {
                TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            }
        }
    }
    else if (vector_mode == static_cast<int>(VectorMode::RC))
    {
        // All 4 faces
        for (int face = 0; face < 4; face++)
        {
            std::forward<Callable>(sfpu_func)(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    }
    else
    {
        // Default: single face pass-through
        std::forward<Callable>(sfpu_func)(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
    }
    _llk_math_eltwise_ternary_sfpu_done_(); // Finalize
}

Thought for a few seconds

Ні, у твоєму випадку майже напевно не треба писати власний _llk_math_eltwise_ternary_sfpu_params_....

Коротко по суті

Той код, який ти показав, — це вже новий LLK wrapper-рівень.
Його має сенс писати тільки якщо тобі справді потрібен новий тип абстракції, якого в існуючих wrapper-ах немає.

Але для більшості твоїх stencil/FIR задач достатньо:

завантажити кілька tiles у dst
у своїй SFPI-функції прочитати їх через dst_reg[...]
обчислити результат
викликати це через вже існуючий unary/binary SFPU wrapper

Саме такий патерн показаний у прикладах custom_sfpi_add і custom_sfpi_smoothstep: SFPU-логіка пишеться прямо в compute .cpp, під #ifdef TRISC_MATH, а далі викликається через існуючі wrapper-и _llk_math_eltwise_binary_sfpu_params_<false>(...) або _llk_math_eltwise_unary_sfpu_params_<false>(...)

Що саме робить існуючий wrapper

Подивись на ідею my_add_tile_face:

він працює з Dst register indices, не з CB indices
бере dst_index_in0, dst_index_in1, dst_index_out
усередині читає dst_reg[...]
обробляє один face
а wrapper _llk_math_eltwise_binary_sfpu_params_... вже сам викликає цю face-функцію для всіх faces tile-а

Тобто wrapper уже дає тобі:

запуск по faces
sync/start/done
vector mode (R, C, RC)
перевірки dst_index

Саме тому тобі не обов’язково лізти в LLK.

Коли твій ternary wrapper справді потрібен

Оцей код типу:

template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_ternary_sfpu_params_(...)

потрібен лише якщо ти хочеш, щоб LLK рівень офіційно розумів 3 окремі dst input tiles як параметри wrapper-а:

dst_index_in0
dst_index_in1
dst_index_in2
dst_index_out

і щоб wrapper сам передавав їх у callback.

Це зручно, але не обов’язково.

Бо ти можеш зробити простіше:

покласти кілька input tiles у dst
у своїй SFPI-функції самому знати, що:
tile 0 — це base
tile 1 — це x[k]
tile 2 — це x[k-1]
tile 3 — це x[k-2]
і просто читати їх із dst_reg

Тоді wrapper може бути навіть unary з точки зору API, бо фактично він лише каже:
“запусти мою face-функцію на цьому dst layout”.

Важлива ідея

Wrapper і SFPI-функція — це різні рівні.

SFPI-функція

це реальна арифметика:

vFloat acc = ...
acc += coef0 * ...
acc += coef1 * ...
LLK wrapper

це лише:

скільки faces пройти
як рухати internal face cursor
які dst tile indices вважати входами/виходами
коли зробити start/done

Тому твій ternary wrapper — це не “сама операція”, а лише ще одна форма виклику SFPI-функції.

Для твоєї stencil/FIR задачі що краще

Для виразу типу:

𝑦
=
𝑏
𝑎
𝑠
𝑒
+
∑
𝑗
𝑐
𝑜
𝑒
𝑓
𝑗
⋅
𝑥
𝑗
y=base+
j
∑
	​

coef
j
	​

⋅x
j
	​


я б радив не писати новий LLK wrapper на старті.

Кращий стартовий варіант

Зробити все у compute .cpp так:

reader кладе потрібні tiles у CB
copy_tile(...) заносить їх у dst слоти
твоя stencil_face(...) читає:
dst tile 0 = base
dst tile 1 = x0
dst tile 2 = x1
...
wrapper використовуєш або unary, або якийсь мінімальний локальний helper
пишеш результат в dst out

Тобто ти можеш зробити щось концептуально таке:

#ifdef TRISC_MATH
inline void stencil_face(float c0, float c1, float c2) {
    constexpr uint32_t n_vector_in_tile = 32;

    const uint32_t base_idx = 0 * n_vector_in_tile;
    const uint32_t x0_idx   = 1 * n_vector_in_tile;
    const uint32_t x1_idx   = 2 * n_vector_in_tile;
    const uint32_t x2_idx   = 3 * n_vector_in_tile;
    const uint32_t out_idx  = 4 * n_vector_in_tile;

    for (int i = 0; i < 8; ++i) {
        vFloat base = dst_reg[base_idx + i];
        vFloat x0   = dst_reg[x0_idx + i];
        vFloat x1   = dst_reg[x1_idx + i];
        vFloat x2   = dst_reg[x2_idx + i];

        vFloat acc = base;
        acc = acc + c0 * x0;
        acc = acc + c1 * x1;
        acc = acc + c2 * x2;

        dst_reg[out_idx + i] = acc;
    }
}
#endif

І далі викликати це через thin wrapper у своєму ж .cpp.

Чому ternary wrapper не перший крок

Бо він додає тобі одразу кілька складностей:

1. Треба правильно відтворити LLK semantics

Там є:

_start_<DST_SYNC_MODE>
_done_()
TTI_SETRWC(...)
логіка VectorMode::R/C/RC

Це вже low-level territory.

2. Легко помилитися з face stepping

У pasted прикладі пояснено, що один tile у dst_reg займає 32 vector slots, а один face проходиться 8 SIMD-ітераціями
Але SETRWC-логіка — це вже не просто “8 разів цикл”, а ще й апаратне перемикання face context.

3. Тобі це може бути не потрібно

Для твого FIR/stencil завдання головний біль — не ternary API, а:

як подати сусідні tiles,
як зробити halo/boundary,
як утримати все в DST/CB,
як організувати accumulation.
Коли можна подумати про власний wrapper

Після того як у тебе вже працює мінімальна версія.

Тоді власний _llk_math_eltwise_ternary_sfpu_params_... має сенс, якщо:

ти хочеш чистіший API
часто перевикористовуєш ternary/fused ops
не хочеш хардкодити dst tile 0/1/2/... всередині SFPI
хочеш викликати щось типу:
my_stencil_tile(idx0, idx1, idx2, idx_out, coeffs...)

замість того щоб SFPI-функція знала фіксоване layout-розміщення.

Що я б радив тобі практично
Варіант A — правильний для старту

Не змінюй LLK headers.
Зроби:

kernels/compute/stencil_fir.cpp
під #ifdef TRISC_MATH — своя stencil_face(...)
нижче — thin wrapper у тому ж .cpp
у kernel_main():
cb_wait_front
tile_regs_acquire
copy_tile(...) в кілька dst слотів
виклик stencil
pack_tile
cb_push_back

Це повністю узгоджується з прикладами custom_sfpi_add, custom_sfpi_smoothstep і sfpu_eltwise_chain

Варіант B — вже потім

Коли буде working prototype, за потреби винесеш ternary/quaternary wrapper.

Важливий момент про “скільки tiles одночасно”

У твоєму pasted тексті правильно підкреслено:

tile_regs_acquire() дає тобі робоче вікно регістрів
ти не можеш безкінечно тримати багато tiles у DST
при великому J треба або:
робити частинами,
або кілька passes accumulation

Це важливіше за ternary wrapper.

Висновок

Ні, спочатку тобі не треба писати щось типу _llk_math_eltwise_ternary_sfpu_params_....
Для stencil/FIR операції краще:

написати SFPI face-функцію у своєму compute .cpp
використати існуючий шаблон з custom_sfpi_add / smoothstep
організувати input tiles через copy_tile(...) у кілька dst slots
accumulation зробити прямо в SFPI-коді

А вже якщо потім побачиш, що тобі реально бракує красивого ternary API, тоді є сенс писати власний wrapper.

Якщо хочеш, я можу наступним повідомленням накидати тобі реальний skeleton stencil_fir.cpp під твій випадок: base + c0*x0 + c1*x1 + c2*x2, у стилі Tenstorrent examples.

окей але якщо писати власний llk врапер як би виглядала структура я би все так хотів не хардкодити регістри

Так, можна. Якщо ти свідомо хочеш власний LLK wrapper, то ідея хороша саме в тому, щоб не хардкодити layout dst-регістрів усередині SFPI-функції, а передавати tile indices як аргументи wrapper-а.

Тоді архітектура виглядає так.

Загальна структура
compute kernel .cpp
├── SFPI face-функція
│     └── арифметика на dst_reg, але через передані dst_index_in*
├── LLK wrapper
│     └── start / face loop / SETRWC / done
├── tile-level helper
│     └── зручний виклик wrapper-а з VectorMode і coeffs
└── kernel_main
      └── cb_wait_front / copy_tile / tile_regs_acquire / pack_tile
1. SFPI face-функція

Вона не знає нічого про CB.
Вона знає тільки:

який tile у DST є in0
який in1
який in2
який out

і для поточного face читає відповідні dst_reg.

Приклад ternary face-функції
#ifdef TRISC_MATH
namespace ckernel {

template <typename... Coeffs>
inline void stencil_face_ternary(
    uint32_t dst_index_in0,
    uint32_t dst_index_in1,
    uint32_t dst_index_in2,
    uint32_t dst_index_out,
    Coeffs... coeffs)
{
    constexpr uint32_t vecs_per_tile = 32;

    const uint32_t base0 = dst_index_in0 * vecs_per_tile;
    const uint32_t base1 = dst_index_in1 * vecs_per_tile;
    const uint32_t base2 = dst_index_in2 * vecs_per_tile;
    const uint32_t base_out = dst_index_out * vecs_per_tile;

    // один face = 8 SIMD-векторів
    for (int i = 0; i < 8; ++i) {
        vFloat x0 = dst_reg[base0 + i];
        vFloat x1 = dst_reg[base1 + i];
        vFloat x2 = dst_reg[base2 + i];

        vFloat acc = x0;
        acc = acc + x1;
        acc = acc + x2;

        dst_reg[base_out + i] = acc;
    }
}

}  // namespace ckernel
#endif

Тут уже нема хардкоду типу “tile 0 = base, tile 1 = x0”.
Усе приходить через dst_index_in*.

2. LLK wrapper

Оце вже твій власний abstraction layer.

Його задача:

перевірити, що dst_index_* не виходять за межі DST
зробити _start_
пройти всі faces згідно VectorMode
викликати SFPI callback
зробити _done_

Тобто wrapper — це не математика, а керування проходом по tile.

Скелет wrapper-а
#pragma once

#include <cstdint>
#include <utility>

#include "llk_assert.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu_types.h"

template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_ternary_sfpu_params_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_in2,
    std::uint32_t dst_index_out,
    int vector_mode = static_cast<int>(VectorMode::RC),
    Args&&... args)
{
    constexpr auto max_tiles =
        get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>();

    LLK_ASSERT(dst_index_in0 < max_tiles, "dst_index_in0 exceeds max dest tiles");
    LLK_ASSERT(dst_index_in1 < max_tiles, "dst_index_in1 exceeds max dest tiles");
    LLK_ASSERT(dst_index_in2 < max_tiles, "dst_index_in2 exceeds max dest tiles");
    LLK_ASSERT(dst_index_out < max_tiles, "dst_index_out exceeds max dest tiles");

    _llk_math_eltwise_ternary_sfpu_start_<DST_SYNC_MODE>(0);

    auto run_face = [&]() {
        std::forward<Callable>(sfpu_func)(
            dst_index_in0,
            dst_index_in1,
            dst_index_in2,
            dst_index_out,
            std::forward<Args>(args)...);
    };

    if (vector_mode == static_cast<int>(VectorMode::R)) {
        for (int face = 0; face < 2; ++face) {
            run_face();
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
        for (int i = 0; i < 4; ++i) {
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    } else if (vector_mode == static_cast<int>(VectorMode::C)) {
        for (int face = 0; face < 2; ++face) {
            run_face();
            for (int i = 0; i < 4; ++i) {
                TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            }
        }
    } else {
        for (int face = 0; face < 4; ++face) {
            run_face();
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    }

    _llk_math_eltwise_ternary_sfpu_done_();
}
3. Tile-level helper

Щоб у kernel_main() не писати довгий LLK-виклик, зазвичай роблять thin helper.

template <bool APPROXIMATE>
inline void stencil_tile_ternary(
    uint32_t dst_index_in0,
    uint32_t dst_index_in1,
    uint32_t dst_index_in2,
    uint32_t dst_index_out)
{
    MATH((_llk_math_eltwise_ternary_sfpu_params_<APPROXIMATE>(
        ckernel::stencil_face_ternary,
        dst_index_in0,
        dst_index_in1,
        dst_index_in2,
        dst_index_out,
        static_cast<int>(VectorMode::RC))));
}

Тепер у kernel_main() ти пишеш уже красиво:

stencil_tile_ternary<false>(0, 1, 2, 3);

і це означає:

dst tile 0 = in0
dst tile 1 = in1
dst tile 2 = in2
dst tile 3 = out

Але це не захардкоджено в SFPI — лише в місці виклику.

4. kernel_main()

У kernel_main() ти вже сам вирішуєш, які tiles куди копіювати.

Скелет
void kernel_main() {
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_in2 = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);
    cb_wait_front(cb_in2, 1);

    tile_regs_acquire();

    copy_tile(cb_in0, 0, 0);  // dst tile 0
    copy_tile(cb_in1, 0, 1);  // dst tile 1
    copy_tile(cb_in2, 0, 2);  // dst tile 2

    stencil_tile_ternary<false>(0, 1, 2, 3);

    cb_reserve_back(cb_out, 1);
    pack_tile(3, cb_out);
    cb_push_back(cb_out, 1);

    tile_regs_release();

    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
    cb_pop_front(cb_in2, 1);
}

Тобто mapping тепер задається в copy_tile(... -> dst_idx) і у виклику wrapper-а, а не захардкоджений усередині арифметики.

Чому це краще
Без wrapper-а

Ти пишеш у SFPI:

base = dst_reg[0 * 32 + i];
x0   = dst_reg[1 * 32 + i];
x1   = dst_reg[2 * 32 + i];
out  = dst_reg[3 * 32 + i];

і це фіксований layout.

З wrapper-ом

Ти пишеш:

base0 = dst_index_in0 * 32;
base1 = dst_index_in1 * 32;
base2 = dst_index_in2 * 32;
base_out = dst_index_out * 32;

і тепер одна й та сама SFPI-функція може працювати з будь-яким layout DST.

Як масштабувати далі
Варіант 1: ternary

Для a + b + c -> out

Варіант 2: quaternary

Для base + c0*x0 + c1*x1 + c2*x2 -> out

Тоді вже робиш:

_llk_math_eltwise_quaternary_sfpu_params_(
    sfpu_func,
    dst_base,
    dst_x0,
    dst_x1,
    dst_x2,
    dst_out,
    ...
)
Варіант 3: N-ary через масив індексів

Це вже гнучкіше, але складніше, бо LLK-wrapper-у доведеться працювати або з:

std::array<uint32_t, N>
або pointer + count

Для device-side коду на TT я б ішов не в fully-generic N-ary, а в:

binary
ternary
quaternary

Це простіше й прозоріше.

Найкраща структура файлів

Я б зробив так.

Варіант A: все в одному .cpp

Для першої версії:

kernels/compute/stencil_fir.cpp

всередині:

SFPI callback
ternary/quaternary wrapper
tile helper
kernel_main
Варіант B: локальний helper header

Коли код розростеться:

kernels/compute/
├── stencil_fir.cpp
└── sfpu/
    ├── llk_math_eltwise_ternary_sfpu.h
    └── llk_math_eltwise_quaternary_sfpu.h

Але я б не клав це всередину tt-metal/.../hw/ckernels.
Краще тримати локально у своєму проєкті.

Практична порада по дизайну

Для твого випадку я б зробив не ternary, а одразу quaternary wrapper, бо FIR-стенсіл частіше виглядає так:

𝑦
=
𝑏
𝑎
𝑠
𝑒
+
𝑐
0
𝑥
0
+
𝑐
1
𝑥
1
+
𝑐
2
𝑥
2
y=base+c
0
	​

x
0
	​

+c
1
	​

x
1
	​

+c
2
	​

x
2
	​


Тобто тобі природно мати:

dst_index_base
dst_index_x0
dst_index_x1
dst_index_x2
dst_index_out

І всередині SFPI:

acc = base;
acc += c0 * x0;
acc += c1 * x1;
acc += c2 * x2;

Тоді API буде дуже чистий:

stencil_tile_quaternary<false>(base_idx, x0_idx, x1_idx, x2_idx, out_idx, c0, c1, c2);
Межа цього підходу

Навіть із красивим wrapper-ом ти все одно не уникнеш того, що:

copy_tile(...) повинен покласти inputs у DST slots
DST slots обмежені
при великому stencil radius доведеться робити кілька passes

Тобто wrapper прибирає хардкод індексів усередині арифметики, але не прибирає потребу керувати розкладкою DST.

Рекомендована мінімальна архітектура

Я б радив такий рівень абстракції:

SFPI callback
stencil_face_quaternary(dst_base, dst_x0, dst_x1, dst_x2, dst_out, c0, c1, c2)
LLK wrapper
_llk_math_eltwise_quaternary_sfpu_params_<false>(...)
tile helper
stencil_tile_quaternary<false>(...)
kernel_main
copy_tile(... -> dst 0..3)
stencil_tile_quaternary<false>(0,1,2,3,4,c0,c1,c2)
pack_tile(4,...)

Якщо хочеш, я можу одразу написати тобі готовий skeleton stencil_fir.cpp з:

локальним ternary або quaternary LLK wrapper-ом,
SFPI callback,
kernel_main() у стилі TT examples.
Pasted text.txt
Документ
Pasted text (2).txt
Документ
Thought for a few seconds

Так, і твої пасти це добре підтверджують: правильна структура власного LLK wrapper-а — це не “переписати весь LLK”, а зробити тонкий локальний шар поверх того самого патерну, який уже є для unary/binary. У TT коді llk_math_eltwise_binary_sfpu_binop(...) просто викликає _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(...) з face-функцією, індексами DST tiles і vector_mode . А сам params wrapper у цій абстракції відповідає за face iteration, SETRWC/DST addressing і start/done sync, тоді як SFPI face function робить лише SIMD-обчислення через dst_reg[] .

Яка має бути структура твого рішення
Рівень 1: SFPI face function

Тут тільки арифметика на одному face.

Вона має виглядати концептуально так:

template <typename... Args>
inline void calculate_my_stencil_face(
    uint32_t dst_index_in0,
    uint32_t dst_index_in1,
    uint32_t dst_index_in2,
    uint32_t dst_index_out,
    Args... args)
{
    constexpr uint32_t dst_tile_size_sfpi = 32;

    for (int i = 0; i < 8; ++i) {
        vFloat x0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        vFloat x1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        vFloat x2 = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi];

        vFloat y = ...;  // твоя формула
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = y;

        sfpi::dst_reg++;
    }
}

Чому саме так:

у прикладі calculate_sfpu_binary_mul(...) low-level SFPI face kernel працює саме через sfpi::dst_reg[dst_index * 32] і потім інкрементує sfpi::dst_reg++ усередині loop-а
у custom_sfpi_add теж пояснено, що один tile займає 32 vector slots, а один face — це 8 SIMD-ітерацій
Рівень 2: твій params wrapper

Оце місце, де ти хочеш уникнути хардкоду індексів.
Тоді wrapper приймає:

callback face_fn
dst_index_in0
dst_index_in1
dst_index_in2
dst_index_out
vector_mode
додаткові runtime args

і робить:

assert на допустимість DST tile indices
_start_
loop по faces
TTI_SETRWC(...)
_done_()

Саме це, за твоїм pasted summary, і є обов’язком _llk_math_eltwise_*_sfpu_params_: orchestration, а не математика .

Рівень 3: op-specific wrapper

Тонка функція рівня:

template <bool APPROXIMATE>
inline void llk_math_eltwise_ternary_sfpu_stencil(
    uint32_t dst_index0,
    uint32_t dst_index1,
    uint32_t dst_index2,
    uint32_t dst_out,
    int vector_mode = (int)VectorMode::RC)
{
    _llk_math_eltwise_ternary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_my_stencil_face<APPROXIMATE>,
        dst_index0,
        dst_index1,
        dst_index2,
        dst_out,
        vector_mode);
}

Це аналог того, як TT робить для binary max/min чи generic binary binop: вони не пишуть всю логіку заново в API function, а лише прокидають callback у _params_ wrapper .

Рівень 4: tile API у compute kernel

Щось на кшталт:

inline void my_stencil_tile(
    uint32_t idx0,
    uint32_t idx1,
    uint32_t idx2,
    uint32_t idx_out,
    float c0,
    float c1)
{
    MATH((_llk_math_eltwise_ternary_sfpu_params_<false>(
        calculate_my_stencil_face,
        idx0, idx1, idx2, idx_out,
        (int)VectorMode::RC,
        c0, c1)));
}

Тут MATH(...) важливий, бо виклик params wrapper-а має відбуватись лише на TRISC_MATH; це прямо зафіксовано і в прикладах custom_sfpi_add, custom_sfpi_smoothstep .

Як би я реально організував файли

Для твого проєкту я б робив так:

tt-wavelet/
└── kernels/
    └── compute/
        ├── stencil_fir.cpp
        └── sfpu/
            ├── ckernel_sfpu_stencil.h
            ├── llk_math_eltwise_ternary_sfpu_params.h
            └── llk_math_eltwise_ternary_sfpu_stencil.h
ckernel_sfpu_stencil.h

Тільки face-функція:

calculate_my_stencil_face(...)
llk_math_eltwise_ternary_sfpu_params.h

Тільки generic orchestration:

assert
start/done
face stepping
vector mode
llk_math_eltwise_ternary_sfpu_stencil.h

Оп-specific thin wrapper:

llk_math_eltwise_ternary_sfpu_stencil(...)
stencil_fir.cpp
kernel_main()
copy_tile(...)
pack_tile(...)
виклик my_stencil_tile(...)
Найважливіша відмінність: де НЕ треба дублювати код

Тобі не треба писати новий wrapper для кожної операції.
Тобі треба:

один generic ternary params wrapper
багато різних face functions
багато тонких op-specific wrappers

Тобто так:

generic ternary params wrapper
    ├── stencil op A
    ├── stencil op B
    ├── ternary blend
    └── fused affine ternary

Це така сама ідея, як у unary macro system: там теж один базовий _llk_math_eltwise_unary_sfpu_params_, а поверх нього багато різних ops і навіть макросів для різної кількості runtime параметрів .

Чи треба робити init wrapper

Так, якщо хочеш стилістично бути ближче до TT pattern, то добре мати ще init-функцію, бо unary і binary ops мають окремий *_init() шлях, який конфігурує address modifiers / MOP templates .

Тобто пара буде така:

template <bool APPROXIMATE>
inline void llk_math_eltwise_ternary_sfpu_stencil_init() {
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::...>();
}

Але тут є практична деталь: для custom op тобі треба вирішити, чи ти:

використовуєш існуючий init pattern,
чи твоя SFPI face function не потребує нічого особливого, окрім стандартної конфігурації.

Для першого working prototype можна почати без складного custom init, а вже потім, якщо операція вимагає специфічних MOP/address mods, доробити.

Який шаблон я б радив для тебе

Не fully-generic N-ary, а конкретно:

binary
ternary
quaternary

Бо це ще читається, дебажиться і не перетворюється на цирк із масивами індексів у device-side коді.

Для FIR/stencil найкорисніший варіант

Одразу quaternary:

base
x0
x1
x2
out

бо формула типу

𝑦
=
𝑏
𝑎
𝑠
𝑒
+
𝑐
0
𝑥
0
+
𝑐
1
𝑥
1
+
𝑐
2
𝑥
2
y=base+c
0
	​

x
0
	​

+c
1
	​

x
1
	​

+c
2
	​

x
2
	​


натурально лягає саме на 4 input tiles + 1 output.

Підсумок

Так виглядає правильна структура, якщо ти хочеш не хардкодити dst layout у SFPI:

SFPI face function приймає dst_index_*
generic params wrapper управляє faces і sync
op-specific wrapper лише прокидає callback
kernel_main() сам вирішує, у які DST slots покласти tiles через copy_tile(...)

І це повністю узгоджується з тією ж архітектурою, яку ти зібрав у pasted notes: _llk_math_eltwise_*_sfpu_params_ — це orchestration layer, а ckernel_sfpu_* — actual compute layer .

Наступним кроком я б уже писав тобі готовий ternary або quaternary skeleton під tt-wavelet, а не загальну теорію.

stencil\_fir.cpp

tt-wavelet/
└── kernels/
    └── compute/
        ├── stencil_fir.cpp
        └── sfpu/
            ├── ckernel_sfpu_stencil.h
            ├── llk_math_eltwise_ternary_sfpu_params.h
            └── llk_math_eltwise_ternary_sfpu_stencil.h
що буде містити кожен файл

Так. Розкладу по файлах і по ролях, щоб було ясно, що куди класти.

Тут головна ідея така:

stencil_fir.cpp — це сам compute kernel entrypoint
ckernel_sfpu_stencil.h — це чиста SFPI-арифметика
llk_math_eltwise_ternary_sfpu_params.h — це generic LLK orchestration wrapper
llk_math_eltwise_ternary_sfpu_stencil.h — це операційний thin wrapper саме для твоєї stencil-операції
1. stencil_fir.cpp

Це головний файл compute kernel-а.

Саме тут буде kernel_main(), і саме цей файл ти передаєш у CreateKernel(..., ".../stencil_fir.cpp", ...).

Що він містить
A. include-и
стандартні TT compute include-и
твій ckernel_sfpu_stencil.h
твій llk_math_eltwise_ternary_sfpu_stencil.h
B. init compute path

Якщо треба:

binary_op_init_common(...)
або інший init для SFPU / tile regs / dst
C. kernel_main()

Оце найголовніше. Тут буде:

cb_wait_front(...)
чекати, поки в circular buffer-ах з’являться input tiles
tile_regs_acquire()
захопити DST регістри
copy_tile(...)
покласти кілька input tiles у DST slots

викликати твою tile-level операцію
щось типу:

stencil_tile_ternary<false>(0, 1, 2, 3, c0, c1);
cb_reserve_back(...)
pack_tile(...)
cb_push_back(...)
упакувати output tile у вихідний CB
tile_regs_release()
cb_pop_front(...)
прибрати використані input tiles з CB
Як це виглядає логічно
void kernel_main() {
    // 1. дістати runtime args / coeffs
    // 2. чекати inputs
    // 3. copy_tile в dst slots
    // 4. виклик stencil op
    // 5. pack result
}
Що тут НЕ треба класти

У stencil_fir.cpp не варто класти:

великий generic LLK wrapper
складну SFPI face-арифметику
reusable helper-и, які краще винести

Тобто цей файл має бути оркестратором одного конкретного kernel launch path.

2. sfpu/ckernel_sfpu_stencil.h

Це файл з реальною арифметикою на SFPU.

Тут живе функція рівня:

template <typename... Args>
inline void calculate_stencil_face(
    uint32_t dst_index_in0,
    uint32_t dst_index_in1,
    uint32_t dst_index_in2,
    uint32_t dst_index_out,
    Args... args)
Що саме вона робить

Для одного face:

читає vFloat із dst_reg
множить / додає / акумулює
записує результат назад у dst_reg

Наприклад:

vFloat x0 = sfpi::dst_reg[dst_index_in0 * 32];
vFloat x1 = sfpi::dst_reg[dst_index_in1 * 32];
vFloat x2 = sfpi::dst_reg[dst_index_in2 * 32];
vFloat y = x0 + c0 * x1 + c1 * x2;
sfpi::dst_reg[dst_index_out * 32] = y;
sfpi::dst_reg++;

і так 8 разів для одного face.

Що тут має бути
#ifdef TRISC_MATH
namespace типу ckernel::sfpu
одна або кілька face-функцій

Наприклад:

calculate_stencil_face_ternary
calculate_stencil_face_quaternary
можливо окремі fused варіанти
Що тут НЕ має бути
cb_wait_front
copy_tile
pack_tile
kernel_main
логіка проходу по всіх faces
TTI_SETRWC(...) orchestration

Бо це вже не арифметика, а wrapper-рівень.

3. sfpu/llk_math_eltwise_ternary_sfpu_params.h

Це generic LLK wrapper, який керує запуском твоєї SFPI face-функції.

Це найнижчий з твоїх wrapper-рівнів.

Що він містить

Функцію типу:

template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_ternary_sfpu_params_(
    Callable&& sfpu_func,
    uint32_t dst_index_in0,
    uint32_t dst_index_in1,
    uint32_t dst_index_in2,
    uint32_t dst_index_out,
    int vector_mode,
    Args&&... args)
Що вона робить
1. Перевірки
LLK_ASSERT(...)
чи не виходять dst_index_* за межі DST
2. Start
_llk_math_eltwise_ternary_sfpu_start_<DST_SYNC_MODE>(...)
3. Loop по faces

Залежно від:

VectorMode::R
VectorMode::C
VectorMode::RC

викликає передану face-функцію потрібну кількість разів

4. Перемикання face cursor
TTI_SETRWC(...)
5. Done
_llk_math_eltwise_ternary_sfpu_done_()
Найважливіше

Цей файл не знає, що саме означає твоя операція.
Для нього callback — просто callable.

Тобто цей wrapper можна буде перевикористати і для:

stencil sum
fused affine ternary
ternary blend
іншої 3-input SFPU операції
Що тут НЕ має бути
конкретна формула stencil
kernel_main
copy_tile
знання про base, x[k-1], x[k-2]
4. sfpu/llk_math_eltwise_ternary_sfpu_stencil.h

Це тонкий op-specific wrapper.

Він уже каже:
“візьми саме calculate_stencil_face(...) і запусти її через generic ternary params wrapper”.

Що він містить
A. include-и
ckernel_sfpu_stencil.h
llk_math_eltwise_ternary_sfpu_params.h
B. одну або кілька зручних функцій

Наприклад:

template <bool APPROXIMATE>
inline void llk_math_eltwise_ternary_sfpu_stencil(
    uint32_t dst_index_in0,
    uint32_t dst_index_in1,
    uint32_t dst_index_in2,
    uint32_t dst_index_out,
    int vector_mode,
    float c0,
    float c1)
{
    _llk_math_eltwise_ternary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_stencil_face,
        dst_index_in0,
        dst_index_in1,
        dst_index_in2,
        dst_index_out,
        vector_mode,
        c0,
        c1);
}
Чому він потрібен

Щоб у stencil_fir.cpp не писати кожного разу довжелезний generic виклик руками.

Тобто замість:

MATH((_llk_math_eltwise_ternary_sfpu_params_<false>(
    ckernel::sfpu::calculate_stencil_face,
    0, 1, 2, 3,
    static_cast<int>(VectorMode::RC),
    c0, c1)));

ти пишеш просто:

MATH((llk_math_eltwise_ternary_sfpu_stencil<false>(
    0, 1, 2, 3,
    static_cast<int>(VectorMode::RC),
    c0, c1)));
Що тут НЕ має бути
kernel_main
generic face-loop code
copy_tile
низькорівневий TTI_SETRWC(...)
Як вони взаємодіють разом
Потік виклику такий
stencil_fir.cpp

каже:

у DST slot 0,1,2 уже лежать input tiles, порахуй output у slot 3

↓

llk_math_eltwise_ternary_sfpu_stencil.h

каже:

для цієї stencil-операції виклич generic ternary wrapper з calculate_stencil_face

↓

llk_math_eltwise_ternary_sfpu_params.h

каже:

запущу callback по всіх потрібних faces і зроблю весь LLK control flow

↓

ckernel_sfpu_stencil.h

каже:

на одному face порахую арифметику через dst_reg

У дуже короткому вигляді
stencil_fir.cpp

compute kernel orchestration

kernel_main
cb_wait_front
copy_tile
invoke op
pack_tile
ckernel_sfpu_stencil.h

арифметика

читає/пише dst_reg
рахує формулу для одного face
llk_math_eltwise_ternary_sfpu_params.h

generic LLK orchestration

asserts
start/done
face loop
TTI_SETRWC
llk_math_eltwise_ternary_sfpu_stencil.h

зручний wrapper для твоєї конкретної операції

зв’язує generic wrapper з твоєю SFPI face function
Я б ще трохи підправив структуру

Для назви stencil_fir.cpp я б зробив так:

kernels/compute/
├── stencil_fir.cpp
└── sfpu/
    ├── ckernel_sfpu_stencil.h
    ├── llk_math_eltwise_ternary_sfpu_params.h
    └── llk_math_eltwise_stencil.h

Бо файл llk_math_eltwise_ternary_sfpu_stencil.h трохи задовгий і занадто прив’язаний до arity.
Якщо потім підеш у quaternary, назва стане не дуже зручною.

Краще:

llk_math_eltwise_stencil.h
всередині вже можна мати:
ternary version
quaternary version
Практична порада для твого кейсу

Оскільки в тебе FIR/stencil, я б реально розглянув не ternary, а одразу quaternary layout:

in0 = base
in1 = x0
in2 = x1
in3 = x2
out

тоді файли будуть ті самі за роллю, просто:

ckernel_sfpu_stencil.h матиме quaternary face function
llk_math_eltwise_*_params.h стане quaternary
llk_math_eltwise_stencil.h ховатиме це від stencil_fir.cpp

Бо FIR майже завжди хоче або base + sum(...), або accumulation.

Якщо хочеш, я наступним повідомленням можу вже розписати точний skeleton вмісту stencil_fir.cpp по секціях: include-и, init, runtime args, cb_wait_front, copy_tile, виклик op, pack_tile, cleanup.

Pasted text.txt
Документ
окей тоді якщо я хочу щоб ця опа була не тільки під нашу задачу а була як загальний stencil_mac (multiply acuu) то треба писати llk

Thought for a few seconds

Так — якщо ти хочеш, щоб це була не разова опа під wavelet/FIR, а більш загальна reusable op типу stencil_mac, тоді вже є сенс писати власний LLK-level wrapper.

Але тут важливо розрізняти який саме LLK.

Що значить “писати llk” у твоєму випадку

Це не означає:

лізти в tt-metal/.../hw/ckernels/... і міняти внутрішні файли Tenstorrent
переписувати існуючий unary/binary framework

Це означає:

у своєму проєкті зробити новий reusable wrapper-рівень, аналогічний тому, як TT робить:
llk_math_eltwise_unary_sfpu_typecast(...)
llk_math_eltwise_binary_sfpu_left_shift(...)
llk_math_eltwise_unary_sfpu_reduce(...)

У всіх цих прикладах pattern однаковий: op-specific wrapper лише підбирає callback і compile-time параметри, а далі викликає generic _llk_math_eltwise_*_sfpu_params_<...>(...)

Коли відповідь “так, треба”
Якщо ти хочеш API рівня:
stencil_mac<APPROXIMATE, RADIUS, ...>(
    dst_base,
    dst_x0,
    dst_x1,
    dst_x2,
    dst_out,
    vector_mode,
    c0, c1, c2);

або навіть:

stencil_mac<APPROXIMATE, 3>(indices..., coeffs...)

Тоді так — краще зробити власний LLK wrapper, бо:

не буде хардкоду DST layout у compute kernel
з’явиться нормальний op-level API
можна буде перевикористовувати op в різних kernel-ах
можна винести compile-time параметри, як у typecast/shift/reduce wrappers
Коли відповідь “ще не обов’язково”

Якщо ти просто хочеш один working prototype для конкретного FIR/stencil у одному compute kernel — тоді LLK wrapper необов’язковий.
Можна прожити з локальною SFPI face function + thin helper.

Для reusable stencil_mac яка правильна архітектура

Я б робив так:

1. SFPI compute primitive

Файл на кшталт:

ckernel_sfpu_stencil.h

Там:

calculate_stencil_mac_face<...>(...)

Це чиста арифметика.

2. Generic params wrapper

Наприклад:

llk_math_eltwise_stencil_sfpu_params.h

Там:

start/done
face loop
VectorMode
TTI_SETRWC
asserts на dst_index

Це аналог _llk_math_eltwise_unary_sfpu_params_ / _binary_....

3. Op-level wrapper

Наприклад:

llk_math_eltwise_stencil_mac.h

Там уже API типу:

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_stencil_mac(
    uint32_t dst_base,
    uint32_t dst_x0,
    uint32_t dst_x1,
    uint32_t dst_x2,
    uint32_t dst_out,
    int vector_mode,
    float c0,
    float c1,
    float c2);

Саме цей рівень уже і є твоєю загальною операцією.

Чому цього не достатньо зробити просто як “SFPI helper”

Бо тоді у тебе не буде справжньої op abstraction.

Ти знову залишишся на рівні:

вручну copy в DST
вручну передавати індекси
вручну розуміти, який slot що означає

А з reusable LLK wrapper-ом у тебе з’являється семантичний рівень:

це не “щось там із 4 dst slots”
це саме stencil MAC op
Який дизайн кращий: ternary чи щось більш загальне

Для stencil_mac я б не називав його ternary.

Бо “ternary” описує лише кількість входів, а не сенс операції.

Краще мислити так:

Варіант A: stencil_mac_3tap
llk_math_eltwise_stencil_mac_3tap(...)

Для:

𝑦
=
𝑏
𝑎
𝑠
𝑒
+
𝑐
0
𝑥
0
+
𝑐
1
𝑥
1
+
𝑐
2
𝑥
2
y=base+c
0
	​

x
0
	​

+c
1
	​

x
1
	​

+c
2
	​

x
2
	​

Варіант B: stencil_mac_4tap, 5tap, ...

Якщо хочеш кілька явних arity-specific реалізацій.

Варіант C: family of wrappers
llk_math_eltwise_stencil_mac_3tap
llk_math_eltwise_stencil_mac_5tap
llk_math_eltwise_stencil_mac_7tap

Це, на мою думку, краще ніж один псевдо-generic N-ary wrapper, бо device-side код стає складним дуже швидко.

Що можна параметризувати, якщо робити це “по-LLK-стильному”

Твоє pasted summary дуже влучне: у LLK wrapper-ів параметри діляться на compile-time і runtime

Compile-time

Сюди варто винести:

APPROXIMATE
ITERATIONS = 8
можливо DATA_FORMAT
можливо ACCUMULATE_IN_FP32
можливо NUM_TAPS

Наприклад:

template <
    bool APPROXIMATE,
    uint32_t NUM_TAPS,
    uint32_t ITERATIONS = 8>
inline void llk_math_eltwise_stencil_mac(...)
Runtime

Сюди:

c0, c1, c2, ...
vector_mode
можливо scale
можливо bias

Це дуже схоже на unary reduce/typecast/shift wrappers, де частина параметрів compile-time, а частина runtime

Але є важливе practical обмеження
stencil_mac — це не просто “ще одна унарна активація”

Бо тобі треба кілька input tiles одночасно.

Тому твій wrapper повинен явно моделювати:

кілька input DST indices
один output DST index

Тобто API вже буде ближчий до:

binary
ternary
quaternary

а не unary.

Саме тому я б для старту робив arity-specific reusable ops, а не “супер generic stencil for any N”.

Моє практичне рішення для тебе
Так, якщо хочеш загальну reusable операцію — пиши LLK wrapper.

Але роби локальний LLK-style layer у своєму проєкті, а не патчі в tt-metal.

Як би я це назвав і розклав
tt-wavelet/
└── kernels/
    └── compute/
        ├── stencil_fir.cpp
        └── sfpu/
            ├── ckernel_sfpu_stencil_mac.h
            ├── llk_math_eltwise_stencil_mac_params.h
            ├── llk_math_eltwise_stencil_mac_3tap.h
            └── llk_math_eltwise_stencil_mac_5tap.h
ckernel_sfpu_stencil_mac.h
face-level arithmetic
calculate_stencil_mac_3tap_face
calculate_stencil_mac_5tap_face
llk_math_eltwise_stencil_mac_params.h
generic orchestration for “multi-input stencil MAC”
start/done
vector_mode
face stepping
dst asserts
llk_math_eltwise_stencil_mac_3tap.h
op-level wrapper for 3 taps
llk_math_eltwise_stencil_mac_5tap.h
op-level wrapper for 5 taps
Чому не один повністю generic stencil_mac<N>

Можна, але я б не радив як першу версію, бо:

доведеться тягнути масиви індексів
доведеться узгоджувати runtime coeff arrays
складніше дебажити DST layout
складніше писати читабельний SFPI callback

Для TT compute kernel кращий баланс — це кілька чітких arity-specific ops.

Висновок

Так, для загальної reusable операції stencil_mac уже є сенс писати власний LLK-style wrapper.
Але правильно це робити так:

SFPI face function окремо
generic params orchestration окремо
op-specific reusable wrapper окремо
усе локально в tt-wavelet, без змін у внутрішніх заголовках TT

І тоді в compute kernel ти матимеш уже нормальний API рівня:

llk_math_eltwise_stencil_mac_3tap<false>(
    dst_base, dst_x0, dst_x1, dst_x2, dst_out,
    (int)VectorMode::RC,
    c0, c1, c2);

а не просто “ручну арифметику на якихось слотах”.

Наступним кроком я б уже спроєктував для тебе конкретно stencil_mac_3tap vs stencil_mac_5tap, щоб вибрати правильний рівень узагальнення.

дивись але це ж можна generalize у мене в вейвлетах бувають фільтри різної довжини
коли я роблю predict
d[n] = od[n] + sum^{l-1}_j=0 coeffs_j *even[k-j]

як це не розбивати за довжинами фільтрів

Так, узагальнити можна, і для вейвлетів це якраз правильний напрямок, бо довжина predict/update фільтрів справді змінюється.

Але тут є важлива межа:

Математично — так, це один generic stencil MAC

Твоя формула:

𝑑
[
𝑛
]
=
𝑜
𝑑
[
𝑛
]
+
∑
𝑗
=
0
𝐿
−
1
𝑐
𝑗
⋅
𝑒
𝑣
𝑒
𝑛
[
𝑘
−
𝑗
]
d[n]=od[n]+
j=0
∑
L−1
	​

c
j
	​

⋅even[k−j]

це просто:

base = od[n]
inputs[j] = even[k-j]
coeffs[j] = c_j
output = base + dot(coeffs, inputs)

Тобто на рівні абстракції це справді один generic op:

stencil_mac(base, inputs[0..L-1], coeffs[0..L-1], out)
Але апаратно є проблема

На SFPU/LLK ти не працюєш з “математичним масивом input vectors” напряму.
Ти працюєш з:

DST tile slots
dst_reg
обмеженим числом tiles, які можна одночасно тримати
фіксованим device-side control flow

І через це повністю generic L зробити можна, але не в стилі “одна магічна функція для будь-якого L без наслідків”.

Правильна генералізація

Не треба розбивати по конкретних фільтрах типу db2, cdf97, bior4.4.

Треба розбивати по способу виконання.

Ось правильна модель.

Рівень 1: один загальний API

Ти мислиш операцію так:

predict_stencil_mac(
    dst_base,
    dst_inputs_begin,
    dst_out,
    num_taps,
    coeffs...)

або концептуально:

stencil_mac(base, x[0:L], coeffs[0:L], out)

Це правильна семантична абстракція.

Рівень 2: всередині — chunked execution

Бо L може бути:

2
4
6
8
10
14
...

А DST не безмежний.

Тому generic реалізація має бути не “окремий kernel для кожного L”, а:

варіант A: один generic loop по taps

Якщо твої coeffs і input indices доступні як параметри.

варіант B: chunked MAC

Якщо L завеликий, ти рахуєш суму частинами:

𝑎
𝑐
𝑐
1
=
∑
𝑗
=
0
𝑝
−
1
𝑐
𝑗
𝑥
𝑗
acc
1
	​

=
j=0
∑
p−1
	​

c
j
	​

x
j
	​

𝑎
𝑐
𝑐
2
=
𝑎
𝑐
𝑐
1
+
∑
𝑗
=
𝑝
𝑞
−
1
𝑐
𝑗
𝑥
𝑗
acc
2
	​

=acc
1
	​

+
j=p
∑
q−1
	​

c
j
	​

x
j
	​


і так далі.

Тобто:

𝑑
[
𝑛
]
=
𝑜
𝑑
[
𝑛
]
+
chunk
0
+
chunk
1
+
⋯
d[n]=od[n]+chunk
0
	​

+chunk
1
	​

+⋯
Найкраща абстракція для тебе
Не stencil_mac_3tap, stencil_mac_5tap, ...

а:

stencil_mac_fixed_max_taps<MAX_TAPS>

і runtime parameter:

num_taps

Наприклад:

template <bool APPROXIMATE, uint32_t MAX_TAPS>
inline void llk_math_eltwise_stencil_mac(
    uint32_t dst_base,
    const uint32_t* dst_input_indices,
    uint32_t dst_out,
    uint32_t num_taps,
    int vector_mode,
    const float* coeffs)

Ідея така:

wrapper підтримує до MAX_TAPS
реально використовується лише num_taps
цикл усередині SFPI йде по j = 0 .. num_taps-1
Але тут є нюанс device-side C++

На TT kernel side не все так комфортно, як на host:

масиви параметрів
pointer arithmetic
runtime-sized loops
передача багатьох float coefficients

це треба робити обережно.

Тому practically я б радив такий компроміс.

Практично правильний дизайн
1. Один загальний op: stencil_mac

Семантично — один.

2. Два режими виконання
Режим small filter

Для L <= MAX_IN_DST
усі input tiles одночасно лежать у DST.

Тоді:

𝑜
𝑢
𝑡
=
𝑏
𝑎
𝑠
𝑒
+
∑
𝑗
=
0
𝐿
−
1
𝑐
𝑗
𝑥
𝑗
out=base+
j=0
∑
L−1
	​

c
j
	​

x
j
	​


рахується за один прохід.

Режим chunked / accumulated

Для довших фільтрів:

перший chunk рахує:

𝑎
𝑐
𝑐
=
𝑏
𝑎
𝑠
𝑒
+
∑
𝑗
=
0
𝑝
−
1
𝑐
𝑗
𝑥
𝑗
acc=base+
j=0
∑
p−1
	​

c
j
	​

x
j
	​


наступний chunk:

𝑎
𝑐
𝑐
=
𝑎
𝑐
𝑐
+
∑
𝑗
=
𝑝
𝑞
−
1
𝑐
𝑗
𝑥
𝑗
acc=acc+
j=p
∑
q−1
	​

c
j
	​

x
j
	​


Тобто out стає новим base для наступного проходу.

Чому це краще за “розбивати по довжинах”

Бо ти не хочеш робити:

stencil_mac_2tap
stencil_mac_4tap
stencil_mac_6tap
stencil_mac_8tap
...

Це погано масштабується.

Натомість ти робиш:

один generic kernel contract
base + Σ coeff[j] * x[j]

і окремо:

execution policy
скільки taps береться за один pass
Як це виглядає для predict

Твоя predict формула:

𝑑
[
𝑛
]
=
𝑜
𝑑
[
𝑛
]
+
∑
𝑗
=
0
𝐿
−
1
𝑐
𝑗
⋅
𝑒
𝑣
𝑒
𝑛
[
𝑘
−
𝑗
]
d[n]=od[n]+
j=0
∑
L−1
	​

c
j
	​

⋅even[k−j]

в kernel design виглядає так:

Reader
читає od[n]
читає even[k-j] для j=0..L-1
або всі одразу, або chunk-ами
Compute
бере base = od[n] або попередній accumulator
виконує generic stencil MAC
Writer
пише d[n]
Найкращий рівень генералізації

Я б радив таку ієрархію.

Семантичний рівень
predict_stencil_mac(...)
update_stencil_mac(...)
Низький reusable рівень
llk_math_eltwise_stencil_mac(...)
Усередині
num_taps
coeffs
input indices
chunking

Тобто predict/update — це не різні арифметичні примітиви.
Це той самий stencil_mac, просто з різними:

coeffs
source stream (even або odd)
boundary handling
Що саме тоді писати
Не family by filter length

Погана ідея:

stencil_mac_2tap
stencil_mac_4tap
stencil_mac_6tap
А generic op + max capacity

Краща ідея:

template <bool APPROXIMATE, uint32_t MAX_TAPS_PER_PASS>
llk_math_eltwise_stencil_mac(...)

де:

num_taps — runtime
MAX_TAPS_PER_PASS — compile-time
якщо L > MAX_TAPS_PER_PASS, робиш кілька passes
Як мислити DST slots

Припустимо ти можеш безпечно тримати:

1 tile для base
6 tiles для inputs
1 tile для output

Тоді MAX_TAPS_PER_PASS = 6.

Якщо фільтр довжини 4

один pass.

Якщо фільтр довжини 10

два passes:

pass 1: taps 0..5
pass 2: taps 6..9, а base = partial_sum

Оце і є правильна генералізація.

Як може виглядати API

Щось такого стилю:

template <bool APPROXIMATE, uint32_t MAX_TAPS_PER_PASS>
inline void llk_math_eltwise_stencil_mac(
    uint32_t dst_base,
    const uint32_t* dst_inputs,
    uint32_t num_taps,
    uint32_t dst_out,
    int vector_mode,
    const float* coeffs)

Але якщо pointer-based API для kernel side незручний, тоді можна зробити простіше:

template <bool APPROXIMATE, uint32_t MAX_TAPS_PER_PASS>
inline void llk_math_eltwise_stencil_mac(
    uint32_t dst_base,
    uint32_t dst_input0,
    uint32_t dst_input1,
    uint32_t dst_input2,
    uint32_t dst_input3,
    uint32_t dst_input4,
    uint32_t dst_input5,
    uint32_t num_taps,
    uint32_t dst_out,
    int vector_mode,
    float c0, float c1, float c2, float c3, float c4, float c5)

Тобто compile-time верхня межа фіксована, а реально активних taps менше.

Це менш красиво, зате набагато реалістичніше для device code.

Висновок

Так, це треба узагальнювати не по довжинах фільтрів, а як один generic stencil_mac:

𝑜
𝑢
𝑡
=
𝑏
𝑎
𝑠
𝑒
+
∑
𝑗
=
0
𝐿
−
1
𝑐
𝑗
𝑥
𝑗
out=base+
j=0
∑
L−1
	​

c
j
	​

x
j
	​


Але реалізаційно правильно робити так:

один reusable stencil_mac
runtime num_taps
compile-time MAX_TAPS_PER_PASS
довгі фільтри обробляти chunk-ами, а не окремими kernel-ами під кожну довжину

Тобто не 3tap/5tap/7tap як основна архітектура, а:

generic op + execution chunks

Це саме той дизайн, який тобі потрібен для wavelet predict/update.

Можу далі накидати конкретний дизайн API для stencil_mac під TT kernel side, щоб він був реалістичний з урахуванням DST slots, coeffs і multiple passes.