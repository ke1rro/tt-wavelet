# Tracy Profiling Guide для tt-wavelet

## Що таке Tracy?

**Tracy Profiler** — це потужний інструмент профілювання з відкритим кодом, адаптований Tenstorrent для роботи з їхнім hardware (N300/Wormhole, Blackhole). 

### Ключові можливості:
- ⏱️ **Наносекундна точність** real-time профілювання
- 🔍 **Hybrid профайлер**: інструментація коду + семплінг  
- 🌐 **Віддалений телеметричний доступ**: підключення до програм на N300
- 📊 Профілювання: CPU, memory allocations, locks, context switches
- 🎯 Візуалізація виконання на timeline з деталями по кожній зоні

### Як це працює в екосистемі Tenstorrent?

```
┌─────────────────┐
│   Ваш код C++   │  ← Додаємо макроси ZoneScoped
│   tt-wavelet    │
└────────┬────────┘
         │ збирається з
         ↓
┌─────────────────┐
│  TracyClient    │  ← Бібліотека з tt-metal
│   (library)     │
└────────┬────────┘
         │ передає дані через TCP
         ↓
┌─────────────────┐       ┌──────────────┐
│ capture-release │  або  │  Tracy GUI   │
│   (CLI tool)    │       │ (візуалізація)│
└─────────────────┘       └──────────────┘
         │                        │
         ↓                        ↓
    profile.tracy   →   відкрити у GUI
```

---

## Встановлення і налаштування

### 1. Збірка tt-metal з Tracy

Tracy **увімкнений за замовчуванням** у tt-metal, але наша збірка його вимкнула. Потрібно пересібрати:

```bash
cd /home/ostap/tt-wavelet

# Очистити стару збірку
rm -rf build/

# Зібрати з Tracy
./build.sh Release

# Перевірити, що Tracy увімкнено
grep "ENABLE_TRACY" build/CMakeCache.txt
# Повинно бути: ENABLE_TRACY:BOOL=ON
```

### 2. Перевірка Tracy tools

Після збірки перевірте наявність CLI інструментів:

```bash
# Шукаємо Tracy tools
find build -type f -name "capture-release" -o -name "csvexport-release"

# Маси бути:
# build/third-party/tt-metal/tools/profiler/bin/capture-release
# build/third-party/tt-metal/tools/profiler/bin/csvexport-release
```

### 3. Tracy GUI (опціонально)

**Проблема:** Tracy форк від Tenstorrent має специфічні залежності (enchantum SDK), тому збирати GUI складно.

**Рішення:**
- **Варіант А:** Використовувати CLI tools для захоплення + відкривати `.tracy` файли на іншій машині (Windows/Mac з встановленим GUI)
- **Варіант Б:** Використовувати тільки CSV експорт для аналізу в Python/Jupyter
- **Варіант В:** Real-time підключення з іншої машини (потрібен SSH port forwarding)

---

## Інструментація коду

### Базовий приклад

Створіть header file `tt-wavelet/tracy_profiler.hpp`:

```cpp
#pragma once

#if defined(TRACY_ENABLE)
    #include <tracy/Tracy.hpp>
    #define FWT_ZONE_SCOPED         ZoneScoped
    #define FWT_ZONE_NAMED(name)    ZoneScopedN(name)
    #define FWT_ZONE_COLOR(color)   ZoneScopedC(color)
    #define FWT_FRAME_MARK          FrameMark
    #define FWT_MESSAGE(msg)        TracyMessage(msg, sizeof(msg))
#else
    #define FWT_ZONE_SCOPED
    #define FWT_ZONE_NAMED(name)
    #define FWT_ZONE_COLOR(color)
    #define FWT_FRAME_MARK
    #define FWT_MESSAGE(msg)
#endif
```

### Використання у коді

```cpp
#include "tracy_profiler.hpp"
#include "tt-metalium/host_api.hpp"

void fast_wavelet_transform(const Tensor& input) {
    FWT_ZONE_SCOPED;  // Профілювати всю функцію
    
    // Initialization
    {
        FWT_ZONE_NAMED("FWT_Initialize");
        // ... ініціалізація ...
    }
    
    // Host->Device transfer
    {
        FWT_ZONE_COLOR(0xFF0000);  // Червоний колір
        auto device = tt::tt_metal::GetDefaultDevice();
        // ... передача даних ...
    }
    
    // Kernel execution
    {
        FWT_ZONE_NAMED("FWT_Kernel_Execute");
        FWT_MESSAGE("Starting wavelet decomposition");
        // ... виконання кернелів ...
    }
    
    FWT_FRAME_MARK;  // Позначити кінець фрейму
}
```

### Оновлення CMakeLists.txt

Додайте Tracy до вашого таргету:

```cmake
if (BUILD_TT_WAVELET)
    add_executable(tt_wavelet_test main.cpp)
    
    target_link_libraries(tt_wavelet_test PRIVATE 
        Metalium::Metal
    )
    
    # Додати Tracy, якщо увімкнено
    if(TARGET TracyClient)
        target_link_libraries(tt_wavelet_test PRIVATE TracyClient)
        message(STATUS "Tracy profiling enabled for tt_wavelet_test")
    endif()
endif()
```

---

## Захоплення профілів

### Метод 1: CLI Capture (рекомендований для N300)

```bash
# Запустити capture в фоновому режимі
./build/third-party/tt-metal/tools/profiler/bin/capture-release -o fwt_profile.tracy &
CAPTURE_PID=$!

# Запустити вашу програму
./build/tt-wavelet/tt_wavelet_test

# Зупинити capture
kill $CAPTURE_PID

# Файл fwt_profile.tracy тепер містить профіль
ls -lh fwt_profile.tracy
```

### Метод 2: Python модуль (для Python коду)

```bash
# Профілювати Python скрипт
python -m tracy tests/test_fwt.py

# З pytest
python -m tracy -m pytest tests/test_wavelet_kernels.py
```

### Метод 3: Real-time з GUI (через SSH)

На локальній машині:
```bash
# Forwarding порту 8086
ssh -NL 8086:127.0.0.1:8086 root@<N300_IP>
```

На N300:
```bash
# Просто запустити програму, GUI автоматично підключиться
./build/tt-wavelet/tt_wavelet_test
```

---

## Аналіз результатів

### Опція 1: Експорт у CSV

```bash
# Конвертувати .tracy файл у CSV
./build/third-party/tt-metal/tools/profiler/bin/csvexport-release \
    -u fwt_profile.tracy > fwt_data.csv

# Аналіз у Python/Pandas
python3 << 'EOF'
import pandas as pd

df = pd.read_csv('fwt_data.csv')
print(df.head())
print("\nТоп-10 найповільніших зон:")
print(df.nlargest(10, 'time'))
EOF
```

### Опція 2: Tracy GUI

Якщо у вас є доступ до Tracy GUI на іншій машині:

1. Скопіювати `.tracy` файл:
   ```bash
   scp root@<N300_IP>:~/tt-wavelet/fwt_profile.tracy ./
   ```

2. Відкрити у GUI:
   ```bash
   # На Windows/Mac/Linux з встановленим Tracy
   Tracy-release fwt_profile.tracy
   ```

---

## Best Practices

### 1. Стратегічна інструментація

**НЕ РОБІТЬ:**
```cpp
for (int i = 0; i < 1000000; i++) {
    ZoneScoped;  // ❌ Overhead буде величезний!
    // ...
}
```

**РОБІТЬ:**
```cpp
{
    ZoneScopedN("HotLoop");  // ✅ Одна зона для всього циклу
    for (int i = 0; i < 1000000; i++) {
        // ...
    }
}
```

### 2. Пріоритетні місця для інструментації

- ✅ Host-Device комунікація
- ✅ Kernel launches
- ✅ Memory allocations/transfers (DRAM ↔ L1)
- ✅ Критичні обчислення (FWT decomposition, reconstruction)
- ✅ Синхронізації та блокування

### 3. Використання кольорів

```cpp
#define COLOR_HOST_DEVICE   0xFF0000  // Червоний
#define COLOR_KERNEL        0x00FF00  // Зелений
#define COLOR_MEMORY        0x0000FF  // Синій
#define COLOR_SYNC          0xFFFF00  // Жовтий

// У коді:
ZoneScopedC(COLOR_HOST_DEVICE);
```

### 4. Signposts для маркування подій

```cpp
#include "tracy/Tracy.hpp"

TracyMessage("Starting batch processing", 25);
// ... код ...
TracyMessage("Batch complete", 14);
```

---

## Troubleshooting

### Tracy tools не знайдено після збірки

```bash
# Перевірити статус Tracy у CMake
grep ENABLE_TRACY build/CMakeCache.txt

# Якщо OFF, пересібрати:
rm -rf build/
./build.sh Release
```

### Програма не з'єднується з Tracy GUI

```bash
# Перевірити, чи відкритий порт 8086
ss -tulpn | grep 8086

# Перевірити firewall
sudo ufw allow 8086/tcp

# Або використовувати SSH forwarding
ssh -NL 8086:127.0.0.1:8086 root@<N300_IP>
```

### Overhead від профілювання

```bash
# Збірка без Tracy для production
cmake -DENABLE_TRACY=OFF ../
make -j$(nproc)
```

---

## Корисні посилання

- [Tracy офіційна документація](https://github.com/wolfpld/tracy/releases/latest/download/tracy.pdf) — розділи 1-3
- [tt-metal Tracy docs](../third-party/tt-metal/docs/source/tt-metalium/tools/tracy_profiler.rst)
- [Tracy форк Tenstorrent](https://github.com/tenstorrent-metal/tracy)
- [Приклади інструментації](../third-party/tt-metal/tests/ttnn/profiling/)

---

## Швидкий старт checklist

- [ ] Зібрати tt-metal з `ENABLE_TRACY=ON`
- [ ] Перевірити наявність `capture-release` у `build/tools/profiler/bin/`
- [ ] Додати `tracy_profiler.hpp` до проекту
- [ ] Інструментувати 2-3 ключові функції FWT
- [ ] Оновити `CMakeLists.txt` для лінку з `TracyClient`
- [ ] Захопити тестовий профіль: `capture-release -o test.tracy`
- [ ] Експортувати в CSV та проаналізувати bottlenecks

**Бажаю успішного профілювання! 🚀**
