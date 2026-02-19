# Tracy Profiling: Build Troubleshooting Guide

## вступ
тут будуть зібрані усі проблеми та рішення до них з якими я (sodaqqq) стикнувся при білді проекту з tracy.\

@sodaqq_ моя інста підписуйтесь

## Некоректні шляхи до tt-metal підмодуля

### Опис проблеми

По перше, неправильно вказані шляхи до tt-metal.

```
CMake Error: Could not find directory: /home/user/tt-wavelet/tt-metal
```

Структура проекту була змінена - tt-metal перемістили в `third-party/`, але конфігураційні файли не оновили.

### Рішення

Оновити всі посилання на tt-metal у наступних файлах:

**scripts/common.sh**:
```bash
# Було
TT_METAL_DIR="$ROOT_DIR/tt-metal"

# Стало
TT_METAL_DIR="$ROOT_DIR/third-party/tt-metal"
```

**CMakeLists.txt**:
```cmake
# Було
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/tt-metal")

# Стало
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third-party/tt-metal")
```

**cmake/tt-metal.cmake**:
```cmake
# Було
"${CMAKE_SOURCE_DIR}/tt-metal/tt_metal"

# Стало
"${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_metal"
```

## Відсутність MPI для distributed support

CMake конфігурація завершувалася з помилкою через відсутність MPI (Message Passing Interface), необхідного для distributed compute.

MPI (Message Passing Interface) - це стандарт для паралельного програмування на кластерах. Дозволяє запускати програму одночасно на багатьох машинах/процесорах і обмінюватися даними між ними.

У контексті tt-metal: якщо у кілька N300 карт, MPI дозволяє розподілити обчислення між ними.

Для профілювання однієї N300 це не потрібно (наш варік).

### Причина

За замовчуванням tt-metal має увімкнений `ENABLE_DISTRIBUTED`, але MPI не встановлено в системі. Для базового профілювання Tracy distributed support не потрібен.

### Рішення

Додати прапорець `-DENABLE_DISTRIBUTED=OFF` до CMake конфігурації:

**scripts/common.sh**:
```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-20 \
    -DCMAKE_CXX_COMPILER=clang++-20 \
    -DENABLE_TRACY=ON \
    -DENABLE_DISTRIBUTED=OFF \
    # ... інші прапорці
```

Або прямий виклик CMake:
```bash
cmake -S . -B build -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_TRACY=ON \
    -DENABLE_DISTRIBUTED=OFF
```

### Альтернатива

Встановити MPI (OpenMPI або MPICH):
```bash
sudo apt-get install libopenmpi-dev openmpi-bin
```
для локального профілювання це не обов'язково.

## Відсутні директорії hardware toolchain

Під час збірки hardware компонентів CMake намагався створити linker scripts (`.ld` файли) у директоріях `runtime/hw/toolchain/{blackhole,grayskull,wormhole,wormhole_b0}`, які не існували.

```
error: unable to open output file '/home/user/tt-wavelet/third-party/tt-metal/runtime/hw/toolchain/blackhole/firmware_aerisc.ld': 'No such file or directory'
1 error generated.
gmake[2]: *** [firmware_aerisc.ld] Error 1
```

CMake очікує, що директорії toolchain вже існують для всіх підтримуваних архітектур Tenstorrent.

### Рішення

Створити відсутні директорії перед збіркою:

```bash
mkdir -p third-party/tt-metal/runtime/hw/toolchain/blackhole
mkdir -p third-party/tt-metal/runtime/hw/toolchain/grayskull
mkdir -p third-party/tt-metal/runtime/hw/toolchain/wormhole
mkdir -p third-party/tt-metal/runtime/hw/toolchain/wormhole_b0
mkdir -p third-party/tt-metal/runtime/hw/toolchain/quasar
```

Або одним командою:
```bash
mkdir -p third-party/tt-metal/runtime/hw/toolchain/{blackhole,grayskull,wormhole,wormhole_b0,quasar}
```

## Недостатні права запису в toolchain директорії

### Опис проблеми

Навіть після створення директорій, збірка могла провалюватися через недостатні права запису. CMake не міг записати згенеровані linker scripts.

### Рішення

Переконатися, що поточний користувач має права запису:

```bash
chmod -R u+w third-party/tt-metal/runtime/hw/toolchain/
```

Або встановити повні права:
```bash
chmod -R 755 third-party/tt-metal/runtime/hw/toolchain/
```

## Tracy Profiler GUI

Tracy GUI не збирається автоматично разом з tt-metal. Потрібна окрема збірка з `profiler/build/unix/`. При спробі збірки виникають наступні помилки:

```
fatal error: enchantum/enchantum.hpp: No such file or directory
fatal error: nlohmann/json.hpp: No such file or directory
```

Tracy Profiler GUI має власну систему збірки (Unix Makefiles) і потребує Tenstorrent-специфічних залежностей, які не включені в стандартні include paths.

### Рішення

1. **Встановити системні залежності GUI:**

```bash
sudo apt-get install \
    libfreetype6-dev \
    libcapstone-dev \
    libwayland-dev \
    libegl-dev \
    libxkbcommon-dev \
    libdbus-1-dev
```

2. **Знайти шляхи до залежностей:**

```bash
# enchantum
find third-party/tt-metal/.cpmcache -path "*/enchantum/*/include" -type d

# nlohmann_json
find third-party/tt-metal/.cpmcache -path "*/nlohmann_json/*/include" -type d
```

3. **Модифікувати build.mk:**

Редагувати `third-party/tt-metal/tt_metal/third_party/tracy/profiler/build/unix/build.mk`:

```makefile
INCLUDES := -I../../../imgui \
    $(shell pkg-config --cflags freetype2 capstone wayland-egl egl wayland-cursor xkbcommon) \
    -I../../../../import-chrome/src/ \
    -I/home/user/tt-wavelet/third-party/tt-metal/.cpmcache/enchantum/<hash>/enchantum/include \
    -I/home/user/tt-wavelet/third-party/tt-metal/.cpmcache/nlohmann_json/<hash>/include
```

**Важливо:** Замінити `<hash>` на актуальні значення з `.cpmcache/`.

4. **Зібрати Tracy GUI:**

```bash
cd third-party/tt-metal/tt_metal/third_party/tracy/profiler/build/unix
make -j$(nproc)
```

5. **Перевірити результат:**

```bash
ls -lh Tracy-release
# рчікується: executable ~19MB
```

### Автоматизований скрипт build_tracy_gui.sh

```bash
#!/bin/bash
set -e

ROOT_DIR="/home/$(whoami)/tt-wavelet"
PROFILER_DIR="$ROOT_DIR/third-party/tt-metal/tt_metal/third_party/tracy/profiler/build/unix"
CPMCACHE="$ROOT_DIR/third-party/tt-metal/.cpmcache"

echo "Finding dependency paths..."
ENCHANTUM_PATH=$(find "$CPMCACHE" -path "*/enchantum/*/enchantum/include" -type d | head -1)
JSON_PATH=$(find "$CPMCACHE" -path "*/nlohmann_json/*/include" -type d | head -1)

if [[ -z "$ENCHANTUM_PATH" ]] || [[ -z "$JSON_PATH" ]]; then
    echo "Error: Cannot find enchantum or nlohmann_json in .cpmcache"
    exit 1
fi

echo "Enchantum: $ENCHANTUM_PATH"
echo "JSON: $JSON_PATH"

cd "$PROFILER_DIR"

# Backup original build.mk
cp build.mk build.mk.bak

# Update INCLUDES in build.mk
sed -i "s|^\(INCLUDES := .*\)|\1 -I$ENCHANTUM_PATH -I$JSON_PATH|" build.mk

echo "Building Tracy GUI..."
make -j$(nproc)

echo "Tracy GUI built successfully: $(pwd)/Tracy-release"
```

## інше
перший запуск довгий.

Процес git clone займає багато часу для великих залежностей.

Залежності кешуються в `third-party/tt-metal/.cpmcache/`.


**Перший раз**: просто зачекати завершення. Це одноразова операція.

**надалі:**
1. Використовувати кеш CPM - при повторній збірці залежності беруться з `.cpmcache/`
2. Не видаляти директорію `third-party/tt-metal/.cpmcache/`
3. Використовувати ccache для прискорення компіляції: `-DENABLE_CCACHE=TRUE`

### Моніторинг процесу

```bash
watch -n 5 'ls -1 third-party/tt-metal/.cpmcache/ | wc -l'
```

## Рекомендований процес збірки з Tracy

```bash
sudo apt-get update
sudo apt-get install clang-20

clang-20 --version
```


```bash
cd /path/to/tt-wavelet

mkdir -p third-party/tt-metal/runtime/hw/toolchain/{blackhole,grayskull,wormhole,wormhole_b0,quasar}

chmod -R u+w third-party/tt-metal/runtime/hw/toolchain/
```

```bash
export TT_METAL_HOME="$(pwd)/third-party/tt-metal"

cmake -S . -B build -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-20 \
    -DCMAKE_CXX_COMPILER=clang++-20 \
    -DBUILD_TT_WAVELET=ON \
    -DENABLE_TRACY=ON \
    -DENABLE_DISTRIBUTED=OFF \
    -DCMAKE_DISABLE_PRECOMPILE_HEADERS=TRUE \
    -DENABLE_CCACHE=TRUE \
    -DTT_UNITY_BUILDS=ON \
    -DWITH_PYTHON_BINDINGS=ON \
    -DCMAKE_CXX_FLAGS="-std=c++20"
```

```bash
cmake --build build -j8
```

### Перевірка результатів

```bash
ls -lh build/third-party/tt-metal/tools/profiler/bin/
# Очікується: capture-release, csvexport-release

ls -lh build/third-party/tt-metal/lib/libtracy*

# tt-wavelet binary
ls -lh build/tt-wavelet/tt_wavelet_test

# Перевірка Tracy в CMake cache
grep "ENABLE_TRACY" build/CMakeCache.txt
# Очікується: ENABLE_TRACY:BOOL=ON
```

### Скрипт setup_tracy.sh


```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
TT_METAL_DIR="$ROOT_DIR/third-party/tt-metal"

echo "Preparing tt-wavelet build with Tracy profiling..."

if ! command -v clang-20 &> /dev/null; then
    echo "Error: clang-20 not found. Install it first:"
    echo "  sudo apt-get install clang-20"
    exit 1
fi

echo "Creating hardware toolchain directories..."
mkdir -p "$TT_METAL_DIR/runtime/hw/toolchain"/{blackhole,grayskull,wormhole,wormhole_b0,quasar}
chmod -R u+w "$TT_METAL_DIR/runtime/hw/toolchain/"

export TT_METAL_HOME="$TT_METAL_DIR"

echo "Configuring CMake with Tracy enabled..."
cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-20 \
    -DCMAKE_CXX_COMPILER=clang++-20 \
    -DBUILD_TT_WAVELET=ON \
    -DENABLE_TRACY=ON \
    -DENABLE_DISTRIBUTED=OFF \
    -DCMAKE_DISABLE_PRECOMPILE_HEADERS=TRUE \
    -DENABLE_CCACHE=TRUE \
    -DTT_UNITY_BUILDS=ON \
    -DWITH_PYTHON_BINDINGS=ON \
    -DCMAKE_CXX_FLAGS="-std=c++20"

echo "Configuration complete. Run: cmake --build build -j\$(nproc)"
```

### Використання

```bash
chmod +x scripts/setup_tracy.sh
./scripts/setup_tracy.sh
cmake --build build -j$(nproc)
```

## Використання Tracy Profiler

<mark>станом на 20 лютого я ше нічо не пробував</mark>

### Remote Profiling

Використовуйте **offline profiling** з capture-release.

#### Крок 1: Підготовка на N300

```bash
# SSH до N300
ssh user@n300-host

cd /path/to/tt-wavelet
export TT_METAL_HOME="$(pwd)/third-party/tt-metal"
export ARCH_NAME=wormhole_b0для N150

tt-smi

# check capture tool
ls -lh build/third-party/tt-metal/tools/profiler/bin/capture-release
```

#### Крок 2: Запуск 

**Варіант A: Capture + програма в різних терміналах**

Це якийсь молдованскій метод мені здається.

Термінал 1 (capture):
```bash
build/third-party/tt-metal/tools/profiler/bin/capture-release \
    -o wavelet_trace_$(date +%Y%m%d_%H%M%S).tracy \
    -a 127.0.0.1 \
    -p 8086

# capture чекає на підключення програми...
```

Термінал 2 (програма):
```bash
cd build/tt-wavelet
./tt_wavelet_test

# програма підключиться до capture, виконає роботу, trace збережеться
```

**Варіант B: Автоматичний запуск з timeout**

Поки шо буду користуватися таким.

```bash
# capture stops after 60 sec
timeout 60 build/third-party/tt-metal/tools/profiler/bin/capture-release \
    -o wavelet_trace.tracy \
    -a 127.0.0.1 &

sleep 2

build/tt-wavelet/tt_wavelet_test

wait

ls -lh wavelet_trace.tracy
```

**Варіант C: Скрипт для автоматизації**


Це продвінутий варіант, гляну потім.


#### Крок 3: Передача trace на локал

```bash
scp user@n300-host:/path/to/tt-wavelet/wavelet_trace.tracy ~/Downloads/

# або rsync для великих файлів
rsync -avz --progress user@n300-host:/path/to/tt-wavelet/*.tracy ~/traces/

# або взагалі модно через Koyeb CLI
koyeb files download wavelet_trace.tracy
```

#### Крок 4: Аналіз в Tracy GUI

```bash
cd ~/tt-wavelet

third-party/tt-metal/tt_metal/third_party/tracy/profiler/build/unix/Tracy-release &
```

Ще не аналізував нічо.

### Запуск Tracy GUI (локально)

```bash
third-party/tt-metal/tt_metal/third_party/tracy/profiler/build/unix/Tracy-release &

