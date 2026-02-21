# Tracy Profiler Setup Guide

## Overview

This document describes the integration of Tracy Profiler v0.10.0 with the tt-wavelet project for Tenstorrent N300s hardware profiling.

Tracy is a real-time, nanosecond resolution profiler with hybrid frame and sampling support. The tt-metal repository includes a modified version with TT-specific device profiling capabilities.

## Prerequisites

- Ubuntu/Debian-based system
- clang-20 toolchain
- CMake 3.20+
- Required libraries: `nlohmann-json3-dev`, `libfreetype-dev`, `libcapstone-dev`, `libwayland-dev`, `libegl-dev`

Install dependencies:
```bash
sudo apt-get install -y nlohmann-json3-dev libfreetype-dev libcapstone-dev \
    libwayland-dev libwayland-egl1 libegl-dev libxkbcommon-dev libdbus-1-dev
```

## Building Tracy

### 1. Configure Tracy GUI Build

Tracy profiler uses Tenstorrent-specific device data structures that depend on the `enchantum` library. The enchantum library is already available through CPM at `tt-metal/.cpmcache/enchantum/*/enchantum/include`, but the Tracy GUI Makefile needs to be configured to find it.

**File to modify:** `tt-metal/tt_metal/third_party/tracy/profiler/build/unix/build.mk`

**Line 4** - Add enchantum include path to the `INCLUDES` variable:

**Original:**
```makefile
INCLUDES := -I../../../imgui $(shell pkg-config --cflags freetype2 capstone wayland-egl egl wayland-cursor xkbcommon) -I../../../../import-chrome/src/
```

**Modified:**
```makefile
INCLUDES := -I../../../imgui $(shell pkg-config --cflags freetype2 capstone wayland-egl egl wayland-cursor xkbcommon) -I../../../../import-chrome/src/ -I../../../../../../.cpmcache/enchantum/2fb7ab238e36c101b9848892ddb6382276b65837/enchantum/include
```

This enables the Tracy GUI to properly use `enchantum::to_string()`.

### 2. Build the Project with Tracy Enabled

Use the provided build script (**recommended**):

```bash
./scripts/build_with_tracy.sh [Debug|Release] [jobs]
```

Or build manually:

```bash
export ENABLE_TRACY=ON
./build.sh Release $(nproc)
```

### 3. Build Tracy GUI

```bash
cd tt-metal/tt_metal/third_party/tracy/profiler/build/unix
make -j$(nproc)
```

The Tracy GUI executable will be at: `tt-metal/tt_metal/third_party/tracy/profiler/build/unix/Tracy-release`

For convenience, create a symlink:
```bash
ln -sf tt-metal/tt_metal/third_party/tracy/profiler/build/unix/Tracy-release tracy-gui
```

### 3. Build Tracy Command-Line Tools (Optional)

```bash
cd tt-metal/tt_metal/third_party/tracy/capture/build/unix
make -j$(nproc)

cd ../../../csvexport/build/unix
make -j$(nproc)
```

## Demo Programs

### `n300s_demo.cpp`

Real TT-Metal program for N300s with Tracy profiling:
- Device initialization profiling
- DRAM buffer operations (write/read 64KB)
- Data verification
- Custom plots (buffer size, device memory)
- Color-coded messages

**Important:** Includes pause for Tracy connection:
```cpp
std::cout << "Waiting for Tracy profiler to connect..." << std::endl;
std::cout << "Connect Tracy GUI to this program, then press Enter to continue..." << std::endl;
std::cin.get();  // Wait for user to connect Tracy GUI
```

## Using Tracy

### 1. Instrument Your Code

Add Tracy zones to C++ code:

```cpp
#include <tracy/Tracy.hpp>

void my_function() {
    ZoneScoped;  // Automatically profiles the entire function
    
    // Your code here
    
    {
        ZoneScopedN("Inner Section");  // Named zone for specific code section
        // More code
    }
}

int main() {
    ZoneScoped;  // Profile main
    
    // IMPORTANT: For programs that run quickly, add a pause
    // to allow time to connect Tracy GUI before execution
    std::cout << "Press Enter to start profiling..." << std::endl;
    std::cin.get();
    
    // ... your code ...
    FrameMark;  // Mark frame boundaries for frame profiling
    return 0;
}
```

### 2. Remote Profiling (Live Mode - Recommended)

This workflow enables real-time profiling of N300s hardware from your local machine.

#### Prerequisites

**On N300s Server:**
- SSH access to N300s machine
- Project repository cloned
- Environment set up

**On Local Machine:**
- Tracy GUI built and ready
- SSH client

---

#### Step 1: Build

⚠️ **IMPORTANT:** Due to GLIBC version differences, you **must build directly on the N300s server**. Binaries built on newer systems (Ubuntu 24.04) won't run on older systems (Ubuntu 22.04).

**SSH into N300s:**
```bash
ssh -p <port> user@n300s-server
```

**Clone and build:**
```bash
# Clone repository
git clone <your-repo-url> tt-wavelet
cd tt-wavelet
git submodule update --init --recursive

# Build with Tracy enabled
./scripts/build_with_tracy.sh Release $(nproc)
```

This builds TT-Metal, Tracy, and your n300s_demo with matching system libraries.

**Verify build:**
```bash
cd build/tt-wavelet
ls -lh n300s_demo  # Should exist
```

---

#### Step 2: Create SSH Tunnel

Forward Tracy port (8086) from N300s to your local machine:

**On local machine:**
```bash
ssh -p 24756 -L 8086:localhost:8086 root@01.proxy.koyeb.app
```

This tunnel maps:
- `localhost:8086` (local) ← **Tracy listens here**
- `localhost:8086` (N300s) ← **n300s_demo connects here**

**Keep this SSH session open** during profiling!

---

#### Step 3: Launch Tracy GUI Locally

In a **new terminal** on your local machine:

```bash
./tracy-gui
```

---

#### Step 4: Run n300s_demo on N300s

In the **SSH tunnel terminal** (still connected to N300s):

```bash
# Set up environment
cd /root/tt-wavelet  # or your project path
source .venv/bin/activate
source scripts/set_env.sh  # Sets TT_METAL_HOME

# Run demo
cd build/tt-wavelet
./n300s_demo
```

**Expected output:**
```
Waiting for Tracy profiler to connect...
Connect Tracy GUI to this program, then press Enter to continue...
```

---

#### Step 5: Connect Tracy GUI

In **Tracy GUI window**:
1. Click **"Connect"** button (or it may auto-connect)
2. If prompted, select **"127.0.0.1"**

---

#### Step 6: Start Profiling

Back in the **SSH terminal** with n300s_demo:
- **Press Enter**

The program will execute and **Tracy GUI will show live profiling data in real-time**!

**What you'll see:**
- Timeline with function zones (Device Initialization, Buffer Allocation, etc.)
- Nested zones (CreateDevice → Cluster → initialize)
- Color-coded messages (cyan=init, green=success, magenta=complete)
- Execution times for each zone

---

#### Step 7: Save Trace (Optional)

After profiling completes in Tracy GUI:
1. **File → Save**
2. Choose filename (e.g., `n300s_profile.tracy`)
3. Saved traces can be reopened later: `./tracy-gui n300s_profile.tracy`

---

### Tracy API

| Action | Code |
|--------|------|
| Profile function | `ZoneScoped;` |
| Named zone | `ZoneScopedN("Name");` |
| Mark frame | `FrameMark;` |
| Log message | `TracyMessage("text", length);` |
| Colored message | `TracyMessageC("text", length, 0xRRGGBB);` |
| Plot value | `TracyPlot("name", value);` |
| Track allocation | `TracyAlloc(ptr, size);` |
| Track free | `TracyFree(ptr);` |

---

#### File Transfer (If Needed)

**Copy updated source files to N300s:**
```bash
# From local machine
scp -P <port> tt-wavelet/n300s_demo.cpp user@n300s:/path/to/tt-wavelet/tt-wavelet/

# Rebuild on N300s
ssh -p <port> user@n300s "cd /path/to/tt-wavelet/build && ninja n300s_demo"
```

**Copy trace files from N300s:**
```bash
scp -P <port> user@n300s:/path/to/trace.tracy ./
./tracy-gui trace.tracy
```

## Advanced Configuration

### Tracy Compile-Time Options

Edit tt-metal CMakeLists.txt to customize Tracy:

```cmake
# Disable on-demand profiling (always profile)
target_compile_definitions(tracy PUBLIC TRACY_ON_DEMAND)

# Set higher sampling rate
target_compile_definitions(tracy PUBLIC TRACY_SAMPLING_HZ=10000)

# Disable callstack capture
target_compile_definitions(tracy PUBLIC TRACY_NO_CALLSTACK)
```

### Environment Variables

```bash
# Change Tracy network port (default 8086)
export TRACY_PORT=9999

# Disable Tracy at runtime
export TRACY_NO_TRACE=1

# Set TT-Metal root (required for N300s demos)
export TT_METAL_HOME=/path/to/tt-metal_demo
```

This creates `n300s_demo.tracy` file.

#### On Your Local Machine:

```bash
# Copy trace file
scp user@koyeb:/path/to/n300s_demo.tracy ./

# Open in Tracy GUI
./Tracy-release n300s_demo.tracy
```

**When to use File Mode:**
- SSH tunnel issues
- Unstable network connection
- Want to archive traces for later analysis

## Tracy Features

### Zone Profiling

```cpp
void process_data() {
    ZoneScoped;  // Measures entire function
    
    {
        ZoneScopedN("Load Data");
        load_data();
    }
    
    {
        ZoneScopedN("Process");
        process();
    }
}
```
### Memory Profiling

```cpp
void* ptr = malloc(1024);
TracyAlloc(ptr, 1024);

// ... use memory ...

TracyFree(ptr);
free(ptr);
```

### Custom Plots

```cpp
TracyPlot("FPS", fps);
TracyPlot("Memory Usage", memory_bytes);
```

### Messages and Logs

```cpp
TracyMessage("Started processing batch", 24);
TracyMessageC("Error occurred", 14, 0xFF0000);  // red color
```

## Tenstorrent-Specific Features

### Device Markers

The tt-metal fork includes TT-specific device profiling:

```cpp
#include <tracy/public/common/TracyTTDeviceData.hpp>

// Device markers use enchantum for enum reflection
TTDeviceMarker marker;
marker.risc_type = RiscType::TENSIX;
marker.marker_type = TTDeviceMarkerType::ZONE_START;
marker.core_x = 0;
marker.core_y = 0;
```
---

*Updated: 21.02.26*
*Tracy Version: v0.10.0 (tt-metal fork)*
