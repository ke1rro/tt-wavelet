# Tracy Profiler Setup Guide

## Overview

This document describes the integration of Tracy Profiler with the tt-wavelet project for Tenstorrent N300s hardware profiling.

## Prerequisites

- Ubuntu/Debian-based system
- clang-20 toolchain
- CMake 3.20+
- **Required libraries**: `nlohmann-json3-dev`, `libfreetype-dev`, `libcapstone-dev`, `libwayland-dev`, `libegl-dev`

Install dependencies:
```bash
sudo apt-get install -y nlohmann-json3-dev libfreetype-dev libcapstone-dev \
    libwayland-dev libwayland-egl1 libegl-dev libxkbcommon-dev libdbus-1-dev
```

## Building Tracy

### 1. Build the Project with Tracy Enabled

Use the provided build script:

```bash
./scripts/build_with_tracy.sh [Debug|Release] [jobs]
```

### 2. Build Tracy GUI

Use the automated build script:

```bash
./scripts/build_tracy_gui.sh
```

This script automatically:
- Builds Tracy GUI
- Creates `./tracy-gui` symlink for convenience

**Output:**
- Tracy GUI executable: `tt-metal/tt_metal/third_party/tracy/profiler/build/unix/Tracy-release`
- Symlink: `./tracy-gui` (created in project root)

### 3. Build Tracy Command-Line Tools (Optional)

```bash
cd tt-metal/tt_metal/third_party/tracy/capture/build/unix
make -j$(nproc)

cd ../../../csvexport/build/unix
make -j$(nproc)
```


### 2. Remote Profiling

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

---

#### Step 2: Create SSH Tunnel

Forward Tracy port (8086) from N300s to your local machine:

**On local machine:**
```bash
ssh -p <port> -L 8086:localhost:8086 root@01.proxy.koyeb.app
```

This tunnel maps:
- `localhost:8086` (local) ← **Tracy listens here**
- `localhost:8086` (N300s) ← **main connects here**

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
./main
```

---

#### Step 5: Connect Tracy GUI

In **Tracy GUI window**:
1. Click **"Connect"** button (or it may auto-connect)
2. If prompted, select **"127.0.0.1"**

---

#### Step 6: Start Profiling

The program will execute and **Tracy GUI will show live profiling data in real-time**!

---

#### Step 7: Save Trace (Optional)

After profiling completes in Tracy GUI:
1. **File → Save**
2. Choose filename (e.g., `main.tracy`)
3. Saved traces can be reopened later

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
scp -P <port> tt-wavelet/main.cpp user@n300s:/path/to/tt-wavelet/tt-wavelet/

# Rebuild on N300s
ssh -p <port> user@n300s "cd /path/to/tt-wavelet/build && ninja main"
```

**Copy trace files from N300s:**
```bash
scp -P <port> user@n300s:/path/to/trace.tracy ./
./tracy-gui trace.tracy
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

*Updated: 02.03.26*
