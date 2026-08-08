cmake_minimum_required(VERSION 3.20)
project(tt-wavelet LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build TT-Metal natively in this build tree.  TT-Metal exports
# Metalium::Metal directly, so no local CMake replacement files are needed.
option(BUILD_TT_WAVELET "Build TT-Metal, TTNN, and tt-wavelet" ON)

if(BUILD_TT_WAVELET)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/tt-metal/CMakeLists.txt")
    message(
      FATAL_ERROR
        "BUILD_TT_WAVELET is ON but tt-metal not found at ${CMAKE_SOURCE_DIR}/tt-metal"
    )
  endif()

  set(CMAKE_DISABLE_PRECOMPILE_HEADERS TRUE CACHE BOOL "Disable PCH in the unified build")
  set(ENABLE_CCACHE TRUE CACHE BOOL "Enable ccache for TT-Metal")
  set(TT_UNITY_BUILDS ON CACHE BOOL "Enable TT-Metal unity builds")
  set(WITH_PYTHON_BINDINGS ON CACHE BOOL "Build TTNN Python bindings")
  set(CPM_SOURCE_CACHE "${CMAKE_SOURCE_DIR}/tt-metal/.cpmcache" CACHE PATH "TT-Metal CPM cache")

  add_subdirectory(tt-metal "${CMAKE_BINARY_DIR}/tt-metal")
endif()

add_subdirectory(tt-wavelet)
