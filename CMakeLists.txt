cmake_minimum_required(VERSION 3.20)
project(tt-wavelet LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to build with TT-Metal (Wormhole) support
option(BUILD_TT_WAVELET "Build with TT-Metal (Wormhole) support" OFF)

# TT-Metal build options
set(WITH_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(TT_METAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TTNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_PROGRAMMING_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TT_TRAIN OFF CACHE BOOL "" FORCE)
set(TT_UNITY_BUILDS OFF CACHE BOOL "" FORCE)
set(TT_INSTALL OFF CACHE BOOL "" FORCE)
set(ENABLE_TRACY OFF CACHE BOOL "" FORCE)
set(ENABLE_DISTRIBUTED OFF CACHE BOOL "" FORCE)

# Add third-party dependencies (includes tt-metal when BUILD_TT_WAVELET is ON)
add_subdirectory(third-party)

# Add main project
add_subdirectory(tt-wavelet)
