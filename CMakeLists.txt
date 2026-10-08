cmake_minimum_required(VERSION 3.24)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" TT_WAVELET_VERSION LIMIT_COUNT 1)
project(
  tt-wavelet
  VERSION ${TT_WAVELET_VERSION}
  LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE)
endif()
option(TT_WAVELET_RELOCATABLE_RESOURCES "Require an explicitly initialized runtime resource root"
       OFF)
option(TT_WAVELET_BUILD_PYTHON "Build the ttwt Python companion" OFF)
if(SKBUILD_STATE STREQUAL "wheel" AND (NOT TT_WAVELET_BUILD_PYTHON
                                       OR NOT TT_WAVELET_RELOCATABLE_RESOURCES))
  message(FATAL_ERROR "Wheel builds require Python bindings and relocatable resources")
endif()
include(cmake/Dependencies.cmake)
add_subdirectory(src/ttwt)

if(TT_WAVELET_BUILD_PYTHON)
  add_subdirectory(bindings)
endif()

option(TT_WAVELET_BUILD_BENCHMARKS "Build native performance runner" OFF)
if(TT_WAVELET_BUILD_BENCHMARKS)
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR "TT-Wavelet performance benchmarks require Release")
  endif()
  add_subdirectory(benchmarks)
endif()
