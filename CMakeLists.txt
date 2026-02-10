cmake_minimum_required(VERSION 3.20)
project(tt-wavelet LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to build with TT-Metal (Wormhole) support
option(BUILD_TT_WAVELET "Build with TT-Metal (Wormhole) support" OFF)

if(BUILD_TT_WAVELET)
    if(EXISTS "${CMAKE_SOURCE_DIR}/tt-metal/CMakeLists.txt")
        include("${CMAKE_SOURCE_DIR}/cmake/tt-metal.cmake")
    else()
        message(FATAL_ERROR "BUILD_TT_WAVELET is ON but tt-metal not found at ${CMAKE_SOURCE_DIR}/tt-metal")
    endif()
endif()

# Add main project
add_subdirectory(tt-wavelet)
