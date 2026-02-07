cmake_minimum_required(VERSION 3.20)
project(tt-wavelet LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to build with TT-Metal (Wormhole) support
option(BUILD_TT_WAVELET "Build with TT-Metal (Wormhole) support" OFF)

# Add third-party dependencies (includes tt-metal when BUILD_TT_WAVELET is ON)
if(BUILD_TT_WAVELET)
    add_subdirectory(third-party)
endif()

# Add main project
add_subdirectory(tt-wavelet)
