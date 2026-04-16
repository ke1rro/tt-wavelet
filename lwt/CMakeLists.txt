cmake_minimum_required(VERSION 3.14)
project(tt_wavelet_baseline)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add tt-wavelet library includes
add_library(tt_wavelet INTERFACE)
target_include_directories(tt_wavelet INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../tt-wavelet/include)

add_executable(lwt main.cpp)
target_include_directories(lwt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lwt tt_wavelet)
