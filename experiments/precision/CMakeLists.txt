cmake_minimum_required(VERSION 3.28)

set(PROJECT_NAME "wavelet_test")

project(${PROJECT_NAME} CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)

add_executable(${PROJECT_NAME} src/main.cpp src/Benchmark.cpp src/Configuration.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} PRIVATE tomlplusplus::tomlplusplus)
