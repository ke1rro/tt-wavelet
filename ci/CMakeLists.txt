cmake_minimum_required(VERSION 3.20)
project(tt-wavelet-ci LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TT_WAVELET_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH "Path to tt-wavelet repository root")
set(TT_METAL_SOURCE_ROOT "${TT_WAVELET_SOURCE_ROOT}/tt-metal" CACHE PATH "Path to tt-metal sources")
set(TT_METAL_PREBUILT_ROOT "" CACHE PATH "Path to prebuilt tt-metal package (include/ + lib/)")
option(CI_BUILD_WAVELET_EXECUTABLE "Build tt_wavelet_test executable in CI config" ON)

if(TT_METAL_PREBUILT_ROOT)
    set(TT_METAL_INCLUDE_ROOT "${TT_METAL_PREBUILT_ROOT}/include")
else()
    set(TT_METAL_INCLUDE_ROOT "${TT_METAL_SOURCE_ROOT}")
endif()

set(METALIUM_INCLUDE_DIRS
    ${TT_METAL_INCLUDE_ROOT}
    ${TT_METAL_INCLUDE_ROOT}/ttnn
    ${TT_METAL_INCLUDE_ROOT}/ttnn/cpp
    ${TT_METAL_INCLUDE_ROOT}/ttnn/cpp/ttnn/deprecated
    ${TT_METAL_INCLUDE_ROOT}/tt_metal/api
    ${TT_METAL_INCLUDE_ROOT}/tt_metal
    ${TT_METAL_INCLUDE_ROOT}/tt_metal/include
    ${TT_METAL_INCLUDE_ROOT}/tt_metal/hostdevcommon/api
    ${TT_METAL_INCLUDE_ROOT}/tt_metal/third_party/umd/device/api
    ${TT_METAL_INCLUDE_ROOT}/tt_metal/hw/inc
    ${TT_METAL_INCLUDE_ROOT}/tt_stl
    ${TT_METAL_INCLUDE_ROOT}/tt_stl/tt_stl
)

set(REFLECT_CANDIDATES
    ${TT_METAL_INCLUDE_ROOT}
    ${TT_METAL_SOURCE_ROOT}/build/_deps/reflect-src
)
file(GLOB CPM_REFLECT_CANDIDATES
    "${TT_METAL_SOURCE_ROOT}/.cpmcache/reflect/*"
)
list(APPEND REFLECT_CANDIDATES ${CPM_REFLECT_CANDIDATES})

foreach(REFLECT_DIR IN LISTS REFLECT_CANDIDATES)
    if(EXISTS "${REFLECT_DIR}/reflect")
        list(APPEND METALIUM_INCLUDE_DIRS "${REFLECT_DIR}")
        break()
    endif()
endforeach()

set(NLOHMANN_CANDIDATES
    ${TT_METAL_INCLUDE_ROOT}
    ${TT_METAL_SOURCE_ROOT}/build/_deps/nlohmann_json-src/include
    ${TT_METAL_SOURCE_ROOT}/build/_deps/nlohmann_json-src/single_include
    /usr/include
)
file(GLOB CPM_NLOHMANN_CANDIDATES
    "${TT_METAL_SOURCE_ROOT}/.cpmcache/nlohmann_json/*/include"
    "${TT_METAL_SOURCE_ROOT}/.cpmcache/nlohmann_json/*/single_include"
)
list(APPEND NLOHMANN_CANDIDATES ${CPM_NLOHMANN_CANDIDATES})

foreach(NLOHMANN_DIR IN LISTS NLOHMANN_CANDIDATES)
    if(EXISTS "${NLOHMANN_DIR}/nlohmann/json_fwd.hpp")
        list(APPEND METALIUM_INCLUDE_DIRS "${NLOHMANN_DIR}")
        break()
    endif()
endforeach()

if(TT_METAL_PREBUILT_ROOT)
    set(TT_METAL_LIB_ROOT "${TT_METAL_PREBUILT_ROOT}/lib")
    find_library(TT_METAL_LIBRARY NAMES "tt_metal" PATHS "${TT_METAL_LIB_ROOT}" NO_DEFAULT_PATH)
    find_library(DEVICE_LIBRARY NAMES "device" PATHS "${TT_METAL_LIB_ROOT}" NO_DEFAULT_PATH)

    if(NOT TT_METAL_LIBRARY)
        message(FATAL_ERROR "libtt_metal.so not found in ${TT_METAL_LIB_ROOT}")
    endif()

    add_library(Metalium::Metal SHARED IMPORTED GLOBAL)
    set_target_properties(Metalium::Metal PROPERTIES
        IMPORTED_LOCATION "${TT_METAL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${METALIUM_INCLUDE_DIRS}"
    )

    if(DEVICE_LIBRARY)
        target_link_libraries(Metalium::Metal INTERFACE ${DEVICE_LIBRARY})
    endif()
else()
    add_library(Metalium::Metal INTERFACE IMPORTED GLOBAL)
    set_target_properties(Metalium::Metal PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${METALIUM_INCLUDE_DIRS}"
    )
endif()

add_library(tt_wavelet_analysis OBJECT "${TT_WAVELET_SOURCE_ROOT}/tt-wavelet/main.cpp")
target_link_libraries(tt_wavelet_analysis PRIVATE Metalium::Metal)

if(CI_BUILD_WAVELET_EXECUTABLE)
    if(NOT TT_METAL_PREBUILT_ROOT)
        message(FATAL_ERROR "CI_BUILD_WAVELET_EXECUTABLE=ON requires TT_METAL_PREBUILT_ROOT")
    endif()

    add_executable(tt_wavelet_test "${TT_WAVELET_SOURCE_ROOT}/tt-wavelet/main.cpp")
    target_link_libraries(tt_wavelet_test PRIVATE Metalium::Metal)
endif()
