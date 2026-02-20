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

set(FMT_CANDIDATES
    ${TT_METAL_INCLUDE_ROOT}
    ${TT_METAL_SOURCE_ROOT}/build/_deps/fmt-src/include
)
file(GLOB CPM_FMT_CANDIDATES "${TT_METAL_SOURCE_ROOT}/.cpmcache/fmt/*/include")
list(APPEND FMT_CANDIDATES ${CPM_FMT_CANDIDATES})
foreach(FMT_DIR IN LISTS FMT_CANDIDATES)
    if(EXISTS "${FMT_DIR}/fmt/format.h")
        list(APPEND METALIUM_INCLUDE_DIRS "${FMT_DIR}")
        break()
    endif()
endforeach()

set(SPDLOG_CANDIDATES
    ${TT_METAL_INCLUDE_ROOT}
    ${TT_METAL_SOURCE_ROOT}/build/_deps/spdlog-src/include
)
file(GLOB CPM_SPDLOG_CANDIDATES "${TT_METAL_SOURCE_ROOT}/.cpmcache/spdlog/*/include")
list(APPEND SPDLOG_CANDIDATES ${CPM_SPDLOG_CANDIDATES})
foreach(SPDLOG_DIR IN LISTS SPDLOG_CANDIDATES)
    if(EXISTS "${SPDLOG_DIR}/spdlog/spdlog.h")
        list(APPEND METALIUM_INCLUDE_DIRS "${SPDLOG_DIR}")
        break()
    endif()
endforeach()

set(TT_LOGGER_CANDIDATES
    ${TT_METAL_INCLUDE_ROOT}
    ${TT_METAL_SOURCE_ROOT}/build/_deps/tt-logger-src/include
)
file(GLOB CPM_TT_LOGGER_CANDIDATES "${TT_METAL_SOURCE_ROOT}/.cpmcache/tt-logger/*/include")
list(APPEND TT_LOGGER_CANDIDATES ${CPM_TT_LOGGER_CANDIDATES})
foreach(TT_LOGGER_DIR IN LISTS TT_LOGGER_CANDIDATES)
    if(EXISTS "${TT_LOGGER_DIR}/tt-logger/tt-logger.hpp")
        list(APPEND METALIUM_INCLUDE_DIRS "${TT_LOGGER_DIR}")
        break()
    endif()
endforeach()

set(ENCHANTUM_CANDIDATES
    ${TT_METAL_INCLUDE_ROOT}
    ${TT_METAL_SOURCE_ROOT}/build/_deps/enchantum-src/enchantum/include
    ${TT_METAL_SOURCE_ROOT}/build/_deps/enchantum-src/include
)
file(GLOB CPM_ENCHANTUM_CANDIDATES
    "${TT_METAL_SOURCE_ROOT}/.cpmcache/enchantum/*/enchantum/include"
    "${TT_METAL_SOURCE_ROOT}/.cpmcache/enchantum/*/include"
    "${TT_METAL_SOURCE_ROOT}/.cpmcache/enchantum/*/single_include"
)
list(APPEND ENCHANTUM_CANDIDATES ${CPM_ENCHANTUM_CANDIDATES})
foreach(ENCHANTUM_DIR IN LISTS ENCHANTUM_CANDIDATES)
    if(EXISTS "${ENCHANTUM_DIR}/enchantum/enchantum.hpp")
        list(APPEND METALIUM_INCLUDE_DIRS "${ENCHANTUM_DIR}")
        break()
    endif()
endforeach()

if(TT_METAL_PREBUILT_ROOT)
    set(TT_METAL_LIB_ROOT "${TT_METAL_PREBUILT_ROOT}/lib")
    function(find_prebuilt_shared_lib out_var base_name)
        unset(_direct_match)
        unset(_direct_match CACHE)
        find_library(_direct_match NAMES "${base_name}" PATHS "${TT_METAL_LIB_ROOT}" NO_DEFAULT_PATH)
        if(_direct_match)
            set(${out_var} "${_direct_match}" PARENT_SCOPE)
            return()
        endif()

        file(GLOB _versioned_candidates "${TT_METAL_LIB_ROOT}/lib${base_name}.so*")
        if(_versioned_candidates)
            list(SORT _versioned_candidates)
            list(GET _versioned_candidates 0 _fallback_match)
            set(${out_var} "${_fallback_match}" PARENT_SCOPE)
            return()
        endif()

        set(${out_var} "" PARENT_SCOPE)
    endfunction()

    find_prebuilt_shared_lib(TT_METAL_LIBRARY "tt_metal")
    find_prebuilt_shared_lib(DEVICE_LIBRARY "device")
    find_prebuilt_shared_lib(TT_STL_LIBRARY "tt_stl")
    find_prebuilt_shared_lib(TRACY_LIBRARY "tracy")
    find_prebuilt_shared_lib(FMT_LIBRARY "fmt")
    find_prebuilt_shared_lib(SPDLOG_LIBRARY "spdlog")
    set(METALIUM_RUNTIME_LIBS "")

    if(NOT TT_METAL_LIBRARY)
        message(FATAL_ERROR "libtt_metal.so not found in ${TT_METAL_LIB_ROOT}")
    endif()

    if(NOT TT_STL_LIBRARY)
        message(FATAL_ERROR "libtt_stl.so not found in ${TT_METAL_LIB_ROOT}")
    endif()

    if(NOT TRACY_LIBRARY)
        message(FATAL_ERROR "libtracy.so not found in ${TT_METAL_LIB_ROOT}")
    endif()

    if(NOT FMT_LIBRARY)
        message(FATAL_ERROR "libfmt.so not found in ${TT_METAL_LIB_ROOT}")
    endif()

    find_library(HWLOC_LIBRARY NAMES "hwloc")

    add_library(Metalium::Metal SHARED IMPORTED GLOBAL)
    set_target_properties(Metalium::Metal PROPERTIES
        IMPORTED_LOCATION "${TT_METAL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${METALIUM_INCLUDE_DIRS}"
    )

    list(APPEND METALIUM_RUNTIME_LIBS ${TT_STL_LIBRARY} ${TRACY_LIBRARY} ${FMT_LIBRARY})

    if(DEVICE_LIBRARY)
        list(APPEND METALIUM_RUNTIME_LIBS ${DEVICE_LIBRARY})
    endif()

    if(SPDLOG_LIBRARY)
        list(APPEND METALIUM_RUNTIME_LIBS ${SPDLOG_LIBRARY})
    endif()

    if(HWLOC_LIBRARY)
        list(APPEND METALIUM_RUNTIME_LIBS ${HWLOC_LIBRARY})
    endif()
else()
    add_library(Metalium::Metal INTERFACE IMPORTED GLOBAL)
    set_target_properties(Metalium::Metal PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${METALIUM_INCLUDE_DIRS}"
    )
endif()

add_library(tt_wavelet_analysis OBJECT "${TT_WAVELET_SOURCE_ROOT}/tt-wavelet/main.cpp")
target_link_libraries(tt_wavelet_analysis PRIVATE Metalium::Metal)
if(TT_METAL_PREBUILT_ROOT)
    target_link_libraries(tt_wavelet_analysis PRIVATE ${METALIUM_RUNTIME_LIBS})
endif()

if(CI_BUILD_WAVELET_EXECUTABLE)
    if(NOT TT_METAL_PREBUILT_ROOT)
        message(FATAL_ERROR "CI_BUILD_WAVELET_EXECUTABLE=ON requires TT_METAL_PREBUILT_ROOT")
    endif()

    add_executable(tt_wavelet_test "${TT_WAVELET_SOURCE_ROOT}/tt-wavelet/main.cpp")
    target_link_libraries(tt_wavelet_test PRIVATE Metalium::Metal)
    target_link_libraries(tt_wavelet_test PRIVATE ${METALIUM_RUNTIME_LIBS})
endif()
