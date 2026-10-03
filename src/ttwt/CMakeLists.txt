include(sources.cmake)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(WAVELET_SCHEME_CATALOG ${CMAKE_CURRENT_SOURCE_DIR}/schemes/wavelet_schemes.json)
set(WAVELET_SCHEME_GENERATOR ${CMAKE_CURRENT_SOURCE_DIR}/schemes/generate_wavelet_scheme_headers.py)
set(WAVELET_GENERATED_INCLUDE_ROOT ${PROJECT_BINARY_DIR}/generated)
set(WAVELET_GENERATED_HOST_INCLUDE_DIR ${WAVELET_GENERATED_INCLUDE_ROOT})
set(WAVELET_GENERATED_SCHEME_DIR
    ${WAVELET_GENERATED_HOST_INCLUDE_DIR}/ttwt/generated/wavelet_schemes
)

file(READ ${WAVELET_SCHEME_CATALOG} WAVELET_SCHEME_CATALOG_JSON)
string(JSON WAVELET_SCHEME_COUNT LENGTH "${WAVELET_SCHEME_CATALOG_JSON}")
if(NOT WAVELET_SCHEME_COUNT EQUAL 106)
    message(FATAL_ERROR "Expected 106 Wavelet schemes, found ${WAVELET_SCHEME_COUNT}")
endif()

set(WAVELET_GENERATED_SCHEME_HEADERS)
math(EXPR WAVELET_SCHEME_LAST_INDEX "${WAVELET_SCHEME_COUNT} - 1")
foreach(SCHEME_INDEX RANGE 0 ${WAVELET_SCHEME_LAST_INDEX})
    string(JSON SCHEME_NAME MEMBER "${WAVELET_SCHEME_CATALOG_JSON}" ${SCHEME_INDEX})
    string(REPLACE "." "_" SCHEME_IDENTIFIER ${SCHEME_NAME})
    list(APPEND WAVELET_GENERATED_SCHEME_HEADERS ${WAVELET_GENERATED_SCHEME_DIR}/${SCHEME_IDENTIFIER}.hpp)
endforeach()
list(
    APPEND
    WAVELET_GENERATED_SCHEME_HEADERS
    ${WAVELET_GENERATED_SCHEME_DIR}/scheme_catalog.hpp
    ${WAVELET_GENERATED_SCHEME_DIR}/scheme_dispatch.hpp
)
unset(WAVELET_SCHEME_CATALOG_JSON)

add_custom_command(
    OUTPUT
        ${WAVELET_GENERATED_SCHEME_HEADERS}
    COMMAND
        ${Python3_EXECUTABLE} ${WAVELET_SCHEME_GENERATOR} ${WAVELET_GENERATED_SCHEME_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS
        ${WAVELET_SCHEME_GENERATOR}
        ${WAVELET_SCHEME_CATALOG}
    COMMENT "Generating TTNN Wavelet scheme headers"
    VERBATIM
)

set_source_files_properties(
    ${WAVELET_GENERATED_SCHEME_HEADERS}
    PROPERTIES
        GENERATED
            TRUE
)

add_custom_target(tt_wavelet_schemes DEPENDS ${WAVELET_GENERATED_SCHEME_HEADERS})

add_library(tt_wavelet_native STATIC ${TTNN_OP_WAVELET_SRCS} ${WAVELET_GENERATED_SCHEME_HEADERS})
add_library(TTWavelet::Native ALIAS tt_wavelet_native)
get_filename_component(WAVELET_SOURCE_INCLUDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
target_include_directories(tt_wavelet_native BEFORE PUBLIC
    "${WAVELET_SOURCE_INCLUDE_ROOT}" "${WAVELET_GENERATED_HOST_INCLUDE_DIR}")
target_compile_definitions(tt_wavelet_native PRIVATE
    TTNN_WAVELET_GENERATED_INCLUDE_ROOT="${WAVELET_GENERATED_INCLUDE_ROOT}"
    TT_WAVELET_SOURCE_INCLUDE_ROOT="${WAVELET_SOURCE_INCLUDE_ROOT}"
    TT_WAVELET_KERNEL_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/device/kernels")
target_link_libraries(tt_wavelet_native PUBLIC TTNN::TTNN TT::Metalium)
# Match the pinned TTNN directory's existing compiler policy without changing
# authoritative runtime-argument initializers.
target_compile_options(tt_wavelet_native PRIVATE
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-narrowing>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-narrowing>")
