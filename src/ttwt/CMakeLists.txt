include(sources.cmake)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(TT_WAVELET_SCHEME_CATALOG ${CMAKE_CURRENT_SOURCE_DIR}/schemes/wavelet_schemes.json)
set(TT_WAVELET_SCHEME_GENERATOR ${CMAKE_CURRENT_SOURCE_DIR}/schemes/generate_wavelet_scheme_headers.py)
set(TT_WAVELET_GENERATED_INCLUDE_ROOT ${PROJECT_BINARY_DIR}/generated)
set(TT_WAVELET_GENERATED_SCHEME_DIR
    ${TT_WAVELET_GENERATED_INCLUDE_ROOT}/ttwt/generated/wavelet_schemes
)

file(READ ${TT_WAVELET_SCHEME_CATALOG} TT_WAVELET_SCHEME_CATALOG_JSON)
string(JSON TT_WAVELET_SCHEME_COUNT LENGTH "${TT_WAVELET_SCHEME_CATALOG_JSON}")
if(NOT TT_WAVELET_SCHEME_COUNT EQUAL 106)
    message(FATAL_ERROR "Expected 106 Wavelet schemes, found ${TT_WAVELET_SCHEME_COUNT}")
endif()

set(TT_WAVELET_GENERATED_SCHEME_HEADERS)
math(EXPR TT_WAVELET_SCHEME_LAST_INDEX "${TT_WAVELET_SCHEME_COUNT} - 1")
foreach(SCHEME_INDEX RANGE 0 ${TT_WAVELET_SCHEME_LAST_INDEX})
    string(JSON SCHEME_NAME MEMBER "${TT_WAVELET_SCHEME_CATALOG_JSON}" ${SCHEME_INDEX})
    string(REPLACE "." "_" SCHEME_IDENTIFIER ${SCHEME_NAME})
    list(APPEND TT_WAVELET_GENERATED_SCHEME_HEADERS ${TT_WAVELET_GENERATED_SCHEME_DIR}/${SCHEME_IDENTIFIER}.hpp)
endforeach()
list(
    APPEND
    TT_WAVELET_GENERATED_SCHEME_HEADERS
    ${TT_WAVELET_GENERATED_SCHEME_DIR}/scheme_catalog.hpp
    ${TT_WAVELET_GENERATED_SCHEME_DIR}/scheme_dispatch.hpp
)
unset(TT_WAVELET_SCHEME_CATALOG_JSON)

add_custom_command(
    OUTPUT
        ${TT_WAVELET_GENERATED_SCHEME_HEADERS}
    COMMAND
        ${Python3_EXECUTABLE} ${TT_WAVELET_SCHEME_GENERATOR} ${TT_WAVELET_GENERATED_SCHEME_DIR}
    DEPENDS
        ${TT_WAVELET_SCHEME_GENERATOR}
        ${TT_WAVELET_SCHEME_CATALOG}
    COMMENT "Generating TT-Wavelet scheme headers"
    VERBATIM
)

set_source_files_properties(
    ${TT_WAVELET_GENERATED_SCHEME_HEADERS}
    PROPERTIES
        GENERATED
            TRUE
)

add_custom_target(tt_wavelet_schemes DEPENDS ${TT_WAVELET_GENERATED_SCHEME_HEADERS})

add_library(tt_wavelet_native STATIC ${TT_WAVELET_HEADERS} ${TT_WAVELET_SRCS} ${TT_WAVELET_GENERATED_SCHEME_HEADERS})
add_library(TTWavelet::Native ALIAS tt_wavelet_native)
get_filename_component(TT_WAVELET_HOST_INCLUDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
target_include_directories(tt_wavelet_native BEFORE PUBLIC
    "${TT_WAVELET_HOST_INCLUDE_ROOT}" "${TT_WAVELET_GENERATED_INCLUDE_ROOT}")
include(resources.cmake)
add_dependencies(tt_wavelet_native tt_wavelet_resources)
if(NOT TT_WAVELET_RELOCATABLE_RESOURCES)
    target_compile_definitions(tt_wavelet_native PRIVATE
        TT_WAVELET_DEVELOPMENT_RESOURCE_ROOT="${TT_WAVELET_RESOURCE_ROOT}")
endif()
target_link_libraries(tt_wavelet_native PUBLIC TTNN::TTNN TT::Metalium)

target_compile_options(tt_wavelet_native PRIVATE
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-narrowing>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-narrowing>")

if(TT_WAVELET_RELOCATABLE_RESOURCES)
    target_compile_options(tt_wavelet_native PRIVATE
        "-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=ttwt"
        "-ffile-prefix-map=${TT_WAVELET_GENERATED_INCLUDE_ROOT}=generated")
endif()
