get_filename_component(TT_WAVELET_METAL_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/tt-metal" ABSOLUTE)
if(NOT EXISTS "${TT_WAVELET_METAL_SOURCE_DIR}/tt_metal/third_party/umd/CMakeLists.txt")
    message(FATAL_ERROR "Initialize the pinned dependency: git submodule update --init --recursive")
endif()
find_package(Git REQUIRED)
execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY "${TT_WAVELET_METAL_SOURCE_DIR}"
    OUTPUT_VARIABLE _metal_revision OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT _metal_revision STREQUAL "bfeda7615633bf78c7c3bf4c7c21486831ddfbc4")
    message(FATAL_ERROR "TT-Metal revision mismatch: ${_metal_revision}")
endif()
# Options provided by the pinned revision's cmake/project_options.cmake.
set(WITH_PYTHON_BINDINGS ${TT_WAVELET_BUILD_PYTHON} CACHE BOOL "Build matching TTNN Python bindings" FORCE)
set(TT_METAL_BUILD_TESTS OFF CACHE BOOL "Build Metalium tests")
set(TTNN_BUILD_TESTS OFF CACHE BOOL "Build TTNN tests")
set(BUILD_PROGRAMMING_EXAMPLES OFF CACHE BOOL "Build Metalium examples")
set(BUILD_TT_TRAIN OFF CACHE BOOL "Build tt-train")
set(ENABLE_DISTRIBUTED OFF CACHE BOOL "Build MPI support")
# OFF leaves Metalium's unconditional SimulationChip helper references unresolved
# at this revision. Retain its documented default until upstream fixes that edge.
set(TT_UMD_BUILD_SIMULATION ON CACHE BOOL "Build required UMD simulation support")
set(ENABLE_TRACY OFF CACHE BOOL "Build profiling tools")
set(TT_INSTALL OFF CACHE BOOL "Enable dependency installation rules")
add_subdirectory("${TT_WAVELET_METAL_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/tt-metal" EXCLUDE_FROM_ALL)
# Symbol ownership: exclude the dependency's Wavelet without source patches.
get_target_property(_ops_links ttnn_op_operations_shared INTERFACE_LINK_LIBRARIES)
list(REMOVE_ITEM _ops_links TTNN::Ops::Wavelet)
set_property(TARGET ttnn_op_operations_shared PROPERTY INTERFACE_LINK_LIBRARIES "${_ops_links}")
get_target_property(_ops_sources ttnn_op_operations_shared INTERFACE_SOURCES)
list(REMOVE_ITEM _ops_sources "$<TARGET_OBJECTS:TTNN::Ops::Wavelet>")
set_property(TARGET ttnn_op_operations_shared PROPERTY INTERFACE_SOURCES "${_ops_sources}")

if(TT_WAVELET_BUILD_PYTHON)
    include(${CMAKE_CURRENT_LIST_DIR}/TTNNPython.cmake)
endif()
