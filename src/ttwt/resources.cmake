# Runtime JIT closure verified by source traversal and Phase 4A dependency manifests. Paths are
# relative to src/ttwt; these are runtime resources, not a C++ SDK.
set(TT_WAVELET_RUNTIME_FILES
    common/boundary.hpp
    common/signal_extension.hpp
    common/storage_contract.hpp
    device/kernels/compute/lwt_2d_compute.cpp
    device/kernels/compute/lwt_compute.cpp
    device/kernels/dataflow/lwt_2d_reader.cpp
    device/kernels/dataflow/lwt_2d_writer.cpp
    device/kernels/dataflow/lwt_reader.cpp
    device/kernels/dataflow/lwt_writer.cpp
    device/kernels/primitives/blackhole/llk_math_unary_datacopy_32x16_api.h
    device/kernels/primitives/wormhole/llk_math_unary_datacopy_32x16_api.h
    device/kernels/primitives/config_page.hpp
    device/kernels/primitives/interleave.hpp
    device/kernels/primitives/noc_local.hpp
    device/kernels/primitives/stick_cache.hpp
    device/kernels/primitives/tile_2d_layout.hpp
    device/kernels/primitives/tile_move_copy_32x16.h
    device/kernels/primitives/workspace_layout.hpp
    device/kernels/sfpi/horizontal_stencil_sfpi.h
    device/kernels/sfpi/lwt_sfpi_common.h
    device/kernels/sfpi/scale_sfpi.h
    device/kernels/sfpi/vertical_stencil_sfpi.h
    device/protocol/lwt_2d_config.hpp
    device/protocol/lwt_config.hpp
    planner/static_scheme.hpp
    planner/step.hpp)
set(TT_WAVELET_RESOURCE_ROOT "${PROJECT_BINARY_DIR}/python/ttwt/resources")
set(TT_WAVELET_RESOURCE_INPUTS)
set(TT_WAVELET_RESOURCE_OUTPUTS)
set(TT_WAVELET_RESOURCE_NAMES)
foreach(_file IN LISTS TT_WAVELET_RUNTIME_FILES)
  list(APPEND TT_WAVELET_RESOURCE_INPUTS "${CMAKE_CURRENT_SOURCE_DIR}/${_file}")
  list(APPEND TT_WAVELET_RESOURCE_NAMES "ttwt/${_file}")
endforeach()
foreach(_file IN LISTS TT_WAVELET_GENERATED_SCHEME_HEADERS)
  get_filename_component(_name "${_file}" NAME)
  list(APPEND TT_WAVELET_RESOURCE_INPUTS "${_file}")
  list(APPEND TT_WAVELET_RESOURCE_NAMES "ttwt/generated/wavelet_schemes/${_name}")
endforeach()
set(_manifest
    "#pragma once\n#include <array>\n#include <string_view>\nnamespace ttwt::detail::runtime_resources {\ninline constexpr auto required_files = std::to_array<std::string_view>({\n"
)
list(LENGTH TT_WAVELET_RESOURCE_INPUTS _count)
math(EXPR _last "${_count} - 1")
foreach(_index RANGE 0 ${_last})
  list(GET TT_WAVELET_RESOURCE_INPUTS ${_index} _input)
  list(GET TT_WAVELET_RESOURCE_NAMES ${_index} _relative)
  set(_output "${TT_WAVELET_RESOURCE_ROOT}/${_relative}")
  get_filename_component(_directory "${_output}" DIRECTORY)
  add_custom_command(
    OUTPUT "${_output}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_directory}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_input}" "${_output}"
    DEPENDS "${_input}"
    VERBATIM)
  list(APPEND TT_WAVELET_RESOURCE_OUTPUTS "${_output}")
  if(TT_WAVELET_BUILD_PYTHON)
    get_filename_component(_install_directory "${_relative}" DIRECTORY)
    install(
      FILES "${_output}"
      DESTINATION "ttwt/resources/${_install_directory}"
      COMPONENT ttwt-python)
  endif()
  string(APPEND _manifest "    \"${_relative}\",\n")
endforeach()
string(APPEND _manifest "});\n}\n")
file(CONFIGURE OUTPUT "${PROJECT_BINARY_DIR}/generated/ttwt/runtime_resource_manifest.hpp" CONTENT
     "${_manifest}" @ONLY)
add_custom_target(tt_wavelet_resources DEPENDS ${TT_WAVELET_RESOURCE_OUTPUTS})
add_dependencies(tt_wavelet_resources tt_wavelet_schemes)
