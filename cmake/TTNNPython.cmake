# Build-tree adaptation only: never patch the pinned dependency checkout.
set(_ttnn_init "${TT_WAVELET_METAL_SOURCE_DIR}/ttnn/cpp/ttnn-nanobind/__init__.cpp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_ttnn_init}")
file(READ "${_ttnn_init}" _ttnn_init_text)
foreach(_fragment IN ITEMS
    "#include \"ttnn/operations/wavelet/wavelet_nanobind.hpp\"\n"
    "    wavelet::bind_wavelet_operations(mod);\n")
    string(FIND "${_ttnn_init_text}" "${_fragment}" _fragment_position)
    if(_fragment_position LESS 0)
        message(FATAL_ERROR "Pinned TTNN Wavelet binding integration changed")
    endif()
    string(REPLACE "${_fragment}" "" _ttnn_init_text "${_ttnn_init_text}")
endforeach()
set(_ttnn_init_standalone "${CMAKE_BINARY_DIR}/generated/ttnn-python/__init__.cpp")
file(CONFIGURE OUTPUT "${_ttnn_init_standalone}" CONTENT "${_ttnn_init_text}" @ONLY)
get_target_property(_ttnn_python_sources ttnn SOURCES)
list(FILTER _ttnn_python_sources EXCLUDE REGEX "(^|/)wavelet_nanobind\\.cpp$")
list(FILTER _ttnn_python_sources EXCLUDE REGEX "(^|/)ttnn-nanobind/__init__\\.cpp$")
set_property(TARGET ttnn PROPERTY SOURCES "${_ttnn_python_sources};${_ttnn_init_standalone}")
# Stage the real upstream Python package in the build tree, avoiding installs
# into the dependency source directory and avoiding an unrelated installed wheel.
file(COPY "${TT_WAVELET_METAL_SOURCE_DIR}/ttnn/ttnn" DESTINATION "${CMAKE_BINARY_DIR}/python"
    PATTERN "*.so" EXCLUDE PATTERN "__pycache__" EXCLUDE)
set_target_properties(ttnn PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/ttnn")
