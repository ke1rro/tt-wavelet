find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
if(TARGET ttnn)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    if(NOT Python_VERSION VERSION_EQUAL Python3_VERSION)
        message(FATAL_ERROR "The TTNN and ttwt Python interpreter versions must match")
    endif()
endif()
# SUBMODULE provides the exact nanobind pinned by TTNN. SYSTEM users must supply
# the matching development configuration; do not fetch another version.
if(NOT COMMAND nanobind_add_module)
    find_package(nanobind 2.14.0 EXACT CONFIG REQUIRED)
endif()
nanobind_add_module(_ttwt NB_STATIC module.cpp)
target_link_libraries(_ttwt PRIVATE TTWavelet::Native)
add_dependencies(_ttwt tt_wavelet_resources)
set_target_properties(_ttwt PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/ttwt")
set_target_properties(_ttwt PROPERTIES
    INSTALL_RPATH "$ORIGIN/../ttnn/build/lib"
    INSTALL_RPATH_USE_LINK_PATH FALSE)
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/python/ttwt")
configure_file("${PROJECT_SOURCE_DIR}/python/ttwt/__init__.py"
    "${CMAKE_BINARY_DIR}/python/ttwt/__init__.py" COPYONLY)
if(TARGET ttnn)
    add_dependencies(_ttwt ttnn)
endif()
install(TARGETS _ttwt LIBRARY DESTINATION ttwt COMPONENT ttwt-python)
install(FILES "${PROJECT_SOURCE_DIR}/python/ttwt/__init__.py" DESTINATION ttwt COMPONENT ttwt-python)
