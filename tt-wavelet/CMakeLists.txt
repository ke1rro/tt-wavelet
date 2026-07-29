if(BUILD_TT_WAVELET)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_compile_definitions(TT_WAVELET_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

  set(TT_WAVELET_LWT_TARGET lwt)

  add_custom_target(
    tt_wavelet_generate_static_schemes
    COMMAND
      ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/generate_static_schemes.py
      --json-dir ${CMAKE_SOURCE_DIR}/wavelets
      --scheme-dir ${CMAKE_CURRENT_SOURCE_DIR}/tt_wavelet/include/schemes/generated
      --registry ${CMAKE_CURRENT_SOURCE_DIR}/tt_wavelet/include/schemes/generated/registry.hpp
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating static TT-wavelet scheme headers")

  set(TT_WAVELET_INCLUDE_DIRS
      ${CMAKE_SOURCE_DIR}/tt-metal
      ${CMAKE_SOURCE_DIR}/tt-metal/tt_metal/common
      ${CMAKE_SOURCE_DIR}/tt-metal/tt_metal/api/tt-metalium
      ${CMAKE_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR})
  function(tt_wavelet_configure_metal_target TARGET_NAME)
    if(DEFINED METALIUM_INCLUDE_DIRS)
      target_include_directories(${TARGET_NAME} PRIVATE ${METALIUM_INCLUDE_DIRS})
    endif()
    target_include_directories(${TARGET_NAME}
                               PRIVATE ${TT_WAVELET_INCLUDE_DIRS})
    target_link_libraries(${TARGET_NAME} PRIVATE Metalium::Metal)
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                     "${CMAKE_BINARY_DIR}")
  endfunction()

  add_executable(
    ${TT_WAVELET_LWT_TARGET} main.cpp tt_wavelet/src/lifting/device.cpp)
  add_dependencies(${TT_WAVELET_LWT_TARGET} tt_wavelet_generate_static_schemes)
  tt_wavelet_configure_metal_target(${TT_WAVELET_LWT_TARGET})

  add_executable(lwt_2d_planner_tests tests/lwt_2d_planner_tests.cpp)
  add_dependencies(lwt_2d_planner_tests tt_wavelet_generate_static_schemes)
  tt_wavelet_configure_metal_target(lwt_2d_planner_tests)

  add_executable(ilwt_2d_planner_tests tests/ilwt_2d_planner_tests.cpp)
  add_dependencies(ilwt_2d_planner_tests tt_wavelet_generate_static_schemes)
  tt_wavelet_configure_metal_target(ilwt_2d_planner_tests)

  add_executable(signal_extension_tests tests/signal_extension_tests.cpp)
  tt_wavelet_configure_metal_target(signal_extension_tests)

  add_executable(lwt_2d_reference reference_2d.cpp)
  add_dependencies(lwt_2d_reference tt_wavelet_generate_static_schemes)
  tt_wavelet_configure_metal_target(lwt_2d_reference)

  add_executable(lwt_2d main_2d.cpp tt_wavelet/src/lifting/device_2d.cpp)
  add_dependencies(lwt_2d tt_wavelet_generate_static_schemes)
  tt_wavelet_configure_metal_target(lwt_2d)

  add_executable(ilwt_2d main_ilwt_2d.cpp tt_wavelet/src/lifting/device_2d.cpp)
  add_dependencies(ilwt_2d tt_wavelet_generate_static_schemes)
  tt_wavelet_configure_metal_target(ilwt_2d)

  add_executable(vertical_stencil_k17_test
                 tests/vertical_stencil_k17_test.cpp)
  tt_wavelet_configure_metal_target(vertical_stencil_k17_test)

  add_executable(horizontal_dense_stencil_test
                 tests/horizontal_dense_stencil_test.cpp)
  tt_wavelet_configure_metal_target(horizontal_dense_stencil_test)
endif()
