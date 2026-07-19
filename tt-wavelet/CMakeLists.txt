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
endif()
