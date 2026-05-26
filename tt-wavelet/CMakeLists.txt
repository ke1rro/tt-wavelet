if(BUILD_TT_WAVELET)
  add_compile_definitions(TT_WAVELET_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  file(GLOB TT_WAVELET_SCHEME_JSON CONFIGURE_DEPENDS
       "${CMAKE_SOURCE_DIR}/ttnn-wavelet/lifting_schemes/*.json")
  set(TT_WAVELET_SCHEME_REGISTRY
      "${CMAKE_CURRENT_SOURCE_DIR}/tt_wavelet/include/schemes/registry.hpp")
  add_custom_command(
    OUTPUT "${TT_WAVELET_SCHEME_REGISTRY}"
    COMMAND
      ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/scripts/generate_schemes.py"
      --schemes_dir "${CMAKE_SOURCE_DIR}/ttnn-wavelet/lifting_schemes"
      --headers_dir "${CMAKE_CURRENT_SOURCE_DIR}/tt_wavelet/include/schemes"
      --base-header "scheme.hpp" --registry-header "registry.hpp"
    DEPENDS "${CMAKE_SOURCE_DIR}/scripts/generate_schemes.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tt_wavelet/include/schemes/scheme.hpp"
            ${TT_WAVELET_SCHEME_JSON}
    VERBATIM)
  add_custom_target(lwt_generated_schemes DEPENDS "${TT_WAVELET_SCHEME_REGISTRY}")
  set(PROJECT_NAME lwt)
  add_executable(
    ${PROJECT_NAME} main.cpp tt_wavelet/src/pad_split/device.cpp
        tt_wavelet/src/lifting/device.cpp "${TT_WAVELET_SCHEME_REGISTRY}")
  add_dependencies(${PROJECT_NAME} lwt_generated_schemes)
  set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                   "${CMAKE_BINARY_DIR}")
  add_executable(lwt_shift_recover_test tests/shift_recover_test.cpp)
  set_target_properties(lwt_shift_recover_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                          "${CMAKE_BINARY_DIR}")
  add_executable(
    lwt_scheme_registry_test tests/scheme_registry_test.cpp
                             tt_wavelet/src/pad_split/device.cpp
                             tt_wavelet/src/lifting/device.cpp
                             "${TT_WAVELET_SCHEME_REGISTRY}")
  add_dependencies(lwt_scheme_registry_test lwt_generated_schemes)
  set_target_properties(lwt_scheme_registry_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                          "${CMAKE_BINARY_DIR}")

  set(TT_WAVELET_LLK_INCLUDE_CANDIDATES
      ${CMAKE_SOURCE_DIR}/tt-metal/tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib
      ${CMAKE_SOURCE_DIR}/tt-metal/tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc
      ${CMAKE_SOURCE_DIR}/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api
      ${CMAKE_SOURCE_DIR}/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/common
      ${CMAKE_SOURCE_DIR}/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/llk_io
      ${CMAKE_SOURCE_DIR}/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu
      ${CMAKE_SOURCE_DIR}/tt-metal/runtime/sfpi/include
      ${CMAKE_SOURCE_DIR}/tt-metal/tt_metal/hw/inc
      ${CMAKE_SOURCE_DIR}/tt-metal/tt_metal/hw/inc/internal/tt-1xx
      ${CMAKE_SOURCE_DIR}/tt-metal/tt_metal/hw/inc/internal/tt-1xx/wormhole
      ${CMAKE_SOURCE_DIR}/tt-metal/tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines
  )
  set(TT_WAVELET_LLK_INCLUDE_DIRS)

  set(TT_WAVELET_INCLUDE_DIRS
      ${CMAKE_SOURCE_DIR}/tt-metal ${CMAKE_SOURCE_DIR}/tt-metal/tt_metal/common
      ${CMAKE_SOURCE_DIR}/tt-metal/tt_metal/api/tt-metalium
      ${CMAKE_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR})
  foreach(LLK_INCLUDE_DIR IN LISTS TT_WAVELET_LLK_INCLUDE_CANDIDATES)
    if(EXISTS "${LLK_INCLUDE_DIR}")
      list(APPEND TT_WAVELET_LLK_INCLUDE_DIRS "${LLK_INCLUDE_DIR}")
    endif()
  endforeach()
  if(DEFINED METALIUM_INCLUDE_DIRS)
    target_include_directories(${PROJECT_NAME}
                               PRIVATE ${METALIUM_INCLUDE_DIRS})
  endif()
  target_include_directories(${PROJECT_NAME}
                             PRIVATE ${TT_WAVELET_LLK_INCLUDE_DIRS})
  target_include_directories(${PROJECT_NAME}
                             PRIVATE ${TT_WAVELET_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} PRIVATE Metalium::Metal)
  target_include_directories(lwt_shift_recover_test
                             PRIVATE ${TT_WAVELET_LLK_INCLUDE_DIRS})
  target_include_directories(lwt_shift_recover_test
                             PRIVATE ${TT_WAVELET_INCLUDE_DIRS})
  target_link_libraries(lwt_shift_recover_test PRIVATE Metalium::Metal)
  target_include_directories(lwt_scheme_registry_test
                             PRIVATE ${TT_WAVELET_LLK_INCLUDE_DIRS})
  target_include_directories(lwt_scheme_registry_test
                             PRIVATE ${TT_WAVELET_INCLUDE_DIRS})
  target_link_libraries(lwt_scheme_registry_test PRIVATE Metalium::Metal)

  add_test(NAME lwt_scheme_registry COMMAND lwt_scheme_registry_test)
  add_test(
    NAME lwt_scheme_generator
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/scripts/test_generate_schemes.py"
            --root "${CMAKE_SOURCE_DIR}" --cxx "${CMAKE_CXX_COMPILER}")
endif()
