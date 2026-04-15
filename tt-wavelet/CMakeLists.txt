if(BUILD_TT_WAVELET)
  add_compile_definitions(TT_WAVELET_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  set(PROJECT_NAME lwt)
  add_executable(
    ${PROJECT_NAME} main.cpp tt_wavelet/src/lifting/plan.cpp
        tt_wavelet/src/pad_split/device.cpp tt_wavelet/src/lifting/device.cpp)

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
  file(GLOB TT_WAVELET_JSON_INCLUDE_CANDIDATES
       "${CMAKE_SOURCE_DIR}/tt-metal/.cpmcache/nlohmann_json/*/include")
  foreach(LLK_INCLUDE_DIR IN LISTS TT_WAVELET_LLK_INCLUDE_CANDIDATES)
    if(EXISTS "${LLK_INCLUDE_DIR}")
      list(APPEND TT_WAVELET_LLK_INCLUDE_DIRS "${LLK_INCLUDE_DIR}")
    endif()
  endforeach()
  foreach(JSON_INCLUDE_DIR IN LISTS TT_WAVELET_JSON_INCLUDE_CANDIDATES)
    if(EXISTS "${JSON_INCLUDE_DIR}")
      list(APPEND TT_WAVELET_INCLUDE_DIRS "${JSON_INCLUDE_DIR}")
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

  if(ENABLE_TRACY AND TARGET Tracy::TracyClient)
      target_link_libraries(${PROJECT_NAME} PRIVATE Tracy::TracyClient)
      message(STATUS "${PROJECT_NAME}: Tracy profiling enabled")
  else()
      message(STATUS "Tracy not enabled or Tracy::TracyClient target not found. Skipping.")
  endif()
endif()
