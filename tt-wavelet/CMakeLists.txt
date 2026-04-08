if(BUILD_TT_WAVELET)
  add_compile_definitions(TT_WAVELET_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_executable(
    tt_wavelet_test
    main.cpp
    tt_wavelet/src/pad_split_device.cpp
  )

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
      ${CMAKE_SOURCE_DIR}/tt-metal
      ${CMAKE_SOURCE_DIR}/tt-metal/tt_metal/common
      ${CMAKE_SOURCE_DIR}/tt-metal/tt_metal/api/tt-metalium
      ${CMAKE_CURRENT_SOURCE_DIR})
  foreach(LLK_INCLUDE_DIR IN LISTS TT_WAVELET_LLK_INCLUDE_CANDIDATES)
    if(EXISTS "${LLK_INCLUDE_DIR}")
      list(APPEND TT_WAVELET_LLK_INCLUDE_DIRS "${LLK_INCLUDE_DIR}")
    endif()
  endforeach()

  if(DEFINED METALIUM_INCLUDE_DIRS)
    target_include_directories(tt_wavelet_test PRIVATE ${METALIUM_INCLUDE_DIRS})
  endif()
  target_include_directories(tt_wavelet_test
                             PRIVATE ${TT_WAVELET_LLK_INCLUDE_DIRS})
  target_include_directories(tt_wavelet_test PRIVATE ${TT_WAVELET_INCLUDE_DIRS})
  target_link_libraries(tt_wavelet_test PRIVATE Metalium::Metal)

  add_executable(lwt_main
    lwt_main.cpp
    tt_wavelet/src/pad_split_device.cpp
    tt_wavelet/src/lwt_device.cpp
  )
  if(DEFINED METALIUM_INCLUDE_DIRS)
    target_include_directories(lwt_main PRIVATE ${METALIUM_INCLUDE_DIRS})
  endif()
  target_include_directories(lwt_main
    PRIVATE ${TT_WAVELET_LLK_INCLUDE_DIRS}
            ${TT_WAVELET_INCLUDE_DIRS})
  target_link_libraries(lwt_main PRIVATE Metalium::Metal)
endif()
