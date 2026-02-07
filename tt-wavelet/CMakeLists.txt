if (BUILD_TT_WAVELET)
    add_executable(tt_wavelet_test main.cpp)
    target_link_libraries(tt_wavelet_test PRIVATE Metalium::Metal)
endif()
