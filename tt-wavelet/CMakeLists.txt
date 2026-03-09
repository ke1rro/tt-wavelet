if (BUILD_TT_WAVELET)
    add_executable(tt_wavelet_test
            main.cpp)
    target_link_libraries(tt_wavelet_test PRIVATE Metalium::Metal)

    add_executable(tt_wavelet_elt_single_core_test
            main_elt.cpp)
    target_link_libraries(tt_wavelet_elt_single_core_test PRIVATE Metalium::Metal)

    add_executable(tt_wavelet_elt_multicore_test
            main_elt_multicore.cpp)
    target_link_libraries(tt_wavelet_elt_multicore_test PRIVATE Metalium::Metal)
endif()
