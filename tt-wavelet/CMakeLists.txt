if (BUILD_TT_WAVELET)
    # Main loopback test executable
    add_executable(tt_wavelet_test main.cpp)
    target_link_libraries(tt_wavelet_test PRIVATE Metalium::Metal)
    
    # Add Tracy support if enabled
    if(ENABLE_TRACY AND TARGET Tracy::TracyClient)
        target_link_libraries(tt_wavelet_test PRIVATE Tracy::TracyClient)
        message(STATUS "tt_wavelet_test: Tracy profiling enabled")
    endif()

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel/loopback.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/kernel/loopback.cpp
        COPYONLY
    )
endif()
