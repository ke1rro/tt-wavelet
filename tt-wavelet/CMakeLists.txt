if (BUILD_TT_WAVELET)
    add_executable(tt_wavelet_test main.cpp)
    target_link_libraries(tt_wavelet_test PRIVATE Metalium::Metal)
    
    if(ENABLE_TRACY AND TARGET Tracy::TracyClient)
        # Tracy demo executable (CPU-only)
        add_executable(tracy_demo tracy_demo.cpp)
        target_link_libraries(tracy_demo PRIVATE Tracy::TracyClient)
        
        # N300s demo with Tracy profiling
        add_executable(n300s_demo n300s_demo.cpp)
        target_link_libraries(n300s_demo PRIVATE 
            Metalium::Metal
            Tracy::TracyClient
        )
    else()
        message(STATUS "Tracy not enabled or Tracy::TracyClient target not found. Skipping tracy_demo and n300s_demo.")
    endif()
endif()
