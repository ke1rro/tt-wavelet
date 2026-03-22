if (BUILD_TT_WAVELET)
    find_package(nlohmann_json REQUIRED)

    add_executable(tt_wavelet_test main.cpp)

    target_include_directories(tt_wavelet_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    target_link_libraries(tt_wavelet_test 
        PRIVATE 
        Metalium::Metal
        nlohmann_json::nlohmann_json
    )

    target_compile_features(tt_wavelet_test PRIVATE cxx_std_17)
endif()
