add_executable(tt_wavelet_test main.cpp)

if(TARGET Metalium::Metal)
    target_link_libraries(tt_wavelet_test PRIVATE Metalium::Metal)
else()
    target_link_libraries(tt_wavelet_test PRIVATE tt_metal)
endif()

target_include_directories(tt_wavelet_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_metal/api
    ${CMAKE_SOURCE_DIR}/third-party/tt-metal
    ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_metal/include
    ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_metal/hostdevcommon/api
)

message(STATUS "Configured tt-wavelet test executable with manual include paths")
