add_executable(tt_wavelet_test main.cpp)

target_link_libraries(tt_wavelet_test PRIVATE TT::Metalium)

message(STATUS "Configured tt-wavelet test executable")
