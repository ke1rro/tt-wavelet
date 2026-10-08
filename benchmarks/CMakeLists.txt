# Board discovery uses this version-pinned runtime's cluster metadata, never
# hostnames.
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
add_executable(ttwt_benchmark native_benchmark.cpp)
target_link_libraries(
  ttwt_benchmark PRIVATE TTWavelet::Native nlohmann_json::nlohmann_json
                         yaml-cpp::yaml-cpp OpenSSL::Crypto)
target_compile_definitions(ttwt_benchmark PRIVATE TTWT_BENCHMARK_RELEASE=1)
