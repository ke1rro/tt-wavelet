set(TT_WAVELET_HEADERS wavelet.hpp wavelet_types.hpp runtime_resources.hpp)

set(TT_WAVELET_SRCS
    common/wavelet_host.cpp
    device/ilwt_1d_device_operation.cpp
    device/ilwt_1d_program_factory.cpp
    device/ilwt_2d_device_operation.cpp
    device/ilwt_2d_program_factory.cpp
    device/lwt_1d_device_operation.cpp
    device/lwt_1d_program_factory.cpp
    device/lwt_2d_device_operation.cpp
    device/lwt_2d_program_factory.cpp
    device/wavelet_1d_operation_impl.cpp
    device/wavelet_2d_operation_impl.cpp
    device/wavelet_program_utils.cpp
    device/wavelet_tensor_validation.cpp
    wavelet.cpp
    runtime_resources.cpp)
