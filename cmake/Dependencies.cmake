set(TT_WAVELET_TT_METAL_MODE SUBMODULE CACHE STRING "TT-Metal dependency: SUBMODULE or SYSTEM")
set_property(CACHE TT_WAVELET_TT_METAL_MODE PROPERTY STRINGS SUBMODULE SYSTEM)
if(TT_WAVELET_TT_METAL_MODE STREQUAL "SYSTEM")
    find_package(TT-NN CONFIG REQUIRED)
    find_package(TT-Metalium CONFIG REQUIRED)
    # The pinned branch exports Wavelet from TTNN itself. Reject that SDK rather
    # than relying on ELF interposition to choose the extracted implementation.
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_LIBRARIES TTNN::TTNN TT::Metalium)
    unset(TT_WAVELET_SDK_EMBEDS_WAVELET CACHE)
    check_cxx_source_compiles("
        #include <cstdint>
        #include <string_view>
        namespace ttnn { std::uint32_t dwt_coeff_len(std::uint32_t, std::string_view); }
        int main() { return ttnn::dwt_coeff_len(64, \"db1\"); }
        " TT_WAVELET_SDK_EMBEDS_WAVELET)
    unset(CMAKE_REQUIRED_LIBRARIES)
    if(TT_WAVELET_SDK_EMBEDS_WAVELET)
        message(FATAL_ERROR
            "This TT-NN SDK already embeds Wavelet. Use SUBMODULE mode, or a matching SDK built with upstream Wavelet excluded.")
    endif()
elseif(TT_WAVELET_TT_METAL_MODE STREQUAL "SUBMODULE")
    include(${CMAKE_CURRENT_LIST_DIR}/TTMetal.cmake)
else()
    message(FATAL_ERROR "TT_WAVELET_TT_METAL_MODE must be SUBMODULE or SYSTEM")
endif()
