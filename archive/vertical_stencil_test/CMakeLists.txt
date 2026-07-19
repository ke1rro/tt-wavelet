# Historical target definition. This directory is intentionally not added by
# the production build. When included from the old tt-wavelet CMake scope it
# relied on tt_wavelet_configure_metal_target being defined by the parent.
set(TT_WAVELET_VERTICAL_STENCIL_TARGET vertical_stencil_test)
add_executable(${TT_WAVELET_VERTICAL_STENCIL_TARGET} tt-wavelet/stencil.cpp)
tt_wavelet_configure_metal_target(${TT_WAVELET_VERTICAL_STENCIL_TARGET})
