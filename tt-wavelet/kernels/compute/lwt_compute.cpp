#ifndef WAVELET_CHUNK
#define WAVELET_CHUNK 0
#endif

#ifndef WAVELET_CHUNK_SIZE
#define WAVELET_CHUNK_SIZE 4
#endif

#ifndef WAVELET_SCHEME_COUNT
#define WAVELET_SCHEME_COUNT 106
#endif

#define TTWV_WAVELET_REGISTRY_KERNEL_ONLY
#include "../../tt_wavelet/include/lifting/wavelets/wavelet_registry.hpp"

#include "lwt_compute_compiletime_impl.hpp"

void kernel_main() {
    ttwv::kernels::lwt::kernel_main_chunk<WAVELET_CHUNK, WAVELET_CHUNK_SIZE, WAVELET_SCHEME_COUNT>();
}
