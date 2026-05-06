Yes, the plan is valid. Please implement it with the corrected static scheme representation: do not use MaxCoeffs as the coefficient container for every step.

Each lifting step must preserve its exact coefficient arity, so use something like:
cpp
template <StepType Type, int32_t Shift, uint32_t... CoeffBits>
struct StaticStep {
    static constexpr StepType type = Type;
    static constexpr int32_t shift = Shift;
    static constexpr uint32_t k = sizeof...(CoeffBits);
    static constexpr std::array<uint32_t, k> coeff_bits = {CoeffBits...};
};

Then store scheme steps as a heterogeneous std::tuple<StaticStep<...>, ...> or equivalent generated step<I> traits.

This way Step::k is exactly len(coefficients) and the fused compute path can call:
cpp
run_step<Step::k>(Step::coeff_bits, ...);

So please proceed with the fused static-scheme plan, but with exact per-step coefficient arity instead of std::array<uint32_t, MaxCoeffs>.

Show more
6:41 PM

Plan


Open

Static Schemes + Fused LWT Launch
Summary
Replace runtime JSON parsing in the C++ TT-wavelet path with generated compile-time C++ schemes.
Preserve exact per-step coefficient arity with parameter-pack StaticStep types, not fixed MaxCoeffs arrays.
Keep runtime wavelet names by dispatching "bior2.2" to precompiled static specializations.
Fuse predict/update/scale steps into one LWT TT-Metal program launch after the existing pad/split preprocess.
Static Scheme Model
Define exact-arity steps:
template <StepType Type, int32_t Shift, uint32_t... CoeffBits>
struct StaticStep {
    static constexpr StepType type = Type;
    static constexpr int32_t shift = Shift;
    static constexpr uint32_t k = sizeof...(CoeffBits);
    static constexpr std::array<uint32_t, k> coeff_bits = {CoeffBits...};
};
Generate each scheme with step<I> traits rather than std::array<StaticStep<MaxCoeffs>, N>:
struct bior2_2 {
    static constexpr std::string_view name = "bior2.2";
    static constexpr uint32_t tap_size = 6;
    static constexpr int32_t delay_even = 1;
    static constexpr int32_t delay_odd = 2;
    static constexpr uint32_t num_steps = 5;

    template <size_t I>
    struct step;
};

template <>
struct bior2_2::step<0> {
    using type = StaticStep<StepType::kUpdate, -1, f32(-0.5f), f32(-0.5f)>;
};
Step::k must always equal the JSON coefficient count for that step, including 0 for swap and 1 for scale.
Implementation Changes
Replace JSON runtime code:
remove normal-path use of RuntimeLiftingScheme, RuntimeLiftingStep, load_runtime_lifting_scheme, pack_device_step_desc, and packed_steps.
keep JSON only as generator/test input.
Add a Python generator that reads ttnn-wavelet/lifting_schemes/*.json and emits:
generated static scheme headers
a generated name registry
generated per-scheme compute wrapper files if needed by TT-Metal kernel compilation.
Generate coefficient bits as fp32 bit patterns so runtime math matches the old float parse and std::bit_cast<uint32_t> behavior.
Template host planning as make_forward_lifting_plan<Scheme>(...); keep lengths, offsets, output groups, and addresses runtime-derived.
Build one fused LWT program with one reader, one compute kernel, and one writer:
predict/update calls run_step<Step::k>(Step::coeff_bits, ...).
scale calls a fused scale branch with Step::coeff_bits[0].
swap is handled by host route resolution and emits no device work.
Remove the separate terminal tile-sidecar scale path from normal execution.
Use a compact per-step route config buffer for geometry and addresses, avoiding runtime-args capacity limits.
Test Plan
Run generator twice and verify the second run is clean.
Build with ./build.sh Release.
Run:
python3 compare.py --wavelet bior2.2
python3 compare.py --wavelet coif2
python3 compare.py --wavelet haar
one long-step scheme such as coif17
Verify C++ runtime no longer opens or parses lifting-scheme JSON.
Verify generated step arities match JSON coefficient lengths for all schemes.
Assumptions
Preprocess may remain a separate launch; “fused” means all lifting steps after preprocess run in one LWT launch.
Intermediate buffers may remain conservative row-major DRAM ping/pong for this implementation.
L1-only intermediate storage is a later optimization.
Implement plan
6:42 PM

I’m switching from planning to implementation. I’ll first re-check the current build surface and kernel API usage, then I’ll make the generator/static scheme changes before touching the fused device path.


Explored 5 files, 3 searches, 1 list, ran 1 command
I have enough of the existing path now. The main structural change I’m about to make is: all four working streams become DRAM row-major buffers, then the fused reader/writer use a compact route-config buffer so long schemes do not overflow runtime args.


Created 3 files, edited 5 files, explored 3 files, ran 1 command
The host-side static plan and fused program factory are in place. I’m adding the actual fused kernels now: a generic dataflow reader/writer driven by route-config pages, plus a scheme-specialized compute template that uses Step::k and Step::coeff_bits.


Created 4 files
Next I’m generating the static scheme headers and per-scheme compute wrappers from the JSON source files. This is the only place JSON remains in the C++ pipeline: build-time generation, not runtime loading.


Edited 4 files, explored 4 files, 2 searches, ran 3 commands
The code no longer references the old runtime JSON loader or packed step descriptors. I’m starting with a targeted rebuild of lwt; if configuration is stale, CMake should regenerate and run the scheme generator target first.

Ran cmake --build build --target lwt -j2
The rebuild is past CMake regeneration and is compiling the changed TT-wavelet sources. Current output is only existing Metalium deprecation warnings, not failures.

Edited 1 file
The first build failure was a constness mismatch when writing the route-config mesh buffer; I fixed that by taking a local shared pointer before enqueueing. I’m rebuilding the target now.

Ran cmake --build build --target lwt -j2
The lwt target now builds successfully. I’m moving on to runtime comparisons; these will also exercise TT-Metal JIT compilation of the generated fused compute wrapper for each scheme.

Ran python3 compare.py --wavelet bior2.2
$ python3 compare.py --wavelet bior2.2

[output truncated]
p:1,
                 from /root/tt-wavelet/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/common/chlkc_list.h:37,
                 from /root/tt-wavelet/tt-metal/tt_metal/hw/firmware/src/tt-1xx/trisck.cc:13:
/root/tt-wavelet/tt-wavelet/kernels/compute/generated/../../../tt_wavelet/include/device_protocol/step_desc.hpp:5:10: fatal error: tt_wavelet/include/device_protocol/lwt_config.hpp: No such file or directory
    5 | #include "tt_wavelet/include/device_protocol/lwt_config.hpp"
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
 (assert.hpp:104)
2026-05-06 15:57:02.001 | error    |    BuildKernels | trisc1 compile failure -- cmd: cd /root/.cache/tt-metal-cache/f87c34a93e/11358605694596585868/kernels/lwt_fused_bior2_2_compute/14643373009086007484/trisc1/ && /opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++ -O3 -std=c++17 -flto=auto -ffast-math -fno-exceptions -g -MMD -fno-use-cxa-atexit -Wall -Werror -Wno-unknown-pragmas -Wno-deprecated-declarations -Wno-error=multistatement-macros -Wno-error=parentheses -Wno-error=unused-but-set-variable -Wno-unused-variable -Wno-unused-function -mcpu=tt-wh-tensix -I. -I.. -I/root/tt-wavelet/tt-metal/ -I/root/tt-wavelet/tt-metal/ttnn -I/root/tt-wavelet/tt-metal/ttnn/cpp -I/root/tt-wavelet/tt-metal/tt_metal -I/root/tt-wavelet/tt-metal/tt_metal/include -I/root/tt-wavelet/tt-metal/tt_metal/hw/inc -I/root/tt-wavelet/tt-metal/tt_metal/hostdevcommon/api -I/root/tt-wavelet/tt-metal/tt_metal/hw/inc/debug -I/root/tt-wavelet/tt-metal/tt_metal/api/ -I/opt/tenstorrent/sfpi/include -I/root/tt-wavelet/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/common -I/root/tt-wavelet/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/llk_io -I/root/tt-wavelet/tt-metal/tt_metal/hw/inc/internal/tt-1xx -I/root/tt-wavelet/tt-metal/tt_metal/hw/inc/internal/tt-1xx/wormhole -I/root/tt-wavelet/tt-metal/tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines -I/root/tt-wavelet/tt-metal/tt_metal/hw/inc/internal/tt-1xx/wormhole/noc -I/root/tt-wavelet/tt-metal/tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc -I/root/tt-wavelet/tt-metal/tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib -I/root/tt-wavelet/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api -I/root/tt-wavelet/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu -I/root/tt-wavelet/tt-metal/tt_metal/hw/firmware/src/tt-1xx -I/root/tt-wavelet/tt-wavelet/kernels/compute/generated -c -o trisck.o /root/tt-wavelet/tt-metal/tt_metal/hw/firmware/src/tt-1xx/trisck.cc -DIS_NOT_POW2_NUM_DRAM_BANKS=1 -DIS_NOT_POW2_NUM_L1_BANKS=1 -DNUM_DRAM_BANKS=12 -DNUM_L1_BANKS=56 -DPCIE_NOC_X=0 -DPCIE_NOC_Y=3 -DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 -DROUTING_FW_ENABLED -DPROCESSOR_INDEX=3 -DUCK_CHLKC_MATH -DNAMESPACE=chlkc_math -DCOMPILE_FOR_TRISC=1 -DARCH_WORMHOLE -DDISPATCH_MESSAGE_ADDR=4290183304 -DKERNEL_BUILD -DKERNEL_COMPILE_TIME_ARGS=0,1,2,16  (build.cpp:51)
2026-05-06 15:57:02.001 | critical |          Always | TT_THROW: trisc1 build failed. Log: In file included from /root/tt-wavelet/tt-wavelet/kernels/compute/generated/../lwt_compute_utils.hpp:3,
                 from /root/tt-wavelet/tt-wavelet/kernels/compute/generated/../lwt_fused_compute_template.hpp:11,
                 from /root/tt-wavelet/tt-wavelet/kernels/compute/generated/lwt_fused_bior2_2_compute.cpp:1,
                 from /root/tt-wavelet/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/common/chlkc_list.h:21,
                 from /root/tt-wavelet/tt-metal/tt_metal/hw/firmware/src/tt-1xx/trisck.cc:13:
/root/tt-wavelet/tt-wavelet/kernels/compute/generated/../../../tt_wavelet/include/device_protocol/step_desc.hpp:5:10: fatal error: tt_wavelet/include/device_protocol/lwt_config.hpp: No such file or directory
    5 | #include "tt_wavelet/include/device_protocol/lwt_config.hpp"
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
 (assert.hpp:104)
2026-05-06 15:57:02.057 | error    |    BuildKernels | trisc2 compile failure -- cmd: cd /root/.cache/tt-metal-cache/f87c34a93e/11358605694596585868/kernels/lwt_fused_bior2_2_compute/14643373009086007484/trisc2/ && /opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++ -O3 -std=c++17 -flto=auto -ffast-math -fno-exceptions -g -MMD -fno-use-cxa-atexit -Wall -Werror -Wno-unknown-pragmas -Wno-deprecated-declarations -Wno-error=multistatement-macros -Wno-error=parentheses -Wno-error=unused-but-set-variable -Wno-unused-variable -Wno-unused-function -mcpu=tt-wh-tensix -I. -I.. -I/root/tt-wavelet/tt-metal/ -I/root/tt-wavelet/tt-metal/ttnn -I/root/tt-wavelet/tt-metal/ttnn/cpp -I/root/tt-wavelet/tt-metal/tt_metal -I/root/tt-wavelet/tt-metal/tt_metal/include -I/root/tt-wavelet/tt-metal/tt_metal/hw/inc -I/root/tt-wavelet/tt-metal/tt_metal/hostdevcommon/api -I/root/tt-wavelet/tt-metal/tt_metal/hw/inc/debug -I/root/tt-wavelet/tt-metal/tt_metal/api/ -I/opt/tenstorrent/sfpi/include -I/root/tt-wavelet/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/common -I/root/tt-wavelet/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/llk_io -I/root/tt-wavelet/tt-metal/tt_metal/hw/inc/internal/tt-1xx -I/root/tt-wavelet/tt-metal/tt_metal/hw/inc/internal/tt-1xx/wormhole -I/root/tt-wavelet/tt-metal/tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines -I/root/tt-wavelet/tt-metal/tt_metal/hw/inc/internal/tt-1xx/wormhole/noc -I/root/tt-wavelet/tt-metal/tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc -I/root/tt-wavelet/tt-metal/tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib -I/root/tt-wavelet/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api -I/root/tt-wavelet/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu -I/root/tt-wavelet/tt-metal/tt_metal/hw/firmware/src/tt-1xx -I/root/tt-wavelet/tt-wavelet/kernels/compute/generated -c -o trisck.o /root/tt-wavelet/tt-metal/tt_metal/hw/firmware/src/tt-1xx/trisck.cc -DIS_NOT_POW2_NUM_DRAM_BANKS=1 -DIS_NOT_POW2_NUM_L1_BANKS=1 -DNUM_DRAM_BANKS=12 -DNUM_L1_BANKS=56 -DPCIE_NOC_X=0 -DPCIE_NOC_Y=3 -DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 -DROUTING_FW_ENABLED -DPROCESSOR_INDEX=4 -DUCK_CHLKC_PACK -DNAMESPACE=chlkc_pack -DCOMPILE_FOR_TRISC=2 -DARCH_WORMHOLE -DDISPATCH_MESSAGE_ADDR=4290183304 -DKERNEL_BUILD -DKERNEL_COMPILE_TIME_ARGS=0,1,2,16  (build.cpp:51)
2026-05-06 15:57:02.057 | critical |          Always | TT_THROW: trisc2 build failed. Log: In file included from /root/tt-wavelet/tt-wavelet/kernels/compute/generated/../lwt_compute_utils.hpp:3,
                 from /root/tt-wavelet/tt-wavelet/kernels/compute/generated/../lwt_fused_compute_template.hpp:11,
                 from /root/tt-wavelet/tt-wavelet/kernels/compute/generated/lwt_fused_bior2_2_compute.cpp:1,
                 from /root/tt-wavelet/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/common/chlkc_list.h:29,
                 from /root/tt-wavelet/tt-metal/tt_metal/hw/firmware/src/tt-1xx/trisck.cc:13:
/root/tt-wavelet/tt-wavelet/kernels/compute/generated/../../../tt_wavelet/include/device_protocol/step_desc.hpp:5:10: fatal error: tt_wavelet/include/device_protocol/lwt_config.hpp: No such file or directory
    5 | #include "tt_wavelet/include/device_protocol/lwt_config.hpp"
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
 (assert.hpp:104)
2026-05-06 15:57:02.058 | critical |          Always | TT_THROW: Failed to generate binaries for lwt_fused_bior2_2_compute TT_THROW @ /root/tt-wavelet/tt-metal/tt_metal/jit_build/build.cpp:55: tt::exception
info:
trisc0 build failed. Log: In file included from /root/tt-wavelet/tt-wavelet/kernels/compute/generated/../lwt_compute_utils.hpp:3,
                 from /root/tt-wavelet/tt-wavelet/kernels/compute/generated/../lwt_fused_compute_template.hpp:11,
                 from /root/tt-wavelet/tt-wavelet/kernels/compute/generated/lwt_fused_bior2_2_compute.cpp:1,
                 from /root/tt-wavelet/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/common/chlkc_list.h:37,
                 from /root/tt-wavelet/tt-metal/tt_metal/hw/firmware/src/tt-1xx/trisck.cc:13:
/root/tt-wavelet/tt-wavelet/kernels/compute/generated/../../../tt_wavelet/include/device_protocol/step_desc.hpp:5:10: fatal error: tt_wavelet/include/device_protocol/lwt_config.hpp: No such file or directory
    5 | #include "tt_wavelet/include/device_protocol/lwt_config.hpp"
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.

backtrace:
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x4067a9) [0x7f615c6297a9]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5bb54f) [0x7f615c7de54f]
 --- tt::tt_metal::JitBuildState::compile_one(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tt::tt_metal::JitBuildSettings const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) const
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c1824) [0x7f615c7e4824]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c17c9) [0x7f615c7e47c9]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x363a47) [0x7f615c586a47]
 --- /lib/x86_64-linux-gnu/libc.so.6(+0x99ee8) [0x7f615b9b2ee8]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x3639c2) [0x7f615c5869c2]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c1601) [0x7f615c7e4601]
 --- auto tt::tt_metal::detail::async<std::function<void ()> const&>(std::function<void ()> const&)
 --- tt::tt_metal::launch_build_step(std::function<void ()> const&, std::vector<std::shared_future<void>, std::allocator<std::shared_future<void> > >&)
 --- tt::tt_metal::JitBuildState::compile(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tt::tt_metal::JitBuildSettings const*) const
 --- tt::tt_metal::JitBuildState::build(tt::tt_metal::JitBuildSettings const*) const
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c1824) [0x7f615c7e4824]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c17c9) [0x7f615c7e47c9]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x363a47) [0x7f615c586a47]
 --- /lib/x86_64-linux-gnu/libc.so.6(+0x99ee8) [0x7f615b9b2ee8]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x3639c2) [0x7f615c5869c2]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c1601) [0x7f615c7e4601]
 --- auto tt::tt_metal::detail::async<std::function<void ()> const&>(std::function<void ()> const&)
 --- tt::tt_metal::launch_build_step(std::function<void ()> const&, std::vector<std::shared_future<void>, std::allocator<std::shared_future<void> > >&)
 --- tt::tt_metal::jit_build_subset(std::span<tt::tt_metal::JitBuildState const, 18446744073709551615ul>, tt::tt_metal::JitBuildSettings const*)
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x441d62) [0x7f615c664d62]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c1824) [0x7f615c7e4824]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c17c9) [0x7f615c7e47c9]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x363a47) [0x7f615c586a47]
 --- /lib/x86_64-linux-gnu/libc.so.6(+0x99ee8) [0x7f615b9b2ee8]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x3639c2) [0x7f615c5869c2]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c1601) [0x7f615c7e4601]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x367c65) [0x7f615c58ac65]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x366781) [0x7f615c589781]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x365c79) [0x7f615c588c79]
 --- /lib/x86_64-linux-gnu/libstdc++.so.6(+0xdc253) [0x7f615bd27253]
 --- /lib/x86_64-linux-gnu/libc.so.6(+0x94ac3) [0x7f615b9adac3]
 --- /lib/x86_64-linux-gnu/libc.so.6(+0x1268d0) [0x7f615ba3f8d0]
 (assert.hpp:104)
TT_THROW @ /root/tt-wavelet/tt-metal/tt_metal/impl/program/program.cpp:160: tt::exception
info:
Failed to generate binaries for lwt_fused_bior2_2_compute TT_THROW @ /root/tt-wavelet/tt-metal/tt_metal/jit_build/build.cpp:55: tt::exception
info:
trisc0 build failed. Log: In file included from /root/tt-wavelet/tt-wavelet/kernels/compute/generated/../lwt_compute_utils.hpp:3,
                 from /root/tt-wavelet/tt-wavelet/kernels/compute/generated/../lwt_fused_compute_template.hpp:11,
                 from /root/tt-wavelet/tt-wavelet/kernels/compute/generated/lwt_fused_bior2_2_compute.cpp:1,
                 from /root/tt-wavelet/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/common/chlkc_list.h:37,
                 from /root/tt-wavelet/tt-metal/tt_metal/hw/firmware/src/tt-1xx/trisck.cc:13:
/root/tt-wavelet/tt-wavelet/kernels/compute/generated/../../../tt_wavelet/include/device_protocol/step_desc.hpp:5:10: fatal error: tt_wavelet/include/device_protocol/lwt_config.hpp: No such file or directory
    5 | #include "tt_wavelet/include/device_protocol/lwt_config.hpp"
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.

backtrace:
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x4067a9) [0x7f615c6297a9]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5bb54f) [0x7f615c7de54f]
 --- tt::tt_metal::JitBuildState::compile_one(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tt::tt_metal::JitBuildSettings const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) const
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c1824) [0x7f615c7e4824]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c17c9) [0x7f615c7e47c9]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x363a47) [0x7f615c586a47]
 --- /lib/x86_64-linux-gnu/libc.so.6(+0x99ee8) [0x7f615b9b2ee8]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x3639c2) [0x7f615c5869c2]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c1601) [0x7f615c7e4601]
 --- auto tt::tt_metal::detail::async<std::function<void ()> const&>(std::function<void ()> const&)
 --- tt::tt_metal::launch_build_step(std::function<void ()> const&, std::vector<std::shared_future<void>, std::allocator<std::shared_future<void> > >&)
 --- tt::tt_metal::JitBuildState::compile(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tt::tt_metal::JitBuildSettings const*) const
 --- tt::tt_metal::JitBuildState::build(tt::tt_metal::JitBuildSettings const*) const
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c1824) [0x7f615c7e4824]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c17c9) [0x7f615c7e47c9]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x363a47) [0x7f615c586a47]
 --- /lib/x86_64-linux-gnu/libc.so.6(+0x99ee8) [0x7f615b9b2ee8]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x3639c2) [0x7f615c5869c2]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c1601) [0x7f615c7e4601]
 --- auto tt::tt_metal::detail::async<std::function<void ()> const&>(std::function<void ()> const&)
 --- tt::tt_metal::launch_build_step(std::function<void ()> const&, std::vector<std::shared_future<void>, std::allocator<std::shared_future<void> > >&)
 --- tt::tt_metal::jit_build_subset(std::span<tt::tt_metal::JitBuildState const, 18446744073709551615ul>, tt::tt_metal::JitBuildSettings const*)
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x441d62) [0x7f615c664d62]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c1824) [0x7f615c7e4824]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c17c9) [0x7f615c7e47c9]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x363a47) [0x7f615c586a47]
 --- /lib/x86_64-linux-gnu/libc.so.6(+0x99ee8) [0x7f615b9b2ee8]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x3639c2) [0x7f615c5869c2]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c1601) [0x7f615c7e4601]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x367c65) [0x7f615c58ac65]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x366781) [0x7f615c589781]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x365c79) [0x7f615c588c79]
 --- /lib/x86_64-linux-gnu/libstdc++.so.6(+0xdc253) [0x7f615bd27253]
 --- /lib/x86_64-linux-gnu/libc.so.6(+0x94ac3) [0x7f615b9adac3]
 --- /lib/x86_64-linux-gnu/libc.so.6(+0x1268d0) [0x7f615ba3f8d0]

backtrace:
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x44f859) [0x7f615c672859]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x442189) [0x7f615c665189]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c1824) [0x7f615c7e4824]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c17c9) [0x7f615c7e47c9]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x363a47) [0x7f615c586a47]
 --- /lib/x86_64-linux-gnu/libc.so.6(+0x99ee8) [0x7f615b9b2ee8]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x3639c2) [0x7f615c5869c2]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c1601) [0x7f615c7e4601]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x367c65) [0x7f615c58ac65]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x366781) [0x7f615c589781]
 --- /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x365c79) [0x7f615c588c79]
 --- /lib/x86_64-linux-gnu/libstdc++.so.6(+0xdc253) [0x7f615bd27253]
 --- /lib/x86_64-linux-gnu/libc.so.6(+0x94ac3) [0x7f615b9adac3]
 --- /lib/x86_64-linux-gnu/libc.so.6(+0x1268d0) [0x7f615ba3f8d0]

[25f78357:2492856] *** Process received signal ***
[25f78357:2492856] Signal: Segmentation fault (11)
[25f78357:2492856] Signal code: Address not mapped (1)
[25f78357:2492856] Failing at address: 0x7f674a0ff227
[25f78357:2492856] [ 0] /lib/x86_64-linux-gnu/libc.so.6(+0x42520)[0x7f615b95b520]
[25f78357:2492856] [ 1] /lib/x86_64-linux-gnu/libc.so.6(+0x1a6932)[0x7f615babf932]
[25f78357:2492856] [ 2] /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x35fe02)[0x7f615c582e02]
[25f78357:2492856] [ 3] /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(_ZN2tt9jit_build23write_dependency_hashesERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES8_+0x41)[0x7f615c7eeae1]
[25f78357:2492856] [ 4] /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(_ZNK2tt8tt_metal13JitBuildState11compile_oneERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKNS0_16JitBuildSettingsES9_S9_+0xf2a)[0x7f615c7ddf1a]
[25f78357:2492856] [ 5] /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c1824)[0x7f615c7e4824]
[25f78357:2492856] [ 6] /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c17c9)[0x7f615c7e47c9]
[25f78357:2492856] [ 7] /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x363a47)[0x7f615c586a47]
[25f78357:2492856] [ 8] /lib/x86_64-linux-gnu/libc.so.6(+0x99ee8)[0x7f615b9b2ee8]
[25f78357:2492856] [ 9] /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x3639c2)[0x7f615c5869c2]
[25f78357:2492856] [10] /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x5c1601)[0x7f615c7e4601]
[25f78357:2492856] [11] /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x367c65)[0x7f615c58ac65]
[25f78357:2492856] [12] /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x366781)[0x7f615c589781]
[25f78357:2492856] [13] /root/tt-wavelet/build/tt-metal/tt_metal/libtt_metal.so(+0x365c79)[0x7f615c588c79]
[25f78357:2492856] [14] /lib/x86_64-linux-gnu/libstdc++.so.6(+0xdc253)[0x7f615bd27253]
[25f78357:2492856] [15] /lib/x86_64-linux-gnu/libc.so.6(+0x94ac3)[0x7f615b9adac3]
[25f78357:2492856] [16] /lib/x86_64-linux-gnu/libc.so.6(+0x1268d0)[0x7f615ba3f8d0]
[25f78357:2492856] *** End of error message ***


Exit code 1
Worked for 14m 50s
The first runtime test reached TT-Metal JIT and exposed a kernel include-path issue: device-side generated headers cannot use repo-root include strings. I’m making the kernel-facing headers use relative includes so the JIT compiler can resolve them from the generated kernel directory.

Edited 3 files
Context automatically compacted
Error running remote compact task: You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at 7:06 PM.
15 files changed
+1323
-955
Undo
Review

compare.py
scripts/generate_static_schemes.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_JSON_DIR = REPO_ROOT / "ttnn-wavelet" / "lifting_schemes"
DEFAULT_SCHEME_DIR = REPO_ROOT / "tt-wavelet" / "tt_wavelet" / "include" / "schemes" / "generated"
DEFAULT_REGISTRY = DEFAULT_SCHEME_DIR / "registry.hpp"
DEFAULT_COMPUTE_DIR = REPO_ROOT / "tt-wavelet" / "kernels" / "compute" / "generated"

STEP_TYPES = {
    "predict": "StepType::kPredict",
    "update": "StepType::kUpdate",
    "scale-even": "StepType::kScaleEven",
    "scale-odd": "StepType::kScaleOdd",
    "swap": "StepType::kSwap",
}


@dataclass(frozen=True)
class Step:
    kind: str
    shift: int
    coeff_bits: tuple[int, ...]


@dataclass(frozen=True)
class Scheme:
    name: str
    ident: str
    tap_size: int
    delay_even: int
    delay_odd: int
    steps: tuple[Step, ...]


def make_ident(name: str) -> str:
    ident = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"w_{ident}"
    return ident


def parse_coeff(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, dict):
        return float(raw["numerator"]) / float(raw["denominator"])
    raise TypeError(f"Unsupported coefficient encoding: {raw!r}")


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def load_scheme(path: Path) -> Scheme:
    obj = json.loads(path.read_text(encoding="utf-8"))
    name = path.stem
    steps: list[Step] = []
    for raw_step in obj["steps"]:
        kind = raw_step["type"]
        if kind not in STEP_TYPES:
            raise ValueError(f"{path.name}: unsupported step type {kind!r}")
        coeff_bits = tuple(f32_bits(parse_coeff(coeff)) for coeff in raw_step.get("coefficients", []))
        if kind in {"scale-even", "scale-odd"} and len(coeff_bits) != 1:
            raise ValueError(f"{path.name}: {kind} must have exactly one coefficient")
        if kind == "swap" and coeff_bits:
            raise ValueError(f"{path.name}: swap must not have coefficients")
        steps.append(Step(kind=kind, shift=int(raw_step["shift"]), coeff_bits=coeff_bits))

    return Scheme(
        name=name,
        ident=make_ident(name),
        tap_size=int(obj["tap_size"]),
        delay_even=int(obj["delay"]["even"]),
        delay_odd=int(obj["delay"]["odd"]),
        steps=tuple(steps),
    )


def coeff_args(step: Step) -> str:
    if not step.coeff_bits:
        return ""
    return ", " + ", ".join(f"0x{bits:08x}U" for bits in step.coeff_bits)


def render_scheme_header(scheme: Scheme) -> str:
    lines: list[str] = [
        "#pragma once",
        "",
        '#include "../../lifting/static_scheme.hpp"',
        "",
        "namespace ttwv::schemes {",
        "",
        f"struct {scheme.ident} {{",
        f'    static constexpr const char* name = "{scheme.name}";',
        f"    static constexpr uint32_t tap_size = {scheme.tap_size}U;",
        f"    static constexpr int32_t delay_even = {scheme.delay_even};",
        f"    static constexpr int32_t delay_odd = {scheme.delay_odd};",
        f"    static constexpr uint32_t num_steps = {len(scheme.steps)}U;",
        (
            f'    static constexpr const char* compute_kernel_path = '
            f'"kernels/compute/generated/lwt_fused_{scheme.ident}_compute.cpp";'
        ),
        "",
        "    template <std::size_t I>",
        "    struct step;",
        "};",
        "",
    ]

    for index, step in enumerate(scheme.steps):
        lines.extend(
            [
                "template <>",
                f"struct {scheme.ident}::step<{index}> {{",
                (
                    f"    using type = StaticStep<{STEP_TYPES[step.kind]}, {step.shift}"
                    f"{coeff_args(step)}>;"
                ),
                f"    static_assert(type::k == {len(step.coeff_bits)}U);",
                "};",
                "",
            ]
        )

    lines.extend(["}  // namespace ttwv::schemes", ""])
    return "\n".join(lines)


def render_registry(schemes: list[Scheme]) -> str:
    first_ident = schemes[0].ident
    includes = "\n".join(f'#include "{scheme.ident}.hpp"' for scheme in schemes)
    enum_entries = "\n".join(f"    k{scheme.ident}," for scheme in schemes)
    info_entries = "\n".join(
        (
            f'    SchemeInfo{{"{scheme.name}", {scheme.tap_size}U, '
            f"{scheme.delay_even}, {scheme.delay_odd}, {len(scheme.steps)}U}},"
        )
        for scheme in schemes
    )

    scheme_id_checks = "\n".join(
        f'    if (name == "{scheme.name}") return SchemeId::k{scheme.ident};' for scheme in schemes
    )
    dispatch_cases = "\n".join(
        (
            f"        case SchemeId::k{scheme.ident}: "
            f"return fn.template operator()<schemes::{scheme.ident}>();"
        )
        for scheme in schemes
    )

    return "\n".join(
        [
            "#pragma once",
            "",
            "#include <array>",
            "#include <cstdint>",
            "#include <span>",
            "#include <string>",
            "#include <string_view>",
            "",
            includes,
            "",
            "#include <tt_stl/assert.hpp>",
            "",
            "namespace ttwv {",
            "",
            "struct SchemeInfo {",
            "    std::string_view name;",
            "    uint32_t tap_size;",
            "    int32_t delay_even;",
            "    int32_t delay_odd;",
            "    uint32_t num_steps;",
            "};",
            "",
            "enum class SchemeId : uint32_t {",
            enum_entries,
            "    kUnknown,",
            "};",
            "",
            f"inline constexpr std::array<SchemeInfo, {len(schemes)}> kSchemeInfos = {{",
            info_entries,
            "};",
            "",
            "[[nodiscard]] inline std::span<const SchemeInfo> available_wavelets() noexcept {",
            "    return kSchemeInfos;",
            "}",
            "",
            "[[nodiscard]] inline SchemeId scheme_id(std::string_view name) noexcept {",
            scheme_id_checks,
            "    return SchemeId::kUnknown;",
            "}",
            "",
            "template <typename Fn>",
            "decltype(auto) dispatch_scheme(std::string_view name, Fn&& fn) {",
            "    switch (scheme_id(name)) {",
            dispatch_cases,
            "        case SchemeId::kUnknown: break;",
            "    }",
            "    TT_THROW(\"Unsupported wavelet scheme: {}\", std::string{name});",
            f"    return fn.template operator()<schemes::{first_ident}>();",
            "}",
            "",
            "}  // namespace ttwv",
            "",
        ]
    )


def render_compute_wrapper(scheme: Scheme) -> str:
    return "\n".join(
        [
            '#include "../lwt_fused_compute_template.hpp"',
            f'#include "../../../tt_wavelet/include/schemes/generated/{scheme.ident}.hpp"',
            "",
            "void kernel_main() {",
            f"    ttwv::kernels::lwt_fused_compute<ttwv::schemes::{scheme.ident}>();",
            "}",
            "",
        ]
    )


def write_if_changed(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def remove_stale(directory: Path, keep: set[Path], pattern: str) -> None:
    if not directory.exists():
        return
    for path in directory.glob(pattern):
        if path not in keep:
            path.unlink()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate static TT-wavelet lifting schemes from JSON.")
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--scheme-dir", type=Path, default=DEFAULT_SCHEME_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--compute-dir", type=Path, default=DEFAULT_COMPUTE_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    json_files = sorted(args.json_dir.glob("*.json"))
    if not json_files:
        print(f"ERROR: no JSON schemes found in {args.json_dir}", file=sys.stderr)
        return 1

    schemes = sorted((load_scheme(path) for path in json_files), key=lambda scheme: scheme.name)
    if len({scheme.ident for scheme in schemes}) != len(schemes):
        raise RuntimeError("Generated scheme identifiers are not unique")

    kept_scheme_headers: set[Path] = set()
    kept_compute_wrappers: set[Path] = set()
    writes = 0

    for scheme in schemes:
        scheme_path = args.scheme_dir / f"{scheme.ident}.hpp"
        compute_path = args.compute_dir / f"lwt_fused_{scheme.ident}_compute.cpp"
        kept_scheme_headers.add(scheme_path)
        kept_compute_wrappers.add(compute_path)
        writes += write_if_changed(scheme_path, render_scheme_header(scheme))
        writes += write_if_changed(compute_path, render_compute_wrapper(scheme))

    writes += write_if_changed(args.registry, render_registry(schemes))
    remove_stale(args.scheme_dir, kept_scheme_headers | {args.registry}, "*.hpp")
    remove_stale(args.compute_dir, kept_compute_wrappers, "lwt_fused_*_compute.cpp")

    print(f"Generated {len(schemes)} static schemes ({writes} files changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
CMakeLists.txt
if(BUILD_TT_WAVELET)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_compile_definitions(TT_WAVELET_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  set(PROJECT_NAME lwt)
  add_custom_target(
    tt_wavelet_generate_static_schemes
    COMMAND
      ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/generate_static_schemes.py
      --json-dir ${CMAKE_SOURCE_DIR}/ttnn-wavelet/lifting_schemes
      --scheme-dir ${CMAKE_CURRENT_SOURCE_DIR}/tt_wavelet/include/schemes/generated
      --registry ${CMAKE_CURRENT_SOURCE_DIR}/tt_wavelet/include/schemes/generated/registry.hpp
      --compute-dir ${CMAKE_CURRENT_SOURCE_DIR}/kernels/compute/generated
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating static TT-wavelet lifting schemes")
  add_executable(
    ${PROJECT_NAME} main.cpp tt_wavelet/src/lifting/plan.cpp
        tt_wavelet/src/pad_split/device.cpp tt_wavelet/src/lifting/device.cpp)
  add_dependencies(${PROJECT_NAME} tt_wavelet_generate_static_schemes)
  set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                   "${CMAKE_BINARY_DIR}")
  add_executable(lwt_shift_recover_test tests/shift_recover_test.cpp)
      ${CMAKE_SOURCE_DIR}/tt-metal/tt_metal/api/tt-metalium
      ${CMAKE_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR})
  file(GLOB TT_WAVELET_JSON_INCLUDE_CANDIDATES
       "${CMAKE_SOURCE_DIR}/tt-metal/.cpmcache/nlohmann_json/*/include")
  foreach(LLK_INCLUDE_DIR IN LISTS TT_WAVELET_LLK_INCLUDE_CANDIDATES)
    if(EXISTS "${LLK_INCLUDE_DIR}")
      list(APPEND TT_WAVELET_LLK_INCLUDE_DIRS "${LLK_INCLUDE_DIR}")
    endif()
  endforeach()
  foreach(JSON_INCLUDE_DIR IN LISTS TT_WAVELET_JSON_INCLUDE_CANDIDATES)
    if(EXISTS "${JSON_INCLUDE_DIR}")
      list(APPEND TT_WAVELET_INCLUDE_DIRS "${JSON_INCLUDE_DIR}")
    endif()
  endforeach()
kernels/compute/lwt_compute_utils.hpp
#pragma once
#include "../../tt_wavelet/include/device_protocol/step_desc.hpp"
#include "../../tt_wavelet/include/device_protocol/lwt_config.hpp"
#include "../sfpi/stencil_sfpi.h"
kernels/compute/lwt_fused_compute_template.hpp
kernels/dataflow/lwt_fused_reader.cpp
kernels/dataflow/lwt_fused_writer.cpp
main.cpp
tt_wavelet/include/device_protocol/lwt_config.hpp
tt_wavelet/include/device_protocol/step_desc.hpp
tt_wavelet/include/lifting/device.hpp
tt_wavelet/include/lifting/plan.hpp
tt_wavelet/include/lifting/static_scheme.hpp
tt_wavelet/src/lifting/device.cpp
tt_wavelet/src/lifting/plan.cpp
