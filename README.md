# TT-Wavelet

TT-Wavelet provides lifting wavelet transforms through the `ttwt` Python
companion package. It accepts and returns TTNN tensors using the same
TTNN/Metalium runtime. The v0.1.0 build baseline is official TT-Metal/TTNN `v0.79.0`, revision
`de546d3b146758714d900f11b218c8f9c805f410`. The Python runtime dependency
is exactly `ttnn==0.79.0`.

TT-Wavelet is the maintained standalone home of this implementation. The original
TT-Metal bounty contribution was reviewed there, then directed to a separate
library rather than a merge into TTNN/TT-Metal. The reviewed branch is historical
source provenance; TTNN remains an external runtime dependency.

## Source layout

- `src/ttwt/`: C++ implementation, planner, device kernels and scheme generator.
- `bindings/`: nanobind companion module, `ttwt._ttwt`.
- `python/ttwt/`: the public Python package.
- `tests/`: native/reference tests and companion interoperability tests.
- `cmake/`: pinned source dependency, SYSTEM SDK and TTNN Python integration.
- `third_party/tt-metal/`: the pinned external dependency.

Headers remain implementation files, not an installed public C++ SDK. Host
code includes Wavelet-owned headers as `ttwt/...`, with `src/` as the source
include root and the build's `generated/` as the generated include root. JIT
kernels keep relative includes within their resource tree. Wavelet-owned C++
entities use the `ttwt` namespace; TTNN types and `TTWavelet::Native` retain
their existing identities.

## Development build

Use a separately provisioned environment with the compiler, Python development
headers and TT-Metal prerequisites described in the migration reports. CMake
does not run dependency installation scripts or system package managers.

```bash
git submodule update --init --recursive
cmake -S . -B build -G Ninja \
  -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 \
  -DTT_WAVELET_BUILD_PYTHON=ON
cmake --build build -j
PYTHONPATH="$PWD/build/python${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest tests/python/test_companion_module.py
```

The default `SUBMODULE` mode builds normal matching TTNN Python bindings,
without altering dependency sources. Standalone Wavelet symbols are isolated
under `ttwt`. `SYSTEM` mode supports a matching normal SDK through `TT_WAVELET_TT_METAL_MODE=SYSTEM` and
`CMAKE_PREFIX_PATH`. Python SYSTEM builds use the v0.79.0 development SDK and nanobind 2.14.0;
the installed companion runs with official PyPI TTNN 0.79.0.

```python
import ttnn
import ttwt

# x is an existing device-resident ttnn.Tensor.
cA, cD = ttwt.dwt(x, "db1")
```

The five public functions are `dwt_coeff_len`, `dwt`, `idwt`, `dwt_2d` and
`idwt_2d`. Python bindings are optional and default OFF. Local builds retain
a staged resource tree at `build/python/ttwt/resources/`. Native development
uses that directory as its default; Python initializes resources from its own
package directory. PEP 517 wheel builds are available with the runtime restrictions below. Existing build directories from before layout normalization should
be replaced with a fresh configure/build directory.

## Schemes and tests

The authoritative JSON in `src/ttwt/schemes/` contains 106 schemes. Its
stdlib-only incremental generator produces 106 scheme headers plus catalog and
dispatch headers under `build/generated/ttwt/generated/wavelet_schemes/`.
Generated headers are build products.

The companion regression suite covers Tensor/MemoryConfig interoperability,
imports, runtime and symbol ownership, Blackhole db1 transforms and caching.
The extracted reference tests retain their original TT-Metal harness/API
requirements; they are not yet the standalone full-scheme test suite.

## Migration record and licenses

See `EXTRACTION_RESEARCH.md`, `PHASE1_EXTRACTION.md`, `PHASE2_NATIVE_BUILD.md`,
`PHASE3_PYTHON_BINDINGS.md`, `TT_UMD_REFERENCE.md`,
`PHASE3_5_REPOSITORY_LAYOUT.md`, `PHASE4A_TTNN_COEXISTENCE.md` and
`PHASE4B_RELOCATABLE_RESOURCES.md` for evidence and limitations.
Historical paths in
these records describe the layout at that phase.

Imported sources retain Apache-2.0 notices; upstream license/notice material is
in `licenses/`. The root MIT license preserves legacy attribution and does not
relicense imported TT-Metal code.

## Relocatable resource validation

Configure a separate local Python build with
`-DTT_WAVELET_RELOCATABLE_RESOURCES=ON -DTT_WAVELET_BUILD_PYTHON=ON`.
This disables native resource
fallback: package initialization must supply its `resources/` directory.
CMake incrementally stages the six kernel sources, recursive owned headers and
108 generated headers under `build/python/ttwt/resources/ttwt/`.

Run `tests/python/test_companion_module.py` and
`tests/python/test_runtime_resources.py` with the build's `python/` directory
on `PYTHONPATH`; set `TT_WAVELET_EXPECT_RELOCATABLE=1` for the no-default-root
test. Hardware tests require the matching, separately provisioned TTNN runtime.
This is a build-tree resource proof, not a complete pip/install workflow.

## v0.1 wheel and runtime compatibility

Supported validation target: **Ubuntu 22.04 LTS, x86_64, CPython 3.10,
Blackhole (p150b tested)**. Other operating systems, Linux distributions,
Python versions, architectures and Wormhole have not been validated.

The local wheel is `tt_wavelet-0.1.0-cp310-cp310-linux_x86_64.whl`.
It contains `ttwt`, its extension and 134 runtime resources. It does not bundle
TTNN, Metalium, UMD, STL or their compiler/runtime assets. The source version
is recorded in `VERSION`; imported Apache-2.0 code and legacy MIT attribution
are preserved in the wheel's license files.

The release baseline is **official TTNN 0.79.0** for both build-time SDK and
runtime. Wheel metadata pins `ttnn==0.79.0`; installing the local wheel
resolves that runtime automatically:

```bash
python -m pip install /path/to/tt_wavelet-0.1.0-cp310-cp310-linux_x86_64.whl
python -c 'import ttnn, ttwt; print(ttwt.__file__)'
```

There is no public TT-Wavelet release yet. Only the exact runtime version above
is validated. The bounty-fork revision is historical provenance, not the
current dependency baseline. The internal experimental 32x16 helper and both
architecture bodies are owned and shipped by TT-Wavelet.

### SFPI prerequisite

TTNN 0.79.0 does **not** install SFPI through pip. Its wheel deliberately omits
the compiler. Provision **SFPI 7.78.0 [935]** separately before hardware use;
the provider looks in `ttnn/runtime/sfpi`, then `/opt/tenstorrent/sfpi`.
A different system SFPI version is not the validated configuration.

Tenstorrent supplies official SFPI Debian/RPM packages and archives.
The release's system setup installs the version listed in
[`sfpi-version`](https://github.com/tenstorrent/tt-metal/blob/v0.79.0/tt_metal/sfpi-version);
TT-Installer also supports system SFPI provisioning. Use the exact 7.78.0
[official release](https://github.com/tenstorrent/sfpi/releases/tag/7.78.0),
rather than an unpinned latest installer.

For an isolated venv, the following manual archive provisioning uses TTNN's
existing package-local compiler lookup and leaves system tools untouched:

```bash
curl -fL -o sfpi_7.78.0_x86_64_debian.txz \
  https://github.com/tenstorrent/sfpi/releases/download/7.78.0/sfpi_7.78.0_x86_64_debian.txz
echo '464c64759e441b841d6bc06216dbd470d7a0f188e0ffa9ac8ebb2e404cf244db  sfpi_7.78.0_x86_64_debian.txz' | sha256sum -c -
TT_WAVELET_PROVIDER_RUNTIME=$(python -c 'from pathlib import Path; import ttnn; print(Path(ttnn.__file__).resolve().parent / "runtime")')
tar -xJf sfpi_7.78.0_x86_64_debian.txz -C "$TT_WAVELET_PROVIDER_RUNTIME"
"$TT_WAVELET_PROVIDER_RUNTIME/sfpi/compiler/bin/riscv-tt-elf-g++" --version
```

This is separate environment provisioning, not a TT-Wavelet install/import
hook. TT-Wavelet neither bundles nor downloads SFPI. Driver, firmware and
system prerequisites must also be provisioned independently. See
[Phase 4D](PHASE4D_RELEASE_BASELINE.md) for the clean-install results and
provider-source evidence.

### Building a local wheel

A separately provisioned development SDK matching the selected runtime baseline must expose
`TT-NN` / `TT-Metalium` CMake packages. The official TTNN runtime wheel does
not provide those development packages. The PEP 517 backend defaults to SYSTEM
mode and builds only the companion target:

```bash
CMAKE_PREFIX_PATH=/path/to/matching-sdk CC=clang-20 CXX=clang++-20 \
  python -m build --wheel
```

Build isolation supplies `scikit-build-core>=0.11,<0.12` and
`nanobind==2.14.0`; the SDK/compiler are external prerequisites. Python bindings
and relocatable resources are mandatory for wheel builds. Explicit
source-dependency builds remain available with
`-Ccmake.define.TT_WAVELET_TT_METAL_MODE=SUBMODULE` after the pinned submodules
and compiler prerequisites have been provisioned. No automatic clone or
dependency-install hook is added.

The initial distribution policy is wheel-first. An experimental source archive
was built and rebuilt into a wheel against the matching SDK; it intentionally
excludes `third_party/` and requires that external SDK. Binary compatibility
with an official runtime, complete provider qualification, release CI and
publication remain prerequisites for a production release.
