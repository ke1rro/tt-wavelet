# SPDX-FileCopyrightText: © 2026 Nikita Lenyk
#
# SPDX-License-Identifier: MIT

"""TT-Wavelet installed-wheel interoperability and runtime ownership regression tests."""

import ctypes
import importlib.metadata
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

API = ("dwt_coeff_len", "dwt", "idwt", "dwt_2d", "idwt_2d")


@pytest.mark.parametrize(
    "imports",
    [
        "import ttnn; import ttwt",
        "import ttwt",
        "import ttwt; import ttnn",
        "import ttwt._ttwt",
    ],
)
def test_installed_import_orders(imports, tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            imports + "; import ttwt; assert ttwt.dwt_coeff_len(64, 'db1') == 32",
        ],
        cwd=tmp_path,
        check=True,
    )


def test_installed_metadata_types_and_ownership():
    import ttnn
    import ttwt
    import ttwt._ttwt as native

    site = Path(sys.prefix) / "lib/python3.10/site-packages"
    assert Path(ttwt.__file__).resolve().is_relative_to(site)
    assert Path(ttnn.__file__).resolve().is_relative_to(site)
    assert Path(native._get_resource_root()) == Path(ttwt.__file__).parent / "resources"
    assert set(ttwt.__all__) == set(API)
    assert importlib.metadata.version("tt-wavelet") == "0.1.0"
    assert importlib.metadata.version("ttnn") == "0.79.0"
    assert importlib.metadata.requires("tt-wavelet") == ["ttnn==0.79.0"]
    print("PROVIDER=" + importlib.metadata.version("ttnn"))
    print("RESOURCE_ROOT=" + native._get_resource_root())
    print("UPSTREAM_WAVELET=" + str(hasattr(ttnn, "dwt")))
    assert ttwt.dwt_coeff_len(65, "db1") == 33
    with pytest.raises(TypeError):
        ttwt.dwt([1.0] * 64, "db1")

    files = {
        str(Path(line.split()[-1]).resolve())
        for line in Path("/proc/self/maps").read_text().splitlines()
        if line.split()[-1].startswith("/")
    }
    runtime = {}
    for family, prefix in (
        ("ttnn", "_ttnn."),
        ("ttnncpp", "_ttnncpp.so"),
        ("ttwt", "_ttwt."),
        ("metal", "libtt_metal.so"),
        ("umd", "libtt-umd.so"),
        ("stl", "libtt_stl.so"),
    ):
        matches = [p for p in files if Path(p).name.startswith(prefix)]
        assert len(matches) == 1, (family, matches)
        assert Path(matches[0]).is_relative_to(site), matches
        runtime[family] = matches[0]
    print("RUNTIME_PATHS=" + json.dumps(runtime, sort_keys=True))

    def defined(path):
        return {
            line.split()[-1]
            for line in subprocess.check_output(
                ["nm", "-D", "--defined-only", path], text=True
            ).splitlines()
        }

    names = defined(native.__file__)
    owned = re.compile(
        r"4ttwt|10operations7wavelet|(?:Ilwt|Lwt)[12]D"
        r"(?:DeviceOperation|ProgramFactory|Params|Inputs)"
    )
    for provider in (runtime["ttnncpp"], runtime["ttnn"]):
        assert not [name for name in names & defined(provider) if owned.search(name)]

    class DlInfo(ctypes.Structure):
        _fields_ = [
            ("filename", ctypes.c_char_p),
            ("base", ctypes.c_void_p),
            ("symbol", ctypes.c_char_p),
            ("address", ctypes.c_void_p),
        ]

    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int
    library = ctypes.CDLL(native.__file__)
    for name in API:
        mangled = [s for s in names if s.startswith(f"_ZN4ttwt{len(name)}{name}E")]
        assert len(mangled) == 1, (name, mangled)
        info = DlInfo()
        assert dladdr(
            ctypes.cast(getattr(library, mangled[0]), ctypes.c_void_p), ctypes.byref(info)
        )
        assert Path(info.filename.decode()).resolve() == Path(native.__file__).resolve()
        print(f"SYMBOL_OWNER ttwt.{name}={info.filename.decode()}")


@pytest.mark.hardware
def test_blackhole_db1_tensor_config_and_cache(tmp_path):
    import torch
    import ttnn
    import ttwt

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=[0])
    try:
        device.enable_program_cache()
        config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        assert isinstance(config, ttnn.MemoryConfig)
        x = ttnn.from_torch(
            torch.ones(64, dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        image = ttnn.from_torch(
            torch.ones((32, 32), dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        def check(tensor, expected, count):
            assert type(tensor) is ttnn.Tensor
            # An ordinary TTNN operation consumes the companion-produced object.
            actual = ttnn.to_torch(tensor).flatten()[:count]
            assert actual.numel() == count and torch.isfinite(actual).all()
            error = float(torch.max(torch.abs(actual - expected)))
            assert error <= 5e-5
            return error

        def twice(name, operation, expected, counts):
            before = device.num_program_cache_entries()
            results = operation()
            errors = [check(t, v, n) for t, v, n in zip(results, expected, counts)]
            first = device.num_program_cache_entries()
            repeated = operation()
            errors += [check(t, v, n) for t, v, n in zip(repeated, expected, counts)]
            second = device.num_program_cache_entries()
            assert first == before + 1 and first == second, (name, before, first, second)
            print(f"HARDWARE {name}: cache={before}/{first}/{second}, max_error={max(errors)}")
            return results

        a, d = twice(
            "dwt", lambda: ttwt.dwt(x, "db1", memory_config=config), [math.sqrt(2), 0], [32, 32]
        )
        (xr,) = twice(
            "idwt", lambda: (ttwt.idwt(a, d, "db1", 64, memory_config=config),), [1], [64]
        )
        bands = twice(
            "dwt_2d",
            lambda: ttwt.dwt_2d(image, "db1", memory_config=config),
            [2, 0, 0, 0],
            [256] * 4,
        )
        (ir,) = twice(
            "idwt_2d",
            lambda: (ttwt.idwt_2d(*bands, "db1", (32, 32), memory_config=config),),
            [1],
            [1024],
        )
        # Resource identity is frozen after descriptor/JIT use, without changing cache semantics.
        import ttwt._ttwt as native

        other_root = tmp_path / "other-resources"
        shutil.copytree(native._get_resource_root(), other_root)
        with pytest.raises(RuntimeError, match="after active use"):
            native._set_resource_root(str(other_root))
        # Also exercise authoritative optional preallocated-output casters.
        aa, dd = ttwt.dwt(x, "db1", output_tensors=(a, d))
        assert (
            aa.buffer_address() == a.buffer_address() and dd.buffer_address() == d.buffer_address()
        )
        rr = ttwt.idwt(a, d, "db1", 64, output_tensor=xr)
        assert rr.buffer_address() == xr.buffer_address()
        bs = ttwt.dwt_2d(image, "db1", output_tensors=bands)
        assert all(t.buffer_address() == ref.buffer_address() for t, ref in zip(bs, bands))
        rr2 = ttwt.idwt_2d(*bands, "db1", (32, 32), output_tensor=ir)
        assert rr2.buffer_address() == ir.buffer_address()
    finally:
        ttnn.close_mesh_device(device)


@pytest.mark.hardware
def test_upstream_wavelet_control():
    """Provider qualification, separate from standalone wheel correctness."""
    import torch
    import ttnn

    if not hasattr(ttnn, "dwt"):
        pytest.skip("The official provider does not expose upstream Wavelet")
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=[0])
    try:
        x = ttnn.from_torch(
            torch.ones(64), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        a, d = ttnn.dwt(x, "db1")
        assert type(a) is ttnn.Tensor and type(d) is ttnn.Tensor
        assert float(torch.max(torch.abs(ttnn.to_torch(a).flatten()[:32] - math.sqrt(2)))) <= 5e-5
        assert float(torch.max(torch.abs(ttnn.to_torch(d).flatten()[:32]))) <= 5e-5
    finally:
        ttnn.close_mesh_device(device)
