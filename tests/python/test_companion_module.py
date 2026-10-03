"""Phase 3 companion tests; baseline upstream tests remain untouched."""

import ctypes
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

API = ("dwt_coeff_len", "dwt", "idwt", "dwt_2d", "idwt_2d")


@pytest.mark.parametrize("imports", [
    "import ttnn; import ttwt",
    "import ttwt",
    "import ttwt; import ttnn",
    "import ttwt._ttwt",
])
def test_import_orders(imports, tmp_path):
    # A new process ensures preceding tests cannot disguise import-order issues.
    subprocess.run([sys.executable, "-c", imports + "; import ttwt; "
                    "assert ttwt.dwt_coeff_len(64, 'db1') == 32"],
                   cwd=tmp_path, check=True)


def loaded_libraries():
    files = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if path.startswith("/"):
            files.add(str(Path(path).resolve()))
    return files


def test_public_contract_and_ownership():
    import ttnn
    import ttwt
    import ttwt._ttwt as native

    assert set(ttwt.__all__) == set(API)
    assert all(callable(getattr(ttwt, name)) for name in API)
    assert all(not hasattr(ttnn, name) for name in API)
    assert ttwt.dwt_coeff_len(64, "db1") == 32
    assert ttwt.dwt_coeff_len(65, "db1") == 33
    with pytest.raises(TypeError):
        ttwt.dwt([1.0] * 64, "db1")  # .noconvert: no Tensor conversion hack
    with pytest.raises(TypeError):
        ttwt.dwt(None, "db1", "symmetric")  # boundary_mode is keyword-only

    files = loaded_libraries()
    runtime = {}
    for prefix in ("_ttnncpp.so", "libtt_metal.so", "libtt-umd.so", "libtt_stl.so"):
        matches = sorted(p for p in files if Path(p).name.startswith(prefix))
        assert len(matches) == 1, (prefix, matches)
        runtime[prefix] = matches[0]
    print("RUNTIME_PATHS=" + json.dumps({**runtime, "ttnn._ttnn": ttnn._ttnn.__file__,
                                        "ttwt._ttwt": native.__file__}, sort_keys=True))
    upstream = subprocess.check_output(["nm", "-D", "--defined-only", "-C", runtime["_ttnncpp.so"]], text=True)
    assert "ttnn::dwt(" not in upstream and "ttnn::idwt(" not in upstream
    binding_symbols = subprocess.check_output(["nm", "-D", "--defined-only", "-C", ttnn._ttnn.__file__], text=True)
    assert "bind_wavelet_operations" not in binding_symbols

    # Resolve the actual function pointer in the loaded companion through ELF.
    class DlInfo(ctypes.Structure):
        _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                    ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]
    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int
    library = ctypes.CDLL(native.__file__)
    symbols = subprocess.check_output(["nm", "-D", "--defined-only", native.__file__], text=True)
    for name in API:
        mangled = [line.split()[-1] for line in symbols.splitlines()
                   if line.split()[-1].startswith(f"_ZN4ttnn{len(name)}{name}E")]
        assert len(mangled) == 1, (name, mangled)
        info = DlInfo()
        assert dladdr(ctypes.cast(getattr(library, mangled[0]), ctypes.c_void_p), ctypes.byref(info))
        assert Path(info.filename.decode()).resolve() == Path(native.__file__).resolve()
        print(f"SYMBOL_OWNER {name}={info.filename.decode()}")


def test_blackhole_db1_tensor_config_and_cache():
    import torch
    import ttnn
    import ttwt

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=[0])
    try:
        device.enable_program_cache()
        config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        assert isinstance(config, ttnn.MemoryConfig)
        x = ttnn.from_torch(torch.ones(64, dtype=torch.float32), dtype=ttnn.float32,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        image = ttnn.from_torch(torch.ones((32, 32), dtype=torch.float32), dtype=ttnn.float32,
                                layout=ttnn.TILE_LAYOUT, device=device)

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

        a, d = twice("dwt", lambda: ttwt.dwt(x, "db1", memory_config=config),
                     [math.sqrt(2), 0], [32, 32])
        (xr,) = twice("idwt", lambda: (ttwt.idwt(a, d, "db1", 64, memory_config=config),), [1], [64])
        bands = twice("dwt_2d", lambda: ttwt.dwt_2d(image, "db1", memory_config=config),
                      [2, 0, 0, 0], [256] * 4)
        (ir,) = twice("idwt_2d", lambda: (ttwt.idwt_2d(*bands, "db1", (32, 32), memory_config=config),),
                      [1], [1024])
        # Also exercise authoritative optional preallocated-output casters.
        aa, dd = ttwt.dwt(x, "db1", output_tensors=(a, d))
        assert aa.buffer_address() == a.buffer_address() and dd.buffer_address() == d.buffer_address()
        rr = ttwt.idwt(a, d, "db1", 64, output_tensor=xr)
        assert rr.buffer_address() == xr.buffer_address()
        bs = ttwt.dwt_2d(image, "db1", output_tensors=bands)
        assert all(t.buffer_address() == ref.buffer_address() for t, ref in zip(bs, bands))
        rr2 = ttwt.idwt_2d(*bands, "db1", (32, 32), output_tensor=ir)
        assert rr2.buffer_address() == ir.buffer_address()
    finally:
        ttnn.close_mesh_device(device)
