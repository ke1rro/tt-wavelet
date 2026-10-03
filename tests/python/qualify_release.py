"""Installed-wheel correctness qualification; no source package or private runtime.

Run from /tmp with --phase host/matrix1d/matrix2d/edges/preallocated/cache.
Results are append-only JSONL, including every failed operation. No exclusions.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import pywt
import torch
import ttnn
import ttwt
import ttwt._ttwt as native

ATOL, RTOL = 5e-5, 1e-5
REPRESENTATIVES = ("db1", "db38", "sym20", "coif17", "bior4.4", "rbio6.8", "dmey")


def inventory():
    root = Path(native._get_resource_root())
    catalog = root / "ttwt/generated/wavelet_schemes/scheme_catalog.hpp"
    items = re.findall(r'SchemeInfo\{"([^"]+)", (\d+)U,', catalog.read_text())
    assert len(items) == 106 and len({name for name, _ in items}) == 106
    doc = ttwt.dwt.__doc__
    mode_text = doc.split("``boundary_mode`` is one of ", 1)[1].split(".\n", 1)[0]
    modes = re.findall(r"``([^\x60]+)``", mode_text)
    assert len(modes) == 8 and all(mode in pywt.Modes.modes for mode in modes)
    assert {name for name, _ in items} == set(pywt.wavelist(kind="discrete"))
    for name, taps in items:
        assert pywt.Wavelet(name).dec_len == int(taps)
    return dict(items), modes, root, catalog


def signal(shape, kind, tag):
    if kind == "impulse":
        result = np.zeros(shape, dtype=np.float32)
        result.flat[result.size // 2] = 1
        return result
    if kind == "ramp":
        return np.linspace(-1, 1, math.prod(shape), dtype=np.float32).reshape(shape)
    seed = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "little")
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def classify(exc):
    text = str(exc).lower()
    if any(word in text for word in ("compilation", "failed to compile", "fatal error", "sfpi")):
        return "JIT compile"
    if any(word in text for word in ("must", "require", "invalid", "unsupported", "expected", "l1", "geometry")):
        return "planner/validation"
    if isinstance(exc, (ValueError, AssertionError)):
        return "reference-comparison issue"
    return "runtime/device"


class Sweep:
    def __init__(self, output, phase):
        output.mkdir(parents=True, exist_ok=True)
        self.file = (output / (phase + ".jsonl")).open("w")
        self.output = output
        self.phase = phase
        self.started = time.monotonic()
        self.count = 0
        self.failures = 0

    def write(self, **row):
        self.count += 1
        self.failures += not row.get("passed", False)
        row.update(phase=self.phase, elapsed_seconds=time.monotonic() - self.started)
        self.file.write(json.dumps(row, allow_nan=False) + "\n")
        self.file.flush()
        return row

    def compare(self, actuals, references, **metadata):
        errors, scores, shapes = [], [], []
        try:
            assert len(actuals) == len(references)
            for actual, reference in zip(actuals, references):
                assert actual.shape == reference.shape, (actual.shape, reference.shape)
                diff = np.abs(actual.astype(np.float64) - reference)
                errors.append(float(np.max(diff)))
                scores.append(float(np.max(diff / (ATOL + RTOL * np.abs(reference)))))
                shapes.append(list(actual.shape))
            finite = all(np.isfinite(v) for v in errors + scores)
            passed = finite and max(scores) <= 1
            diagnostic = None
            if not passed:
                diagnostic = self.phase + "-failure-" + str(self.count + 1) + ".npz"
                np.savez_compressed(self.output / diagnostic,
                    **{f"actual_{i}": v for i, v in enumerate(actuals)},
                    **{f"reference_{i}": v for i, v in enumerate(references)})
            return self.write(**metadata, diagnostic=diagnostic, passed=bool(passed),
                classification=None if passed else "numerical mismatch",
                max_error=max(errors) if finite else None,
                tolerance_ratio=max(scores) if finite else None,
                band_errors=errors if finite else None, output_shapes=shapes,
                atol=ATOL, rtol=RTOL)
        except Exception as exc:
            return self.error(exc, **metadata)

    def error(self, exc, **metadata):
        return self.write(**metadata, passed=False, classification=classify(exc),
                          max_error=None, exception=str(exc))


def upload(data, device, dim):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)),
        dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT if dim == 1 else ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def host_data(tensor, shape, dim):
    assert type(tensor) is ttnn.Tensor
    if dim == 1:
        expected = (math.ceil(shape[0] / 32), 32)
        assert tuple(tensor.shape) == expected, (tuple(tensor.shape), expected)
        # Defined stick-native representation: only logical prefix is valid.
        return ttnn.to_torch(tensor).flatten()[:shape[0]].numpy().copy()
    assert tuple(tensor.shape) == tuple(shape), (tuple(tensor.shape), shape)
    return ttnn.to_torch(tensor).numpy().copy()


def coefficients(data, name, mode, dim):
    if dim == 1:
        return tuple(pywt.dwt(data.astype(np.float64), name, mode=mode))
    ll, (hl, lh, hh) = pywt.dwt2(data.astype(np.float64), name, mode=mode)
    return ll, lh, hl, hh  # TTWT band order is LL,LH/cV,HL/cH,HH.


def reference_inverse(bands, name, mode, shape, dim):
    if dim == 1:
        return pywt.idwt(*bands, name, mode=mode)[:shape[0]]
    ll, lh, hl, hh = bands
    return pywt.idwt2((ll, (hl, lh, hh)), name, mode=mode)[:shape[0], :shape[1]]


def forward(x, name, mode, dim, outputs=None):
    fn = ttwt.dwt if dim == 1 else ttwt.dwt_2d
    return fn(x, name, boundary_mode=mode, memory_config=ttnn.DRAM_MEMORY_CONFIG,
              output_tensors=outputs)


def inverse(bands, name, mode, shape, dim, output=None):
    fn = ttwt.idwt if dim == 1 else ttwt.idwt_2d
    original = shape[0] if dim == 1 else shape
    return fn(*bands, name, original, boundary_mode=mode,
              memory_config=ttnn.DRAM_MEMORY_CONFIG, output_tensor=output)


def combination(sweep, device, name, mode, shape, dim, kind="random"):
    meta = dict(scheme=name, boundary_mode=mode, input_shape=list(shape), data=kind)
    data = signal(shape, kind, name + mode + str(shape) + kind)
    fop, iop = ("dwt", "idwt") if dim == 1 else ("dwt_2d", "idwt_2d")
    expected_invalid = mode in ("reflect", "antireflect") and min(shape) == 1
    try:
        refs = coefficients(data, name, mode, dim) if not expected_invalid else None
        x = upload(data, device, dim)
        outputs = forward(x, name, mode, dim)
        if expected_invalid:
            sweep.write(**meta, operation=fop, passed=False, classification="unsupported input",
                        exception="Expected rejection for singleton reflect/antireflect input")
            return
        coeff_shape = refs[0].shape
        actuals = tuple(host_data(t, coeff_shape, dim) for t in outputs)
        sweep.compare(actuals, refs, **meta, operation=fop,
                      coefficient_shape=list(coeff_shape))
    except Exception as exc:
        if expected_invalid and "greater than one" in str(exc):
            sweep.write(**meta, operation=fop, passed=True, classification="unsupported input",
                        expected_exception=str(exc), max_error=None)
        else:
            sweep.error(exc, **meta, operation=fop)
        outputs = None
        if expected_invalid:
            # Geometry is valid with symmetric coefficients; the requested singleton
            # reflect/antireflect inverse must still reject the original dimensions.
            try:
                valid_bands = coefficients(data, name, "symmetric", dim)
                buffers = tuple(upload(a, device, dim) for a in valid_bands)
                inverse(buffers, name, mode, shape, dim)
                sweep.write(**meta, operation=iop, passed=False, classification="unsupported input",
                            exception="Expected singleton inverse rejection")
            except RuntimeError as inverse_exc:
                sweep.write(**meta, operation=iop,
                            passed="greater than one" in str(inverse_exc),
                            classification="unsupported input", expected_exception=str(inverse_exc))
            return
    # Independent inverse uses reference coefficients, not TTWT-produced coefficients.
    try:
        rounded = tuple(np.asarray(a, dtype=np.float32) for a in refs)
        ref_bands = tuple(upload(a, device, dim) for a in rounded)
        recon = inverse(ref_bands, name, mode, shape, dim)
        expected = reference_inverse(tuple(a.astype(np.float64) for a in rounded),
                                     name, mode, shape, dim)
        sweep.compare((host_data(recon, shape, dim),), (expected,), **meta,
                      operation=iop, coefficient_source="PyWavelets")
    except Exception as exc:
        sweep.error(exc, **meta, operation=iop, coefficient_source="PyWavelets")
    if outputs is not None:
        try:
            recon = inverse(outputs, name, mode, shape, dim)
            baseline = reference_inverse(refs, name, mode, shape, dim)
            reference_roundtrip_error = float(np.max(np.abs(baseline - data)))
            sweep.compare((host_data(recon, shape, dim),), (data.astype(np.float64),),
                          **meta, operation=iop + "_roundtrip",
                          reference_roundtrip_error=reference_roundtrip_error)
        except Exception as exc:
            sweep.error(exc, **meta, operation=iop + "_roundtrip")
    else:
        sweep.write(**meta, operation=iop + "_roundtrip", passed=False,
                    classification="planner/validation", exception="Forward did not produce coefficients")


def preallocated(sweep, device, name, dim):
    shape = (257,) if dim == 1 else (33, 35)
    mode = "symmetric"
    meta = dict(scheme=name, boundary_mode=mode, input_shape=list(shape), data="random")
    try:
        data = signal(shape, "random", "preallocated" + name)
        x = upload(data, device, dim)
        outs = forward(x, name, mode, dim)
        refs = coefficients(data, name, mode, dim)
        ordinary = tuple(host_data(t, refs[0].shape, dim) for t in outs)
        reused = forward(x, name, mode, dim, outs)
        assert all(a.buffer_address() == b.buffer_address() for a, b in zip(outs, reused))
        sweep.compare(tuple(host_data(t, refs[0].shape, dim) for t in reused), ordinary,
                      **meta, operation=("dwt" if dim == 1 else "dwt_2d") + "_preallocated",
                      buffers_reused=True)
        xr = inverse(outs, name, mode, shape, dim)
        ordinary = host_data(xr, shape, dim)
        rr = inverse(outs, name, mode, shape, dim, xr)
        assert rr.buffer_address() == xr.buffer_address()
        sweep.compare((host_data(rr, shape, dim),), (ordinary,), **meta,
                      operation=("idwt" if dim == 1 else "idwt_2d") + "_preallocated",
                      buffers_reused=True)
    except Exception as exc:
        sweep.error(exc, **meta, operation="preallocated_" + str(dim) + "d")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("host", "matrix1d", "matrix2d", "edges", "preallocated", "cache", "smooth_shapes"), required=True)
    parser.add_argument("--schemes", nargs="*")
    args = parser.parse_args()
    names, modes, root, catalog = inventory()
    assert importlib.metadata.version("tt-wavelet") == "0.1.0"
    assert importlib.metadata.version("ttnn") == "0.79.0"
    assert Path(ttwt.__file__).resolve().is_relative_to(Path(__import__("sys").prefix))
    info = dict(schemes=list(names), tap_sizes=names, modes=modes, package=ttwt.__file__,
                resource_root=str(root), catalog=str(catalog),
                catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
                reference_version=pywt.__version__, atol=ATOL, rtol=RTOL)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "inventory.json").write_text(json.dumps(info, indent=2))
    sweep = Sweep(args.output, args.phase)
    selected = args.schemes or list(names)
    if args.phase == "host":
        for name, taps in names.items():
            taps = int(taps)
            lengths = sorted(set(range(1, 513)) | {taps-1, taps, taps+1, 1023, 1024, 1025, 4095, 4096, 4097})
            for mode in modes:
                for length in lengths:
                    try:
                        actual = ttwt.dwt_coeff_len(length, name)
                        expected = pywt.dwt_coeff_len(length, pywt.Wavelet(name).dec_len, mode)
                        sweep.write(scheme=name, boundary_mode=mode, operation="dwt_coeff_len",
                                    input_shape=[length], passed=actual == expected,
                                    classification=None if actual == expected else "reference-comparison issue",
                                    actual=actual, expected=expected, max_error=abs(actual-expected))
                    except Exception as exc:
                        sweep.error(exc, scheme=name, boundary_mode=mode,
                                    operation="dwt_coeff_len", input_shape=[length])
        for bad in ("DB1", "db01", "not_a_wavelet"):
            try:
                ttwt.dwt_coeff_len(64, bad)
                sweep.write(scheme=bad, operation="unknown_scheme", passed=False)
            except RuntimeError as exc:
                sweep.write(scheme=bad, operation="unknown_scheme", passed=True, expected_exception=str(exc))
        try:
            ttwt.dwt_coeff_len(0, "db1")
            sweep.write(scheme="db1", operation="zero_length", passed=False)
        except RuntimeError as exc:
            sweep.write(scheme="db1", operation="zero_length", passed=True, expected_exception=str(exc))
    else:
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1,1), physical_device_ids=[0])
        device.enable_program_cache()
        try:
            if args.phase in ("matrix1d", "matrix2d"):
                dim = 1 if args.phase == "matrix1d" else 2
                shape = (257,) if dim == 1 else (33, 35)
                for index, name in enumerate(selected):
                    for mode in modes:
                        combination(sweep, device, name, mode, shape, dim)
                    print(json.dumps(dict(scheme=name, completed=index+1, total=len(selected),
                        records=sweep.count, elapsed=time.monotonic()-sweep.started,
                        cache_entries=device.num_program_cache_entries())), flush=True)
            elif args.phase == "edges":
                # Multiple lengths for every scheme, without repeating the entire mode Cartesian product.
                for name in selected:
                    taps = pywt.Wavelet(name).dec_len
                    for i, length in enumerate(sorted({2, taps-1, taps, taps+1, 258, 1024})):
                        combination(sweep, device, name, "symmetric", (length,), 1,
                                    ("random", "impulse", "ramp")[i % 3])
                    print("LENGTHS " + name, flush=True)
                for name in REPRESENTATIVES:
                    taps = pywt.Wavelet(name).dec_len
                    for i, length in enumerate(sorted({1,2,3,7,31,32,33,taps-1,taps,taps+1,511,1024})):
                        for mode in modes:
                            combination(sweep, device, name, mode, (length,), 1,
                                        ("random","impulse","ramp")[i % 3])
                    for i, shape in enumerate(((1,1),(2,3),(16,32),(32,32),(65,97),(127,130))):
                        for mode in ("symmetric","antireflect"):
                            combination(sweep, device, name, mode, shape, 2,
                                        ("random","impulse","ramp")[i % 3])
                    print("EDGE " + name, flush=True)
            elif args.phase == "smooth_shapes":
                # Diagnose extrapolation precision without changing acceptance criteria.
                for name in REPRESENTATIVES:
                    for shape in ((16,32),(32,32),(65,97),(127,130)):
                        for kind in ("random","impulse","ramp"):
                            combination(sweep, device, name, "smooth", shape, 2, kind)
                    print("SMOOTH_SHAPES " + name, flush=True)
            elif args.phase == "preallocated":
                for name in REPRESENTATIVES:
                    for dim in (1,2): preallocated(sweep, device, name, dim)
            else:
                for expected_count, (name, length, mode) in zip((1,1,2,2,3,4), (("db1",257,"symmetric"),("db1",257,"symmetric"),
                                          ("db2",257,"symmetric"),("db2",257,"symmetric"),
                                          ("db1",258,"symmetric"),("db1",257,"zero"))):
                    x = upload(signal((length,), "random", "cache"), device, 1)
                    before = device.num_program_cache_entries()
                    outs = forward(x,name,mode,1)
                    ttnn.to_torch(outs[0])
                    after = device.num_program_cache_entries()
                    sweep.write(scheme=name,boundary_mode=mode,operation="cache",
                                input_shape=[length],passed=after == expected_count,
                                classification=None if after == expected_count else "runtime/device",
                                before=before,after=after,expected=expected_count)
                x_alias = upload(signal((257,), "random", "alias"), device, 1)
                db_bands = forward(x_alias, "db1", "symmetric", 1)
                haar_bands = forward(x_alias, "haar", "symmetric", 1)
                sweep.compare(tuple(host_data(t, (129,), 1) for t in haar_bands),
                              tuple(host_data(t, (129,), 1) for t in db_bands),
                              scheme="haar", boundary_mode="symmetric", input_shape=[257],
                              operation="haar_db1_alias")
                try:
                    forward(x_alias, "db1", "periodization", 1)
                    sweep.write(scheme="db1",operation="unsupported_boundary",passed=False)
                except RuntimeError as exc:
                    sweep.write(scheme="db1",operation="unsupported_boundary",
                                passed="Unsupported wavelet boundary" in str(exc),
                                expected_exception=str(exc))
                x = ttnn.from_torch(torch.ones(32),dtype=ttnn.float32,layout=ttnn.TILE_LAYOUT,device=device)
                try:
                    ttwt.dwt(x,"db1")
                    sweep.write(scheme="db1",operation="unsupported_layout",passed=False)
                except RuntimeError as exc:
                    sweep.write(scheme="db1",operation="unsupported_layout",
                                passed="ROW_MAJOR" in str(exc) or "row-major" in str(exc), expected_exception=str(exc))
                x_bf16 = ttnn.from_torch(torch.ones(32), dtype=ttnn.bfloat16,
                                         layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                try:
                    ttwt.dwt(x_bf16, "db1")
                    sweep.write(scheme="db1",operation="unsupported_dtype",passed=False)
                except RuntimeError as exc:
                    sweep.write(scheme="db1",operation="unsupported_dtype",
                                passed="FLOAT32" in str(exc) or "fp32" in str(exc),
                                expected_exception=str(exc))
        finally:
            # Record loaded runtime identities while the exercised process is still alive.
            mapped = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                             if line.split()[-1].startswith("/") and any(token in line.split()[-1]
                             for token in ("_ttnn", "_ttwt", "libtt_metal", "libtt_stl", "libtt-umd"))})
            (args.output / (args.phase + "-maps.json")).write_text(json.dumps(mapped, indent=2))
            ttnn.close_mesh_device(device)
    sweep.file.close()
    (args.output / (args.phase + "-duration.json")).write_text(json.dumps(
        dict(seconds=time.monotonic()-sweep.started, records=sweep.count, failures=sweep.failures), indent=2))
    raise SystemExit(1 if sweep.failures else 0)


if __name__ == "__main__":
    main()
