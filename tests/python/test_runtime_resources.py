# SPDX-FileCopyrightText: © 2026 Nikita Lenyk
#
# SPDX-License-Identifier: MIT

"""TT-Wavelet resource initialization and incomplete-package regression tests."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def package_root():
    import ttwt

    return Path(ttwt.__file__).resolve().parent


def run_python(code, cwd, package_parent=None):
    env = os.environ.copy()
    if package_parent:
        env["PYTHONPATH"] = str(package_parent) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=cwd, env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_initializes_root_and_canonical_reinitialization(tmp_path):
    import ttwt
    from ttwt import _ttwt as native

    root = package_root() / "resources"
    assert Path(native._get_resource_root()) == root
    assert set(ttwt.__all__) == {"dwt_coeff_len", "dwt", "idwt", "dwt_2d", "idwt_2d"}
    assert not hasattr(ttwt, "_set_resource_root")
    assert not hasattr(ttwt, "_get_resource_root")
    alias = tmp_path / "resources-link"
    alias.symlink_to(root, target_is_directory=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: native._set_resource_root(str(alias)), range(32)))
    assert Path(native._get_resource_root()) == root
    assert (
        len(list(root.rglob("*.hpp"))) == 121
    )  # Generated and shared/primitives/protocol/planner headers
    # Other resources are four SFPI .h files, the internal 32x16 .h facade and two LLK bodies,
    # and six kernel .cpp files.
    assert (root / "ttwt/device/kernels/primitives/tile_move_copy_32x16.h").is_file()


@pytest.mark.parametrize(
    "removed",
    [
        "resources",
        "resources/ttwt/device/kernels/compute/lwt_compute.cpp",
        "resources/ttwt/device/kernels/primitives/tile_move_copy_32x16.h",
        "resources/ttwt/device/kernels/primitives/blackhole/llk_math_unary_datacopy_32x16_api.h",
        "resources/ttwt/device/kernels/primitives/wormhole/llk_math_unary_datacopy_32x16_api.h",
        "resources/ttwt/generated/wavelet_schemes/db1.hpp",
    ],
)
def test_incomplete_package_fails_at_import(tmp_path, removed):
    original = package_root()
    copy = tmp_path / "ttwt"
    shutil.copytree(original, copy, ignore=shutil.ignore_patterns("__pycache__"))
    target = copy / removed
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    code = """
try:
    import ttwt
except RuntimeError as error:
    assert 'TT-Wavelet' in str(error)
    print(error)
else:
    raise AssertionError('Incomplete package silently used a development fallback')
"""
    run_python(code, tmp_path, tmp_path)


def test_invalid_and_empty_resources_are_rejected(tmp_path):
    from ttwt import _ttwt as native

    with pytest.raises(RuntimeError, match="expected a runtime resource directory"):
        native._set_resource_root(str(tmp_path / "absent"))
    with pytest.raises(RuntimeError, match="runtime resource is missing or empty"):
        native._set_resource_root(str(tmp_path))
    root = tmp_path / "corrupt"
    shutil.copytree(package_root() / "resources", root)
    missing = root / "ttwt/generated/wavelet_schemes/db1.hpp"
    missing.write_text("")
    with pytest.raises(RuntimeError, match="missing or empty.*db1.hpp"):
        native._set_resource_root(str(root))


def test_relocatable_native_module_requires_initialization(tmp_path):
    if os.environ.get("TT_WAVELET_EXPECT_RELOCATABLE") != "1":
        pytest.skip("This raw-module diagnostic requires the relocatable build")
    library = next(package_root().glob("_ttwt*.so"))
    code = f"""
import importlib.util
spec = importlib.util.spec_from_file_location('_ttwt', {str(library)!r})
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
try:
    native._get_resource_root()
except RuntimeError as error:
    assert 'not initialized' in str(error)
else:
    raise AssertionError('Relocatable module has a development fallback')
"""
    run_python(code, tmp_path)
