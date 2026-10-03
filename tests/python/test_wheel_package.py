# SPDX-FileCopyrightText: © 2026 Nikita Lenyk
#
# SPDX-License-Identifier: MIT

"""Hardware-free wheel inventory, metadata and installed ELF checks."""

from email.parser import Parser
import os
from pathlib import Path
import re
import subprocess
import zipfile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
import pytest


@pytest.fixture
def wheel():
    path = os.environ.get("TT_WAVELET_WHEEL")
    if not path:
        pytest.skip("Set TT_WAVELET_WHEEL to the candidate wheel")
    with zipfile.ZipFile(path) as archive:
        yield archive


def test_wheel_metadata_and_runtime_closure(wheel):
    names = set(wheel.namelist())
    metadata_paths = [n for n in names if n.endswith(".dist-info/METADATA")]
    assert len(metadata_paths) == 1
    metadata = Parser().parsestr(wheel.read(metadata_paths[0]).decode())
    assert metadata["Name"] == "tt-wavelet"
    assert metadata["Version"] == "0.1.0"
    assert SpecifierSet(metadata["Requires-Python"]) == SpecifierSet(">=3.10,<3.11")
    assert [Requirement(r) for r in metadata.get_all("Requires-Dist", [])] == [
        Requirement("ttnn==0.79.0")
    ]
    extensions = {n for n in names if n.startswith("ttwt/_ttwt.") and n.endswith(".so")}
    assert len(extensions) == 1
    root = "ttwt/resources/ttwt/"
    resources = {n for n in names if n.startswith(root) and not n.endswith("/")}
    assert len(resources) == 134
    generated = {n for n in resources if n.startswith(root + "generated/wavelet_schemes/")}
    assert len(generated) == 108
    catalog = wheel.read(root + "generated/wavelet_schemes/scheme_catalog.hpp").decode()
    schemes = re.findall(r'SchemeInfo\{"([^"]+)",', catalog)
    assert len(schemes) == len(set(schemes)) == 106
    generated_root = root + "generated/wavelet_schemes/"
    expected_generated = {
        generated_root + re.sub(r"[^0-9A-Za-z_]", "_", name) + ".hpp" for name in schemes
    } | {generated_root + "scheme_catalog.hpp", generated_root + "scheme_dispatch.hpp"}
    assert generated == expected_generated
    inventory = Path(__file__).resolve().parents[2] / "src/ttwt/resources.cmake"
    source_list = inventory.read_text().split("set(TT_WAVELET_RUNTIME_FILES", 1)[1].split(")", 1)[0]
    expected_sources = {root + p for p in source_list.split()}
    assert len(expected_sources) == 26
    assert resources - generated == expected_sources
    assert all(wheel.read(n) for n in resources)
    dist_info = metadata_paths[0].split("/")[0] + "/"
    wheel_metadata = Parser().parsestr(wheel.read(dist_info + "WHEEL").decode())
    assert wheel_metadata.get_all("Tag") == ["cp310-cp310-manylinux_2_34_x86_64"]
    allowed = resources | extensions | {"ttwt/__init__.py"}
    assert all(n in allowed or n.startswith(dist_info) for n in names)


def test_installed_extension_runpath():
    import ttwt._ttwt as native

    dynamic = subprocess.check_output(["readelf", "-d", native.__file__], text=True)
    paths = re.findall(r"\((?:RUNPATH|RPATH)\).*?\[([^]]*)\]", dynamic)
    assert paths == ["$ORIGIN/../ttnn/build/lib"]
    needed = re.findall(r"\(NEEDED\).*?\[([^]]*)\]", dynamic)
    assert "_ttnncpp.so" in needed and "libtt_metal.so" in needed
    assert all("/" not in library for library in needed)
