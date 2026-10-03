# SPDX-FileCopyrightText: © 2026 Nikita Lenyk
#
# SPDX-License-Identifier: MIT

"""Minimal device lifecycle for standalone accelerator regressions."""

import pytest


def pytest_addoption(parser):
    parser.addoption("--device-id", type=int, default=0, help="Physical Tenstorrent device ID")


@pytest.fixture
def device(request):
    import ttnn

    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[request.config.getoption("--device-id")],
    )
    try:
        mesh.enable_program_cache()
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)


@pytest.fixture
def expect_error():
    return lambda error, message: pytest.raises(error, match=message)
