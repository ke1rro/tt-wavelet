# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import pywt
import torch
import ttnn

BOUNDARY_MODES = [
    "zero",
    "constant",
    "symmetric",
    "reflect",
    "periodic",
    "smooth",
    "antisymmetric",
    "antireflect",
]


def to_device_1d(device, value, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        value,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def to_device_2d(device, value, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        value,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def assert_fp32_close(actual, expected, atol=5e-5):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=atol)


def assert_fp32_identical(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("length", [20, 31, 32, 33])
def test_lwt_ilwt_1d_stick_padding_regression(device, length):
    indices = torch.arange(length, dtype=torch.float32)
    signal = 0.125 * indices + torch.sin(0.7 * indices)
    approximation_ref, detail_ref = pywt.dwt(signal.numpy(), "bior1.3", mode="symmetric")

    approximation, detail = ttnn.lwt(
        to_device_1d(device, signal), "bior1.3", boundary_mode="symmetric"
    )

    assert tuple(approximation.shape) == approximation_ref.shape
    assert tuple(detail.shape) == detail_ref.shape
    assert_fp32_close(ttnn.to_torch(approximation), torch.from_numpy(approximation_ref))
    assert_fp32_close(ttnn.to_torch(detail), torch.from_numpy(detail_ref))

    reconstructed = ttnn.ilwt(
        approximation,
        detail,
        "bior1.3",
        length,
        boundary_mode="symmetric",
    )
    assert_fp32_close(ttnn.to_torch(reconstructed), signal)


@pytest.mark.parametrize("boundary_mode", BOUNDARY_MODES)
def test_lwt_ilwt_1d_boundary_modes(device, boundary_mode):
    signal = torch.linspace(-1.25, 2.75, 33, dtype=torch.float32)
    approximation_ref, detail_ref = pywt.dwt(signal.numpy(), "bior1.3", mode=boundary_mode)

    approximation, detail = ttnn.lwt(
        to_device_1d(device, signal),
        "bior1.3",
        boundary_mode=boundary_mode,
    )
    assert_fp32_close(ttnn.to_torch(approximation), torch.from_numpy(approximation_ref))
    assert_fp32_close(ttnn.to_torch(detail), torch.from_numpy(detail_ref))

    reconstructed = ttnn.ilwt(
        approximation,
        detail,
        "bior1.3",
        signal.numel(),
        boundary_mode=boundary_mode,
    )
    assert_fp32_close(ttnn.to_torch(reconstructed), signal)


def test_ilwt_1d_external_coefficients_shorter_than_one_stick(device):
    signal = (
        torch.arange(20, dtype=torch.float32) ** 2 * 0.03125
        - torch.arange(20, dtype=torch.float32) * 0.25
    )
    approximation, detail = pywt.dwt(signal.numpy(), "bior1.3", mode="symmetric")
    assert approximation.size < 32
    assert detail.size < 32

    reconstructed = ttnn.ilwt(
        to_device_1d(device, torch.from_numpy(approximation)),
        to_device_1d(device, torch.from_numpy(detail)),
        "bior1.3",
        signal.numel(),
        boundary_mode="symmetric",
    )
    assert_fp32_close(ttnn.to_torch(reconstructed), signal)


def test_wavelet_1d_interleaved_l1_input_matches_dram_multichunk(device):
    length = 65_537
    indices = torch.arange(length, dtype=torch.float32)
    signal = torch.sin(indices * 0.013) + indices * 1.0e-5

    dram_outputs = ttnn.lwt(to_device_1d(device, signal), "bior1.3", boundary_mode="antireflect")
    l1_input = to_device_1d(device, signal, ttnn.L1_MEMORY_CONFIG)
    l1_outputs = ttnn.lwt(l1_input, "bior1.3", boundary_mode="antireflect")
    for actual, expected in zip(l1_outputs, dram_outputs):
        assert actual.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert_fp32_identical(ttnn.to_torch(actual), ttnn.to_torch(expected))
    ttnn.deallocate(l1_input)

    dram_reconstructed = ttnn.ilwt(*dram_outputs, "bior1.3", length, boundary_mode="antireflect")
    approximation_host, detail_host = (ttnn.to_torch(tensor) for tensor in dram_outputs)
    approximation_l1 = to_device_1d(device, approximation_host, ttnn.L1_MEMORY_CONFIG)
    detail_l1 = to_device_1d(device, detail_host, ttnn.L1_MEMORY_CONFIG)

    l1_reconstructed = ttnn.ilwt(
        approximation_l1,
        detail_l1,
        "bior1.3",
        length,
        boundary_mode="antireflect",
    )
    mixed_reconstructed = ttnn.ilwt(
        approximation_l1,
        dram_outputs[1],
        "bior1.3",
        length,
        boundary_mode="antireflect",
    )
    assert_fp32_identical(ttnn.to_torch(l1_reconstructed), ttnn.to_torch(dram_reconstructed))
    assert_fp32_identical(ttnn.to_torch(mixed_reconstructed), ttnn.to_torch(dram_reconstructed))
    assert l1_reconstructed.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert mixed_reconstructed.memory_config() == ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize("shape", [(32, 32), (33, 31), (35, 37)])
@pytest.mark.parametrize("boundary_mode", BOUNDARY_MODES)
def test_lwt_ilwt_2d_shapes_and_boundary_modes(device, shape, boundary_mode):
    height, width = shape
    y = torch.arange(height, dtype=torch.float32).reshape(-1, 1)
    x = torch.arange(width, dtype=torch.float32).reshape(1, -1)
    signal = torch.sin(0.17 * x) + torch.cos(0.11 * y) + 0.01 * x - 0.02 * y

    ll_ref, (hl_ref, lh_ref, hh_ref) = pywt.dwt2(signal.numpy(), "bior1.3", mode=boundary_mode)
    ll, lh, hl, hh = ttnn.lwt_2d(
        to_device_2d(device, signal),
        "bior1.3",
        boundary_mode=boundary_mode,
    )

    # TTNN names bands by (vertical, horizontal) result. PyWavelets returns
    # its horizontal-detail band before its vertical-detail band.
    references = [ll_ref, lh_ref, hl_ref, hh_ref]
    for result, reference in zip((ll, lh, hl, hh), references):
        assert tuple(result.shape) == reference.shape
        assert_fp32_close(ttnn.to_torch(result), torch.from_numpy(reference))

    reconstructed = ttnn.ilwt_2d(
        ll,
        lh,
        hl,
        hh,
        "bior1.3",
        shape,
        boundary_mode=boundary_mode,
    )
    assert_fp32_close(ttnn.to_torch(reconstructed), signal)


def test_wavelet_2d_interleaved_l1_input_matches_dram_multichunk(device):
    shape = (257, 259)
    y = torch.arange(shape[0], dtype=torch.float32).reshape(-1, 1)
    x = torch.arange(shape[1], dtype=torch.float32).reshape(1, -1)
    signal = torch.sin(0.017 * x) + torch.cos(0.019 * y) + 1.0e-4 * x * y

    dram_outputs = ttnn.lwt_2d(to_device_2d(device, signal), "bior1.3", boundary_mode="antireflect")
    l1_input = to_device_2d(device, signal, ttnn.L1_MEMORY_CONFIG)
    l1_outputs = ttnn.lwt_2d(l1_input, "bior1.3", boundary_mode="antireflect")
    for actual, expected in zip(l1_outputs, dram_outputs):
        assert actual.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert_fp32_identical(ttnn.to_torch(actual), ttnn.to_torch(expected))
    ttnn.deallocate(l1_input)

    dram_reconstructed = ttnn.ilwt_2d(
        *dram_outputs,
        "bior1.3",
        shape,
        boundary_mode="antireflect",
    )
    l1_bands = tuple(
        to_device_2d(device, ttnn.to_torch(tensor), ttnn.L1_MEMORY_CONFIG)
        for tensor in dram_outputs
    )
    l1_reconstructed = ttnn.ilwt_2d(
        *l1_bands,
        "bior1.3",
        shape,
        boundary_mode="antireflect",
    )
    mixed_reconstructed = ttnn.ilwt_2d(
        l1_bands[0],
        dram_outputs[1],
        l1_bands[2],
        dram_outputs[3],
        "bior1.3",
        shape,
        boundary_mode="antireflect",
    )
    assert_fp32_identical(ttnn.to_torch(l1_reconstructed), ttnn.to_torch(dram_reconstructed))
    assert_fp32_identical(ttnn.to_torch(mixed_reconstructed), ttnn.to_torch(dram_reconstructed))
    assert l1_reconstructed.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert mixed_reconstructed.memory_config() == ttnn.DRAM_MEMORY_CONFIG


def test_wavelet_preallocated_outputs_and_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        signal = torch.arange(20, dtype=torch.float32)
        input_tensor = to_device_1d(device, signal)
        approximation, detail = ttnn.lwt(input_tensor, "bior1.3")
        reconstructed = ttnn.ilwt(approximation, detail, "bior1.3", signal.numel())

        # Retain correctly specified output tensors, then isolate the two
        # preallocated operation cache entries from the allocation run above.
        device.disable_and_clear_program_cache()
        device.enable_program_cache()
        for scale in (-2.5, 0.375):
            next_signal = signal * scale + 7.0
            approximation_out, detail_out = ttnn.lwt(
                to_device_1d(device, next_signal),
                "bior1.3",
                output_tensors=(approximation, detail),
            )
            reconstructed_out = ttnn.ilwt(
                approximation_out,
                detail_out,
                "bior1.3",
                signal.numel(),
                output_tensor=reconstructed,
            )
            assert approximation_out.buffer_address() == approximation.buffer_address()
            assert detail_out.buffer_address() == detail.buffer_address()
            assert reconstructed_out.buffer_address() == reconstructed.buffer_address()
            assert_fp32_close(ttnn.to_torch(reconstructed_out), next_signal)

        assert device.num_program_cache_entries() == 2
    finally:
        device.disable_and_clear_program_cache()


def test_wavelet_2d_preallocated_outputs_and_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        shape = (35, 37)
        signal = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape) * 0.001
        input_tensor = to_device_2d(device, signal)
        outputs = ttnn.lwt_2d(input_tensor, "bior1.3")
        reconstructed = ttnn.ilwt_2d(*outputs, "bior1.3", shape)

        device.disable_and_clear_program_cache()
        device.enable_program_cache()
        for scale in (-0.25, 0.5):
            next_signal = 1.0 + signal * scale
            next_outputs = ttnn.lwt_2d(
                to_device_2d(device, next_signal),
                "bior1.3",
                output_tensors=outputs,
            )
            reconstructed_out = ttnn.ilwt_2d(
                *next_outputs,
                "bior1.3",
                shape,
                output_tensor=reconstructed,
            )
            assert all(
                result.buffer_address() == output.buffer_address()
                for result, output in zip(next_outputs, outputs)
            )
            assert reconstructed_out.buffer_address() == reconstructed.buffer_address()
            assert_fp32_close(ttnn.to_torch(reconstructed_out), next_signal)

        assert device.num_program_cache_entries() == 2
    finally:
        device.disable_and_clear_program_cache()


def test_wavelet_operations_with_program_cache_disabled(device):
    device.disable_and_clear_program_cache()
    try:
        signal_1d = torch.linspace(-1.0, 1.0, 33, dtype=torch.float32)
        approximation, detail = ttnn.lwt(to_device_1d(device, signal_1d), "bior1.3")
        reconstructed_1d = ttnn.ilwt(approximation, detail, "bior1.3", signal_1d.numel())
        assert_fp32_close(ttnn.to_torch(reconstructed_1d), signal_1d)

        signal_2d = signal_1d.reshape(3, 11)
        bands = ttnn.lwt_2d(to_device_2d(device, signal_2d), "bior1.3")
        reconstructed_2d = ttnn.ilwt_2d(*bands, "bior1.3", signal_2d.shape)
        assert_fp32_close(ttnn.to_torch(reconstructed_2d), signal_2d)
        assert device.num_program_cache_entries() == 0
    finally:
        device.enable_program_cache()


def test_wavelet_1d_program_cache_keys_and_address_override(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        signal = torch.sin(torch.arange(33, dtype=torch.float32) * 0.17)
        first_input = to_device_1d(device, signal)
        first_outputs = ttnn.lwt(first_input, "db7")
        assert device.num_program_cache_entries() == 1

        # Identical tensors and new buffers with identical specs both reuse the
        # program. Tensor addresses are runtime arguments, not cache-key data.
        ttnn.lwt(first_input, "db7")
        second_input = to_device_1d(device, signal + 0.25)
        second_outputs = ttnn.lwt(second_input, "db7")
        assert device.num_program_cache_entries() == 1

        db8_outputs = ttnn.lwt(second_input, "db8")
        assert device.num_program_cache_entries() == 2

        reconstructed = ttnn.ilwt(*first_outputs, "db7", signal.numel())
        ttnn.ilwt(*first_outputs, "db7", signal.numel())
        ttnn.ilwt(*second_outputs, "db7", signal.numel())
        assert device.num_program_cache_entries() == 3
        assert_fp32_close(ttnn.to_torch(reconstructed), signal, atol=2e-4)

        ttnn.ilwt(*db8_outputs, "db8", signal.numel())
        assert device.num_program_cache_entries() == 4
    finally:
        device.disable_and_clear_program_cache()


def test_wavelet_2d_program_cache_keys_and_address_override(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        shape = (35, 37)
        signal = torch.sin(
            torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape) * 0.013
        )
        first_input = to_device_2d(device, signal)
        first_outputs = ttnn.lwt_2d(first_input, "db7")
        assert device.num_program_cache_entries() == 1

        ttnn.lwt_2d(first_input, "db7")
        second_input = to_device_2d(device, signal + 0.25)
        second_outputs = ttnn.lwt_2d(second_input, "db7")
        assert device.num_program_cache_entries() == 1

        db8_outputs = ttnn.lwt_2d(second_input, "db8")
        assert device.num_program_cache_entries() == 2

        reconstructed = ttnn.ilwt_2d(*first_outputs, "db7", shape)
        ttnn.ilwt_2d(*first_outputs, "db7", shape)
        ttnn.ilwt_2d(*second_outputs, "db7", shape)
        assert device.num_program_cache_entries() == 3
        assert_fp32_close(ttnn.to_torch(reconstructed), signal, atol=2e-4)

        ttnn.ilwt_2d(*db8_outputs, "db8", shape)
        assert device.num_program_cache_entries() == 4
    finally:
        device.disable_and_clear_program_cache()


def test_wavelet_1d_interleaved_l1_program_cache_keys_and_address_override(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        signal = torch.sin(torch.arange(33, dtype=torch.float32) * 0.17)
        dram_input_a = to_device_1d(device, signal)
        dram_outputs = ttnn.lwt(dram_input_a, "bior1.3")
        ttnn.lwt(to_device_1d(device, signal + 0.25), "bior1.3")
        assert device.num_program_cache_entries() == 1

        l1_input_a = to_device_1d(device, signal, ttnn.L1_MEMORY_CONFIG)
        l1_outputs = ttnn.lwt(l1_input_a, "bior1.3")
        ttnn.lwt(to_device_1d(device, signal + 0.25, ttnn.L1_MEMORY_CONFIG), "bior1.3")
        assert device.num_program_cache_entries() == 2
        for actual, expected in zip(l1_outputs, dram_outputs):
            assert_fp32_identical(ttnn.to_torch(actual), ttnn.to_torch(expected))

        coefficient_values = tuple(ttnn.to_torch(tensor) for tensor in dram_outputs)
        device.disable_and_clear_program_cache()
        device.enable_program_cache()

        dram_coefficients_a = tuple(to_device_1d(device, tensor) for tensor in coefficient_values)
        dram_coefficients_b = tuple(
            to_device_1d(device, tensor + 0.125) for tensor in coefficient_values
        )
        dram_reconstructed = ttnn.ilwt(*dram_coefficients_a, "bior1.3", signal.numel())
        ttnn.ilwt(*dram_coefficients_b, "bior1.3", signal.numel())
        assert device.num_program_cache_entries() == 1

        l1_coefficients_a = tuple(
            to_device_1d(device, tensor, ttnn.L1_MEMORY_CONFIG) for tensor in coefficient_values
        )
        l1_reconstructed = ttnn.ilwt(*l1_coefficients_a, "bior1.3", signal.numel())
        l1_coefficients_b = tuple(
            to_device_1d(device, tensor + 0.125, ttnn.L1_MEMORY_CONFIG)
            for tensor in coefficient_values
        )
        ttnn.ilwt(*l1_coefficients_b, "bior1.3", signal.numel())
        assert device.num_program_cache_entries() == 2

        mixed_reconstructed = ttnn.ilwt(
            l1_coefficients_a[0], dram_coefficients_a[1], "bior1.3", signal.numel()
        )
        ttnn.ilwt(l1_coefficients_b[0], dram_coefficients_b[1], "bior1.3", signal.numel())
        assert device.num_program_cache_entries() == 3
        assert_fp32_identical(ttnn.to_torch(l1_reconstructed), ttnn.to_torch(dram_reconstructed))
        assert_fp32_identical(ttnn.to_torch(mixed_reconstructed), ttnn.to_torch(dram_reconstructed))
    finally:
        device.disable_and_clear_program_cache()


def test_wavelet_2d_interleaved_l1_program_cache_keys_and_address_override(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        shape = (35, 37)
        signal = torch.sin(
            torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape) * 0.013
        )
        dram_input_a = to_device_2d(device, signal)
        dram_outputs = ttnn.lwt_2d(dram_input_a, "bior1.3")
        ttnn.lwt_2d(to_device_2d(device, signal + 0.25), "bior1.3")
        assert device.num_program_cache_entries() == 1

        l1_input_a = to_device_2d(device, signal, ttnn.L1_MEMORY_CONFIG)
        l1_outputs = ttnn.lwt_2d(l1_input_a, "bior1.3")
        ttnn.lwt_2d(to_device_2d(device, signal + 0.25, ttnn.L1_MEMORY_CONFIG), "bior1.3")
        assert device.num_program_cache_entries() == 2
        for actual, expected in zip(l1_outputs, dram_outputs):
            assert_fp32_identical(ttnn.to_torch(actual), ttnn.to_torch(expected))

        band_values = tuple(ttnn.to_torch(tensor) for tensor in dram_outputs)
        device.disable_and_clear_program_cache()
        device.enable_program_cache()

        dram_bands_a = tuple(to_device_2d(device, tensor) for tensor in band_values)
        dram_bands_b = tuple(to_device_2d(device, tensor + 0.125) for tensor in band_values)
        dram_reconstructed = ttnn.ilwt_2d(*dram_bands_a, "bior1.3", shape)
        ttnn.ilwt_2d(*dram_bands_b, "bior1.3", shape)
        assert device.num_program_cache_entries() == 1

        l1_bands_a = tuple(
            to_device_2d(device, tensor, ttnn.L1_MEMORY_CONFIG) for tensor in band_values
        )
        l1_reconstructed = ttnn.ilwt_2d(*l1_bands_a, "bior1.3", shape)
        l1_bands_b = tuple(
            to_device_2d(device, tensor + 0.125, ttnn.L1_MEMORY_CONFIG) for tensor in band_values
        )
        ttnn.ilwt_2d(*l1_bands_b, "bior1.3", shape)
        assert device.num_program_cache_entries() == 2

        mixed_reconstructed = ttnn.ilwt_2d(
            l1_bands_a[0], dram_bands_a[1], l1_bands_a[2], dram_bands_a[3], "bior1.3", shape
        )
        ttnn.ilwt_2d(
            l1_bands_b[0], dram_bands_b[1], l1_bands_b[2], dram_bands_b[3], "bior1.3", shape
        )
        assert device.num_program_cache_entries() == 3
        assert_fp32_identical(ttnn.to_torch(l1_reconstructed), ttnn.to_torch(dram_reconstructed))
        assert_fp32_identical(ttnn.to_torch(mixed_reconstructed), ttnn.to_torch(dram_reconstructed))
    finally:
        device.disable_and_clear_program_cache()


def test_wavelet_1d_validation_errors(device):
    signal = torch.arange(20, dtype=torch.float32)
    input_tensor = to_device_1d(device, signal)

    with pytest.raises(RuntimeError, match="wavelet"):
        ttnn.lwt(input_tensor, "not-a-wavelet")
    with pytest.raises(RuntimeError, match="boundary"):
        ttnn.lwt(input_tensor, "bior1.3", boundary_mode="not-a-mode")
    with pytest.raises(RuntimeError, match="device tensor"):
        ttnn.lwt(ttnn.from_torch(signal, layout=ttnn.ROW_MAJOR_LAYOUT), "bior1.3")
    with pytest.raises(RuntimeError, match="FLOAT32"):
        ttnn.lwt(
            ttnn.from_torch(
                signal, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            ),
            "bior1.3",
        )
    with pytest.raises(RuntimeError, match="exact rank 1"):
        ttnn.lwt(to_device_1d(device, signal.reshape(2, 10)), "bior1.3")
    with pytest.raises(RuntimeError, match="DRAM-interleaved outputs"):
        ttnn.lwt(input_tensor, "bior1.3", memory_config=ttnn.L1_MEMORY_CONFIG)
    with pytest.raises(RuntimeError, match="greater than one"):
        ttnn.lwt(to_device_1d(device, torch.ones(1)), "bior1.3", boundary_mode="reflect")

    approximation, detail = ttnn.lwt(input_tensor, "bior1.3")
    wrong_detail = to_device_1d(device, torch.zeros(detail.shape[0] + 1))
    with pytest.raises(RuntimeError, match="identical shapes"):
        ttnn.ilwt(approximation, wrong_detail, "bior1.3", signal.numel())
    with pytest.raises(RuntimeError, match="greater than zero"):
        ttnn.ilwt(approximation, detail, "bior1.3", 0)

    wrong_output = to_device_1d(device, torch.empty(approximation.shape[0] + 1))
    with pytest.raises(RuntimeError, match="does not match"):
        ttnn.lwt(input_tensor, "bior1.3", output_tensors=(wrong_output, wrong_output))
    with pytest.raises(RuntimeError, match="must not alias"):
        ttnn.lwt(input_tensor, "bior1.3", output_tensors=(approximation, approximation))


def test_wavelet_2d_validation_errors(device):
    signal = torch.arange(35 * 37, dtype=torch.float32).reshape(35, 37)
    input_tensor = to_device_2d(device, signal)

    with pytest.raises(RuntimeError, match="TILE layout"):
        ttnn.lwt_2d(to_device_1d(device, signal), "bior1.3")
    with pytest.raises(RuntimeError, match="both dimensions greater than one"):
        ttnn.lwt_2d(to_device_2d(device, torch.ones(1, 8)), "bior1.3", boundary_mode="antireflect")

    sharded_memory_config = ttnn.create_sharded_memory_config(
        shape=(64, 64),
        core_grid=ttnn.CoreGrid(x=1, y=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
    )
    sharded_input = to_device_2d(
        device,
        torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64),
        sharded_memory_config,
    )
    with pytest.raises(RuntimeError, match="sharded inputs are unsupported"):
        ttnn.lwt_2d(sharded_input, "bior1.3")

    bands = ttnn.lwt_2d(input_tensor, "bior1.3")
    wrong_band = to_device_2d(device, torch.zeros(bands[0].shape[0] + 1, bands[0].shape[1]))
    with pytest.raises(RuntimeError, match="identical shapes"):
        ttnn.ilwt_2d(bands[0], wrong_band, bands[2], bands[3], "bior1.3", signal.shape)
    with pytest.raises(RuntimeError, match="must be positive"):
        ttnn.ilwt_2d(*bands, "bior1.3", (0, signal.shape[1]))
    with pytest.raises(RuntimeError, match="does not match expected shape"):
        ttnn.ilwt_2d(*bands, "bior1.3", (signal.shape[0] + 2, signal.shape[1]))
    with pytest.raises(RuntimeError, match="must not alias"):
        ttnn.lwt_2d(input_tensor, "bior1.3", output_tensors=(bands[0],) * 4)
