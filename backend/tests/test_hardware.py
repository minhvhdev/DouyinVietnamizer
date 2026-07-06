import sys
from unittest import mock

import pytest


def _import_hardware():
    """Re-import hardware module after mocking torch."""
    import importlib
    from dv_backend import hardware
    importlib.reload(hardware)
    return hardware


def test_detect_vulkan_false_on_non_windows():
    if sys.platform == "win32":
        pytest.skip("non-Windows specific test")
    hardware = _import_hardware()
    assert hardware.detect_vulkan() is False


def test_detect_cpu_avx2_true_on_non_windows():
    if sys.platform == "win32":
        pytest.skip("non-Windows specific test")
    hardware = _import_hardware()
    assert hardware.detect_cpu_avx2() is True


def test_detect_espeak_false_on_non_windows():
    if sys.platform == "win32":
        pytest.skip("non-Windows specific test")
    hardware = _import_hardware()
    assert hardware.detect_espeak() is False


def test_get_best_device_returns_cpu_when_no_accelerator(monkeypatch):
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    hardware = _import_hardware()
    assert hardware.get_best_device() == "cpu"
    assert hardware.default_inference_device() == "cpu"


def test_detect_cuda_reports_mps_on_macos(monkeypatch):
    if sys.platform != "darwin":
        pytest.skip("macOS-specific test")
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = True
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    hardware = _import_hardware()
    assert hardware.detect_cuda() is True
    assert hardware.accelerator_available() is True
    assert hardware.resolve_inference_device("cuda:0") == "mps"
    assert hardware.resolve_inference_device("mps") == "mps"


def test_resolve_inference_device_prefers_cuda_when_available(monkeypatch):
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.backends.mps.is_available.return_value = True
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    hardware = _import_hardware()
    assert hardware.resolve_inference_device("cuda:0") == "cuda:0"
    assert hardware.resolve_inference_device("") == "cuda:0"


def test_inference_dtype_uses_float16_on_mps(monkeypatch):
    fake_torch = mock.MagicMock()
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    hardware = _import_hardware()
    assert hardware.inference_dtype_for_device("mps") == "float16"
    assert hardware.inference_dtype_for_device("cuda:0") == "bfloat16"


def test_detect_cuda_falls_back_to_false_when_torch_missing(monkeypatch):
    hardware = _import_hardware()
    monkeypatch.setitem(sys.modules, "torch", None)
    assert hardware.detect_cuda() is False
