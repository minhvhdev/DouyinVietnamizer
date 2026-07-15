from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from dv_backend.adapters.omnivoice_worker import OmniVoiceEngine
from dv_backend.omnivoice_mps import (
    OmniVoiceDeviceError,
    inspect_module_placement,
    omnivoice_runtime_capabilities,
    validate_mps_operator_fallback_environment,
)


def _fake_torch():
    return SimpleNamespace(
        __version__="test",
        float16="float16",
        float32="float32",
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        mps=SimpleNamespace(empty_cache=lambda: None),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_built=lambda: True, is_available=lambda: True)
        ),
    )


def test_worker_loads_main_model_on_mps_with_float16_and_tokenizer_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    model = SimpleNamespace(
        device="mps:0",
        audio_tokenizer=SimpleNamespace(device="cpu"),
    )

    class _OmniVoice:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            calls.append({"model_id": model_id, **kwargs})
            return model

    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "omnivoice", SimpleNamespace(OmniVoice=_OmniVoice))

    engine = OmniVoiceEngine().get(model="test/model", device="mps")

    assert engine._model is model
    assert calls == [
        {
            "model_id": "test/model",
            "device_map": "mps",
            "dtype": "float16",
            "load_asr": False,
        }
    ]


def test_worker_rejects_invalid_mps_tokenizer_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SimpleNamespace(
        device="mps:0",
        audio_tokenizer=SimpleNamespace(device="mps:0"),
    )

    class _OmniVoice:
        @classmethod
        def from_pretrained(cls, _model_id: str, **_kwargs):
            return model

    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "omnivoice", SimpleNamespace(OmniVoice=_OmniVoice))

    with pytest.raises(RuntimeError, match="OMNIVOICE_MPS_PLACEMENT_INVALID"):
        OmniVoiceEngine().get(model="test/model", device="mps")


def test_runtime_capabilities_report_mps_without_loading_model() -> None:
    capabilities = omnivoice_runtime_capabilities(_fake_torch())

    assert capabilities["mps_built"] is True
    assert capabilities["mps_available"] is True
    assert capabilities["cuda_available"] is False


def test_operator_fallback_is_blocked_on_macos_without_separate_opt_in() -> None:
    with pytest.raises(OmniVoiceDeviceError) as exc_info:
        validate_mps_operator_fallback_environment(
            {
                "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                "DV_OMNIVOICE_ALLOW_MPS_OPERATOR_FALLBACK": "0",
            },
            platform_name="darwin",
        )

    assert exc_info.value.code == "OMNIVOICE_MPS_FALLBACK_FORBIDDEN"


def test_operator_fallback_opt_in_is_separate_from_full_cpu_fallback() -> None:
    enabled = validate_mps_operator_fallback_environment(
        {
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "DV_OMNIVOICE_ALLOW_MPS_OPERATOR_FALLBACK": "1",
            "DV_OMNIVOICE_ALLOW_CPU_FALLBACK": "0",
        },
        platform_name="darwin",
    )

    assert enabled is True


def test_tokenizer_placement_inspects_parameters_and_buffers() -> None:
    class _Tensor:
        def __init__(self, device: str, dtype: str, floating: bool) -> None:
            self.device = device
            self.dtype = dtype
            self._floating = floating

        def is_floating_point(self) -> bool:
            return self._floating

    class _Tokenizer:
        device = "cpu"

        @staticmethod
        def named_parameters(recurse: bool = True):
            assert recurse is True
            return [("weight", _Tensor("cpu", "torch.float32", True))]

        @staticmethod
        def named_buffers(recurse: bool = True):
            assert recurse is True
            return [("codes", _Tensor("cpu", "torch.int64", False))]

    placement = inspect_module_placement(_Tokenizer())

    assert placement["devices"] == ["cpu"]
    assert placement["floating_dtypes"] == ["torch.float32"]
    assert placement["tensor_count"] == 2
    assert placement["violations"] == []


def test_tokenizer_placement_detects_mps_or_float16_tensors() -> None:
    class _Tensor:
        device = "mps:0"
        dtype = "torch.float16"

        @staticmethod
        def is_floating_point() -> bool:
            return True

    class _Tokenizer:
        device = "mps:0"

        @staticmethod
        def named_parameters(recurse: bool = True):
            return [("weight", _Tensor())]

        @staticmethod
        def named_buffers(recurse: bool = True):
            return []

    placement = inspect_module_placement(_Tokenizer())

    assert "weight:device=mps:0" in placement["violations"]
    assert "weight:dtype=torch.float16" in placement["violations"]
