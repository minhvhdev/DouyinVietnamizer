"""Tests for the VoxCPM2 TTS adapter, cache, and worker.

This file accumulates the tests for the VoxCPM2 migration (Tasks 1-5).
"""

import wave
from pathlib import Path

import pytest

from dv_backend.adapters.tts import (
    VOXCPM_INSTRUCT_PREFIX,
    VoxCPMTtsAdapter,
    create_tts_adapter,
    parse_voxcpm_voice,
    split_tts_text,
)
from dv_backend.adapters.voxcpm_cache import VoxCPMCache, cache_key
from dv_backend.errors import AppError
import dv_backend.voxcpm_env as voxcpm_env


# ---------------------------------------------------------------------------
# Voice parsing
# ---------------------------------------------------------------------------


def test_parse_voxcpm_voice_modes() -> None:
    assert parse_voxcpm_voice("auto") == (None, None, None)
    assert parse_voxcpm_voice(f"{VOXCPM_INSTRUCT_PREFIX}female, low pitch") == (
        None,
        None,
        "female, low pitch",
    )


def test_parse_voxcpm_voice_with_ref_audio(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    assert parse_voxcpm_voice(str(ref)) == (str(ref), None, None)


# ---------------------------------------------------------------------------
# create_tts_adapter factory
# ---------------------------------------------------------------------------


def test_create_tts_adapter_always_selects_voxcpm() -> None:
    adapter = create_tts_adapter({"tts_backend": "other", "voxcpm_device": "cuda:0"})
    assert type(adapter).__name__ == "VoxCPMTtsAdapter"


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------


def test_cache_key_is_stable() -> None:
    k1 = cache_key(voice_id="auto", text="Xin chào", model="m", num_step=10)
    k2 = cache_key(voice_id="auto", text="  Xin  CHÀO  ", model="m", num_step=10)
    assert k1 == k2


def test_cache_key_differs_on_inputs() -> None:
    base = dict(voice_id="auto", text="Xin chao", model="m", num_step=10)
    assert cache_key(**base) != cache_key(**{**base, "text": "xin chao."})
    assert cache_key(**base) != cache_key(**{**base, "num_step": 16})
    assert cache_key(**base) != cache_key(**{**base, "voice_design": "female"})
    assert cache_key(**base) != cache_key(**{**base, "cfg_value": 3.0})


def test_cache_put_and_materialize(tmp_path: Path) -> None:
    cache = VoxCPMCache(tmp_path / "cache")
    key = cache_key(voice_id="auto", text="hello", model="m", num_step=10)
    src = tmp_path / "src.wav"
    src.write_bytes(b"RIFFdata")
    cache.put(key, src)
    dest = tmp_path / "dest.wav"
    assert cache.materialize(key, dest) is True
    assert dest.read_bytes() == b"RIFFdata"


def test_cache_miss_returns_false(tmp_path: Path) -> None:
    cache = VoxCPMCache(tmp_path / "cache")
    key = cache_key(voice_id="auto", text="missing", model="m", num_step=10)
    dest = tmp_path / "dest.wav"
    assert cache.materialize(key, dest) is False
    assert not dest.exists()


# ---------------------------------------------------------------------------
# voxcpm_env resolver
# ---------------------------------------------------------------------------


def test_voxcpm_venv_root_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DV_VOXCPM_VENV", raising=False)
    root = voxcpm_env.voxcpm_venv_root()
    assert root.name == ".venv-voxcpm"


def test_voxcpm_venv_root_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DV_VOXCPM_VENV", str(tmp_path))
    assert voxcpm_env.voxcpm_venv_root() == tmp_path


def test_resolve_voxcpm_python_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DV_VOXCPM_VENV", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        voxcpm_env.resolve_voxcpm_python()


def test_is_voxcpm_available_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DV_VOXCPM_VENV", str(tmp_path))
    assert voxcpm_env.is_voxcpm_available() is False


# ---------------------------------------------------------------------------
# Worker regression: must call model.generate with VoxCPM2 kwargs
# ---------------------------------------------------------------------------


def test_worker_generate_uses_real_voxcpm_api(monkeypatch) -> None:
    """Regression: worker must call model.generate with VoxCPM2 kwargs."""
    from dv_backend.adapters import voxcpm_worker

    class FakeEngine:
        def __init__(self) -> None:
            self.calls = []

        def generate(self, text, **kwargs):
            self.calls.append((text, kwargs))
            return b"audio"

    fake_voxcpm = type("FakeVoxCPMModule", (), {"VoxCPM": type("VoxCPM", (), {})})
    monkeypatch.setitem(__import__("sys").modules, "voxcpm", fake_voxcpm)
    engine = FakeEngine()

    result = voxcpm_worker._generate(
        engine,
        "(female, low pitch)Xin chao",
        prompt_wav_path="ref.wav",
        prompt_text="hello",
        voice_design="female, low pitch",
        cfg_value=2.0,
        inference_timesteps=10,
    )

    assert result == b"audio"
    text, kwargs = engine.calls[0]
    assert text == "(female, low pitch)Xin chao"
    assert kwargs["prompt_wav_path"] == "ref.wav"
    assert kwargs["prompt_text"] == "hello"
    assert kwargs["cfg_value"] == 2.0
    assert kwargs["inference_timesteps"] == 10
