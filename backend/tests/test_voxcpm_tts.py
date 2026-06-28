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


# ---------------------------------------------------------------------------
# VoxCPMTtsAdapter with an injected fake client
# ---------------------------------------------------------------------------


class FakeClient:
    """Drop-in replacement for VoxCPMWorkerClient used in tests."""

    def __init__(self, *, model: str = "", device: str = "", num_steps: int = 0) -> None:
        self.calls: list[dict] = []
        self.requested_model = model
        self.requested_device = device
        self.requested_num_steps = num_steps
        self.next_response: dict = {"ok": True, "duration_sec": 1.0, "sample_rate": 24000}

    def synthesize(self, **kwargs):
        self.calls.append(kwargs)
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x00\x00" * 240)
        return self.next_response

    def register_with_runner(self, runner):  # pragma: no cover - not invoked here
        return None


def test_adapter_routes_to_client(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = VoxCPMTtsAdapter(
        model="openbmb/VoxCPM2",
        device="cuda:0",
        num_steps=10,
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )
    output = tmp_path / "out.wav"
    adapter.synthesize("Xin chao", output, voice="auto")
    assert client.calls, "client.synthesize was not called"
    call = client.calls[0]
    assert call["text"] == "Xin chao"
    assert call["prompt_wav_path"] is None
    assert call["voice_design"] is None
    assert output.is_file()


def test_adapter_clone_uses_ref_audio_and_ref_text(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = VoxCPMTtsAdapter(
        device="cuda:0",
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"RIFF")
    output = tmp_path / "out.wav"
    adapter.synthesize("Xin chao", output, voice=str(ref_audio), ref_text="hello")

    call = client.calls[0]
    assert call["prompt_wav_path"] == str(ref_audio)
    assert call["prompt_text"] == "hello"


def test_adapter_instruct_prefixes_text(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = VoxCPMTtsAdapter(
        device="cuda:0",
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )
    output = tmp_path / "out.wav"
    adapter.synthesize("Xin chao", output, voice=f"{VOXCPM_INSTRUCT_PREFIX}female, low pitch")
    call = client.calls[0]
    assert call["text"] == "(female, low pitch)Xin chao"
    assert call["voice_design"] == "female, low pitch"


def test_adapter_chunks_long_text(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = VoxCPMTtsAdapter(
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )
    output = tmp_path / "out.wav"
    long_text = " ".join(["cau"] * 200)
    adapter.synthesize(long_text, output, voice="auto")
    assert client.calls
    expected_chunks = len(split_tts_text(long_text))
    assert len(client.calls) == expected_chunks
    for call in client.calls:
        out = Path(call["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x00\x00" * 240)


def test_adapter_rejects_empty_text(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = VoxCPMTtsAdapter(
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )
    with pytest.raises(AppError) as exc:
        adapter.synthesize("   ", tmp_path / "out.wav", voice="auto")
    assert exc.value.info.code == "EMPTY_TTS_TEXT"


def test_adapter_propagates_client_error(tmp_path: Path) -> None:
    class FailingClient(FakeClient):
        def synthesize(self, **kwargs):  # type: ignore[override]
            super().synthesize(**kwargs)
            return {
                "ok": False,
                "code": "VOXCPM_GPU_OOM",
                "message": "Out of memory",
                "retryable": True,
            }

    client = FailingClient()
    adapter = VoxCPMTtsAdapter(
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )
    with pytest.raises(AppError) as exc:
        adapter.synthesize("Xin chao", tmp_path / "out.wav", voice="auto")
    assert exc.value.info.code == "VOXCPM_GPU_OOM"
    assert exc.value.info.retryable is True


def test_adapter_uses_cache(tmp_path: Path) -> None:
    client = FakeClient()
    cache = VoxCPMCache(tmp_path / "cache")
    adapter = VoxCPMTtsAdapter(
        data_dir=tmp_path,
        enable_cache=True,
        _client=client,
        _cache=cache,
    )
    output = tmp_path / "out.wav"
    adapter.synthesize("Xin chao", output, voice="auto")
    calls_after_first = len(client.calls)
    assert calls_after_first == 1
    assert output.is_file()

    output2 = tmp_path / "out2.wav"
    adapter.synthesize("Xin chao", output2, voice="auto")
    assert len(client.calls) == calls_after_first
    assert output2.read_bytes() == output.read_bytes()


def test_adapter_cache_disabled_always_calls_client(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = VoxCPMTtsAdapter(
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )
    output = tmp_path / "out.wav"
    adapter.synthesize("Xin chao", output, voice="auto")
    output2 = tmp_path / "out2.wav"
    adapter.synthesize("Xin chao", output2, voice="auto")
    assert len(client.calls) == 2
