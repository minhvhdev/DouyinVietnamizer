import wave
from pathlib import Path

import pytest

from dv_backend.adapters.omnivoice_cache import OmniVoiceCache, cache_key
from dv_backend.adapters.tts import (
    OMNIVOICE_INSTRUCT_PREFIX,
    OmniVoiceTtsAdapter,
    create_tts_adapter,
    parse_omnivoice_voice,
    split_tts_text,
)
from dv_backend.errors import AppError


# ---------------------------------------------------------------------------
# Voice parsing (unchanged behaviour)
# ---------------------------------------------------------------------------


def test_parse_omnivoice_voice_modes() -> None:
    assert parse_omnivoice_voice("auto") == (None, None, None)
    assert parse_omnivoice_voice(f"{OMNIVOICE_INSTRUCT_PREFIX}female, low pitch") == (
        None,
        "female, low pitch",
        None,
    )


def test_parse_omnivoice_voice_with_ref_audio(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    assert parse_omnivoice_voice(str(ref)) == (str(ref), None, None)


# ---------------------------------------------------------------------------
# create_tts_adapter factory
# ---------------------------------------------------------------------------


def test_create_tts_adapter_always_selects_omnivoice() -> None:
    adapter = create_tts_adapter({"tts_backend": "vieneu", "omnivoice_device": "cuda:0"})
    assert type(adapter).__name__ == "OmniVoiceTtsAdapter"


def test_legacy_omnivoice_python_kwarg_rejected(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        OmniVoiceTtsAdapter(omnivoice_python=tmp_path / "python.exe")


# ---------------------------------------------------------------------------
# OmniVoiceTtsAdapter with an injected fake client
# ---------------------------------------------------------------------------


class FakeClient:
    """Drop-in replacement for OmniVoiceWorkerClient used in tests."""

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
    adapter = OmniVoiceTtsAdapter(
        model="k2-fsa/OmniVoice",
        device="cuda:0",
        num_steps=16,
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )
    output = tmp_path / "out.wav"
    adapter.synthesize("Xin chao", output, voice="auto")
    assert client.calls, "client.synthesize was not called"
    call = client.calls[0]
    assert call["text"] == "Xin chao"
    assert call["ref_audio"] is None
    assert call["instruct"] is None
    assert output.is_file()


def test_adapter_clone_uses_ref_audio_and_ref_text(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = OmniVoiceTtsAdapter(
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
    assert call["ref_audio"] == str(ref_audio)
    assert call["ref_text"] == "hello"


def test_adapter_chunks_long_text(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = OmniVoiceTtsAdapter(
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
        # The real worker writes the file; emulate it with a valid WAV so
        # the post-chunk concat step can read it.
        out = Path(call["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x00\x00" * 240)


def test_adapter_rejects_empty_text(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = OmniVoiceTtsAdapter(
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
                "code": "OMNIVOICE_GPU_OOM",
                "message": "Out of memory",
                "retryable": True,
            }

    client = FailingClient()
    adapter = OmniVoiceTtsAdapter(
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )
    with pytest.raises(AppError) as exc:
        adapter.synthesize("Xin chao", tmp_path / "out.wav", voice="auto")
    assert exc.value.info.code == "OMNIVOICE_GPU_OOM"
    assert exc.value.info.retryable is True


def test_adapter_uses_cache(tmp_path: Path) -> None:
    """A second synthesize call with the same (voice, text) must be a cache hit."""
    client = FakeClient()
    cache = OmniVoiceCache(tmp_path / "cache")
    adapter = OmniVoiceTtsAdapter(
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

    # Second call: cache hit, no new client call.
    output2 = tmp_path / "out2.wav"
    adapter.synthesize("Xin chao", output2, voice="auto")
    assert len(client.calls) == calls_after_first
    assert output2.read_bytes() == output.read_bytes()


def test_adapter_cache_disabled_always_calls_client(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = OmniVoiceTtsAdapter(
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )
    output = tmp_path / "out.wav"
    adapter.synthesize("Xin chao", output, voice="auto")
    output2 = tmp_path / "out2.wav"
    adapter.synthesize("Xin chao", output2, voice="auto")
    assert len(client.calls) == 2


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------


def test_cache_key_is_stable() -> None:
    k1 = cache_key(voice_id="auto", text="Xin chào", model="m", num_step=32)
    k2 = cache_key(voice_id="auto", text="  Xin  CHÀO  ", model="m", num_step=32)
    assert k1 == k2


def test_cache_key_differs_on_inputs() -> None:
    base = dict(voice_id="auto", text="Xin chao", model="m", num_step=32)
    assert cache_key(**base) != cache_key(**{**base, "text": "xin chao."})
    assert cache_key(**base) != cache_key(**{**base, "num_step": 16})
    assert cache_key(**base) != cache_key(**{**base, "instruct": "female"})


def test_cache_put_and_materialize(tmp_path: Path) -> None:
    cache = OmniVoiceCache(tmp_path / "cache")
    key = cache_key(voice_id="auto", text="hello", model="m", num_step=32)
    src = tmp_path / "src.wav"
    src.write_bytes(b"RIFFdata")
    cache.put(key, src)
    dest = tmp_path / "dest.wav"
    assert cache.materialize(key, dest) is True
    assert dest.read_bytes() == b"RIFFdata"


def test_cache_miss_returns_false(tmp_path: Path) -> None:
    cache = OmniVoiceCache(tmp_path / "cache")
    key = cache_key(voice_id="auto", text="missing", model="m", num_step=32)
    dest = tmp_path / "dest.wav"
    assert cache.materialize(key, dest) is False
    assert not dest.exists()


def test_worker_generate_uses_real_omnivoice_api(monkeypatch):
    """Regression: worker must call engine.generate, not OmniVoice(model=...)."""
    from dv_backend.adapters import omnivoice_worker

    class FakeGenerationConfig:
        def __init__(self, num_step: int) -> None:
            self.num_step = num_step

    class FakeEngine:
        def __init__(self) -> None:
            self.calls = []

        def generate(self, texts, **kwargs):
            self.calls.append((texts, kwargs))
            return [b"audio"]

    fake_omnivoice = type("FakeOmniVoiceModule", (), {"OmniVoiceGenerationConfig": FakeGenerationConfig})
    monkeypatch.setitem(__import__("sys").modules, "omnivoice", fake_omnivoice)
    engine = FakeEngine()

    result = omnivoice_worker._generate(
        engine,
        ["Xin chao"],
        ref_audio="ref.wav",
        ref_text="hello",
        instruct=None,
        num_step=16,
    )

    assert result == [b"audio"]
    texts, kwargs = engine.calls[0]
    assert texts == ["Xin chao"]
    assert kwargs["ref_audio"] == "ref.wav"
    assert kwargs["ref_text"] == "hello"
    assert kwargs["generation_config"].num_step == 16
