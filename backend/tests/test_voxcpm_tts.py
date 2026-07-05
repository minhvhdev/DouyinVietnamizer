"""Tests for the VoxCPM2 TTS adapter, cache, and worker.

This file accumulates the tests for the VoxCPM2 migration (Tasks 1-5).
"""

import wave
from pathlib import Path

import pytest

from dv_backend.adapters.tts import (
    VOXCPM_INSTRUCT_PREFIX,
    TtsSession,
    VoxCPMTtsAdapter,
    create_tts_adapter,
    parse_voxcpm_voice,
    split_tts_text,
)
from dv_backend.adapters.voxcpm_cache import VoxCPMCache, cache_key, reference_audio_content_hash, reference_text_hash
from dv_backend.errors import AppError
import dv_backend.adapters.voxcpm_cache as voxcpm_cache_module
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
    with wave.open(str(src), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\0\0" * 240)
    cache.put(key, src)
    dest = tmp_path / "dest.wav"
    assert cache.materialize(key, dest) is True
    assert dest.read_bytes() == src.read_bytes()


def test_cache_miss_returns_false(tmp_path: Path) -> None:
    cache = VoxCPMCache(tmp_path / "cache")
    key = cache_key(voice_id="auto", text="missing", model="m", num_step=10)
    dest = tmp_path / "dest.wav"
    assert cache.materialize(key, dest) is False
    assert not dest.exists()


# ---------------------------------------------------------------------------
# voxcpm_env / GGUF resolver
# ---------------------------------------------------------------------------


def test_voxcpm_venv_root_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DV_MODELS_DIR", raising=False)
    root = voxcpm_env.voxcpm_venv_root()
    assert root.name == "voxcpm2"


def test_voxcpm_venv_root_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DV_MODELS_DIR", str(tmp_path))
    assert voxcpm_env.voxcpm_venv_root() == tmp_path / "voxcpm2"


def test_resolve_voxcpm_python_uses_backend_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DV_VOXCPM_PYTHON", raising=False)
    monkeypatch.delenv("DV_VOXCPM_VENV", raising=False)
    assert voxcpm_env.resolve_voxcpm_python() == Path(__import__("sys").executable).resolve()


def test_resolve_voxcpm_python_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    python = tmp_path / "python.exe"
    python.write_text("stub", encoding="utf-8")
    monkeypatch.setenv("DV_VOXCPM_PYTHON", str(python))
    assert voxcpm_env.resolve_voxcpm_python() == python.resolve()


def test_is_voxcpm_available_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DV_VOXCPM_CLI", str(tmp_path / "missing-cli.exe"))
    monkeypatch.setenv("DV_VOXCPM_TTS_SERVER", str(tmp_path / "missing-server.exe"))
    monkeypatch.setattr(
        "dv_backend.voxcpm_gguf.resolve_voxcpm_gguf_paths",
        lambda model=None: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    assert voxcpm_env.is_voxcpm_available() is False


# ---------------------------------------------------------------------------
# Worker regression: must build correct voxcpm2-cli argv per clone mode
# ---------------------------------------------------------------------------


def test_build_voxcpm_cli_command_modes(tmp_path: Path) -> None:
    from dv_backend.voxcpm_gguf import build_voxcpm_cli_command

    cli = tmp_path / "voxcpm2-cli.exe"
    baselm = tmp_path / "base.gguf"
    acoustic = tmp_path / "acoustic.gguf"
    for path in (cli, baselm, acoustic):
        path.write_text("stub", encoding="utf-8")

    design = build_voxcpm_cli_command(
        cli,
        text="(female, low pitch)Xin chao",
        output_path=str(tmp_path / "out.wav"),
        baselm=baselm,
        acoustic=acoustic,
        device="cuda:0",
        cfg_value=2.0,
        inference_timesteps=10,
        mode="design",
    )
    assert design[:4] == [str(cli), "-t", "(female, low pitch)Xin chao", "-o"]
    assert "--cpu" not in design
    assert design[-2:] == [str(baselm), str(acoustic)]

    reference = build_voxcpm_cli_command(
        cli,
        text="Xin chao",
        output_path=str(tmp_path / "ref.wav"),
        baselm=baselm,
        acoustic=acoustic,
        device="cpu",
        cfg_value=2.0,
        inference_timesteps=10,
        reference_wav_path="anchor.wav",
        mode="reference",
    )
    assert "--cpu" in reference
    assert "-r" in reference
    assert "anchor.wav" in reference

    ultimate = build_voxcpm_cli_command(
        cli,
        text="Xin chao",
        output_path=str(tmp_path / "ult.wav"),
        baselm=baselm,
        acoustic=acoustic,
        device="cuda:0",
        cfg_value=2.0,
        inference_timesteps=10,
        reference_wav_path="anchor.wav",
        prompt_text="transcript",
        mode="ultimate",
    )
    assert "--prompt-wav" in ultimate
    assert "--prompt-text" in ultimate
    assert "transcript" in ultimate


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
        self._responses: dict[str, dict] = {}

    def _write_output(self, output_path: Path) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x00\x00" * 240)

    def synthesize(self, **kwargs):
        self.calls.append(kwargs)
        self._write_output(kwargs["output_path"])
        return self.next_response

    def submit_batch(self, requests):
        ids = []
        for index, request in enumerate(requests):
            self.calls.append(request)
            self._write_output(request["output_path"])
            request_id = f"req-{index}"
            ids.append(request_id)
            self._responses[request_id] = {**self.next_response, "id": request_id}
        return ids

    def wait_batch(self, request_ids):
        return [self._responses[request_id] for request_id in request_ids]

    def register_with_runner(self, runner):  # pragma: no cover - not invoked here
        return None


def test_adapter_synthesize_batch_submits_all_single_chunk_items(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = VoxCPMTtsAdapter(
        model="gguf-q8",
        device="cuda:0",
        num_steps=8,
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )
    items = [
        {"text": "Xin chao", "output_path": tmp_path / "0.wav", "voice": "auto"},
        {"text": "Tam biet", "output_path": tmp_path / "1.wav", "voice": "auto"},
    ]

    adapter.synthesize_batch(items)

    assert [call["text"] for call in client.calls] == ["Xin chao", "Tam biet"]
    assert items[0]["output_path"].is_file()
    assert items[1]["output_path"].is_file()


def test_adapter_synthesize_batch_forwards_ultimate_clone(tmp_path: Path) -> None:
    ref = tmp_path / "voice.wav"
    ref.write_bytes(b"RIFF")
    client = FakeClient()
    adapter = VoxCPMTtsAdapter(
        model="gguf-q8",
        device="cuda:0",
        num_steps=8,
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )

    adapter.synthesize_batch([
        {
            "text": "Xin chao",
            "output_path": tmp_path / "out.wav",
            "voice": str(ref),
            "clone": True,
            "clone_mode": "ultimate",
            "anchor_text": "xin chào gốc",
        }
    ])

    call = client.calls[0]
    assert call["mode"] == "ultimate"
    assert call["reference_wav_path"] == str(ref)
    assert call["prompt_wav_path"] == str(ref)
    assert call["prompt_text"] == "xin chào gốc"
    assert call["anchor_text"] == "xin chào gốc"


def test_acquire_client_keys_include_batch_settings(tmp_path: Path) -> None:
    from dv_backend.adapters.voxcpm_client import acquire_client, release_all_clients

    release_all_clients()
    try:
        first = acquire_client(data_dir=tmp_path, model="m", device="cpu", num_steps=8, max_batch=1, flush_ms=20)
        second = acquire_client(data_dir=tmp_path, model="m", device="cpu", num_steps=8, max_batch=4, flush_ms=150)
        third = acquire_client(data_dir=tmp_path, model="m", device="cpu", num_steps=8, max_batch=4, flush_ms=150)

        assert first is not second
        assert second is third
        assert second.max_batch == 4
        assert second.flush_ms == 150
    finally:
        release_all_clients()


def test_resolve_voxcpm_gguf_paths_from_models_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dv_backend.voxcpm_gguf import resolve_voxcpm_gguf_paths

    models = tmp_path / "models" / "voxcpm2"
    models.mkdir(parents=True)
    (models / "VoxCPM2-BaseLM-Q8_0.gguf").write_text("base", encoding="utf-8")
    (models / "VoxCPM2-Acoustic-F16.gguf").write_text("acoustic", encoding="utf-8")
    monkeypatch.setenv("DV_MODELS_DIR", str(tmp_path / "models"))

    baselm, acoustic = resolve_voxcpm_gguf_paths("gguf-q8")
    assert baselm.name == "VoxCPM2-BaseLM-Q8_0.gguf"
    assert acoustic.name == "VoxCPM2-Acoustic-F16.gguf"


def test_normalize_voxcpm_model_id_maps_legacy_hf_repo() -> None:
    from dv_backend.voxcpm_gguf import normalize_voxcpm_model_id

    assert normalize_voxcpm_model_id("openbmb/VoxCPM2") == "gguf-q8"


def test_adapter_routes_to_client(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = VoxCPMTtsAdapter(
        model="gguf-q8",
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


def test_cache_key_includes_reference_text_hash(tmp_path: Path) -> None:
    ref = tmp_path / "ref.txt"
    ref.write_text("Xin chào", encoding="utf-8")
    assert reference_text_hash(str(ref))
    base = dict(
        voice_id="voice",
        text="Xin chao",
        model="m",
        num_step=10,
        mode="reference",
        reference_text="Xin chào",
    )
    assert cache_key(**base) != cache_key(**{**base, "reference_text": "Tạm biệt"})


def test_reference_audio_content_hash_reuses_cached_value(tmp_path: Path, monkeypatch) -> None:
    ref = tmp_path / "voice.wav"
    ref.write_bytes(b"voice-audio")
    calls = 0
    original = voxcpm_cache_module._file_hash

    def counting_hash(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(voxcpm_cache_module, "_file_hash", counting_hash)
    first = reference_audio_content_hash(ref)
    second = reference_audio_content_hash(ref)

    assert first == second
    assert calls == 1


def test_reference_audio_content_hash_invalidates_on_change(tmp_path: Path, monkeypatch) -> None:
    ref = tmp_path / "voice.wav"
    ref.write_bytes(b"voice-audio")
    calls = 0
    original = voxcpm_cache_module._file_hash

    def counting_hash(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(voxcpm_cache_module, "_file_hash", counting_hash)
    first = reference_audio_content_hash(ref)
    ref.write_bytes(b"voice-audio-changed")
    second = reference_audio_content_hash(ref)

    assert first != second
    assert calls == 2


def test_cache_rejects_corrupt_wav_hit(tmp_path: Path) -> None:
    cache = VoxCPMCache(tmp_path / "cache")
    key = cache_key(voice_id="auto", text="hello", model="m", num_step=10)
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"not a wav")
    cache.put(key, corrupt)
    dest = tmp_path / "dest.wav"
    assert cache.materialize(key, dest) is False
    assert not dest.exists()


def test_tts_session_reuses_single_adapter_and_closes() -> None:
    class Adapter:
        def __init__(self) -> None:
            self.calls = []
            self.closed = False

        def synthesize(self, text, output_path, **kwargs):
            self.calls.append((text, output_path, kwargs))
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b"\0\0" * 240)

        def close(self):
            self.closed = True

    created = []

    def factory(settings, *, data_dir=None, runner=None):
        adapter = Adapter()
        created.append(adapter)
        return adapter

    session = TtsSession({"voxcpm_ref_audio": ""}, data_dir=Path("data"), runner=None, adapter_factory=factory)
    session.synthesize("one", Path("out1.wav"), segment={"text": "一"})
    session.synthesize("two", Path("out2.wav"), segment={"text": "二"})
    session.close()

    assert len(created) == 1
    assert len(created[0].calls) == 2
    assert created[0].closed is True


def test_tts_session_acquires_gpu_lease_and_releases_on_close() -> None:
    from dv_backend.gpu_manager import GpuModelManager

    class Adapter:
        def __init__(self) -> None:
            self.calls = []
            self.closed = False

        def synthesize(self, text, output_path, **kwargs):
            self.calls.append((text, output_path))
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b"\0\0" * 240)

        def close(self):
            self.closed = True

    def factory(settings, *, data_dir=None, runner=None):
        return Adapter()

    manager = GpuModelManager()
    session = TtsSession(
        {"voxcpm_ref_audio": "", "tts_session_reuse_enabled": True},
        data_dir=Path("data"),
        runner=None,
        adapter_factory=factory,
        gpu_manager=manager,
    )
    session.synthesize("one", Path("out1.wav"))
    session.synthesize("two", Path("out2.wav"))
    session.close()
    assert manager.lease_history[-1]["family"] == "tts"
    assert ("tts", "cuda:0") in manager._loaded
