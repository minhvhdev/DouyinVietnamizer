import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dv_backend.adapters.edge_tts import EdgeTtsAdapter, list_edge_tts_voices
from dv_backend.adapters.tts import create_tts_adapter
from dv_backend.errors import AppError


def test_create_tts_adapter_selects_edge_tts() -> None:
    adapter = create_tts_adapter({"tts_backend": "edge_tts", "edge_tts_voice": "vi-VN-NamMinhNeural"})
    assert type(adapter).__name__ == "EdgeTtsAdapter"
    assert adapter.voice == "vi-VN-NamMinhNeural"


def test_create_tts_adapter_selects_gemini_tts() -> None:
    adapter = create_tts_adapter(
        {
            "tts_backend": "gemini_tts",
            "gemini_api_keys": [{"id": "a", "key": "key-a"}],
        }
    )
    assert type(adapter).__name__ == "GeminiTtsAdapter"


def test_list_edge_tts_voices_falls_back_without_network(monkeypatch) -> None:
    def fake_run_async(coro):
        coro.close()
        raise RuntimeError("offline")

    monkeypatch.setattr("dv_backend.adapters.edge_tts._run_async", fake_run_async)
    voices = list_edge_tts_voices(refresh=True)
    assert any(voice["id"] == "vi-VN-HoaiMyNeural" for voice in voices)


@patch("dv_backend.adapters.edge_tts._mp3_to_wav")
@patch("dv_backend.adapters.edge_tts._run_async")
def test_edge_tts_adapter_writes_wav(mock_run_async: MagicMock, mock_mp3_to_wav: MagicMock, tmp_path: Path) -> None:
    output = tmp_path / "edge.wav"

    def fake_run_async(coro):
        coro.close()
        return None

    def fake_mp3_to_wav(_mp3_path: Path, wav_path: Path) -> None:
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x00\x00" * 1200)

    mock_run_async.side_effect = fake_run_async
    mock_mp3_to_wav.side_effect = fake_mp3_to_wav

    EdgeTtsAdapter(voice="vi-VN-HoaiMyNeural").synthesize("Xin chao", output, voice="vi-VN-HoaiMyNeural")

    with wave.open(str(output), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 24000
        assert wav.getnframes() > 0


def test_edge_tts_adapter_rejects_empty_text(tmp_path: Path) -> None:
    with pytest.raises(AppError) as error:
        EdgeTtsAdapter().synthesize("   ", tmp_path / "empty.wav", voice="vi-VN-HoaiMyNeural")
    assert error.value.info.code == "EMPTY_TTS_TEXT"
