import base64
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from dv_backend.adapters.google_tts import (
    DEFAULT_GOOGLE_TTS_VOICE,
    GoogleTtsAdapter,
    _synthesize_cloud_chunk,
)
from dv_backend.adapters.tts import create_tts_adapter
from dv_backend.errors import AppError

_PCM = b"\x01\x00" * 1200


def test_create_tts_adapter_selects_google_tts() -> None:
    adapter = create_tts_adapter(
        {
            "tts_backend": "google_tts",
            "google_tts_api_key": "test-key",
            "google_tts_voice": "vi-VN-Standard-C",
            "google_tts_speaking_rate": 1.1,
        }
    )
    assert type(adapter).__name__ == "GoogleTtsAdapter"
    assert adapter.voice == "vi-VN-Standard-C"
    assert adapter.api_key == "test-key"
    assert adapter.speaking_rate == 1.1


def test_google_tts_requires_api_key(tmp_path: Path) -> None:
    with pytest.raises(AppError) as error:
        GoogleTtsAdapter(api_key="").synthesize("Xin chao", tmp_path / "out.wav", voice=DEFAULT_GOOGLE_TTS_VOICE)
    assert error.value.info.code == "MISSING_GOOGLE_TTS_API_KEY"


@patch("dv_backend.adapters.google_tts._cloud_tts_request")
def test_google_tts_adapter_writes_wav(mock_request, tmp_path: Path) -> None:
    mock_request.return_value = {
        "audioContent": base64.b64encode(_PCM).decode("ascii"),
    }
    output = tmp_path / "google.wav"
    GoogleTtsAdapter(api_key="test-key", voice="vi-VN-Wavenet-A").synthesize(
        "Xin chao",
        output,
        voice="vi-VN-Wavenet-A",
    )
    with wave.open(str(output), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 24000
        assert wav.getnframes() > 0


@patch("dv_backend.adapters.google_tts._cloud_tts_request")
def test_synthesize_cloud_chunk_sends_voice_name(mock_request, tmp_path: Path) -> None:
    mock_request.return_value = {"audioContent": base64.b64encode(_PCM).decode("ascii")}
    wav_path = tmp_path / "chunk.wav"
    _synthesize_cloud_chunk(
        "Xin chao",
        "vi-VN-Standard-B",
        wav_path,
        api_key="test-key",
        speaking_rate=1.0,
    )
    payload = mock_request.call_args.args[1]
    assert payload["voice"]["name"] == "vi-VN-Standard-B"
    assert payload["voice"]["languageCode"] == "vi-VN"


def test_google_tts_adapter_rejects_empty_text(tmp_path: Path) -> None:
    with pytest.raises(AppError) as error:
        GoogleTtsAdapter(api_key="test-key").synthesize("   ", tmp_path / "empty.wav")
    assert error.value.info.code == "EMPTY_TTS_TEXT"
