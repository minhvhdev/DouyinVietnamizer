"""Tests for Silero VAD adapter."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from unittest.mock import patch

from dv_backend.adapters import vad_silero


def _write_tone_wav(
    path: Path,
    *,
    duration_sec: float,
    frequency_hz: float = 440.0,
    amplitude: float = 0.5,
    sample_rate: int = 16000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            sample = int(amplitude * 32767.0 * math.sin(2.0 * math.pi * frequency_hz * index / sample_rate))
            frames.extend(struct.pack("<h", sample))
        handle.writeframes(bytes(frames))


def _write_silence_wav(path: Path, *, duration_sec: float, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\0\0" * int(duration_sec * sample_rate))


@patch("silero_vad.get_speech_timestamps")
def test_vad_step_silero_returns_speech_region_structure(mock_timestamps, tmp_path: Path) -> None:
    vad_silero.reset_vad_model_cache()
    audio = tmp_path / "audio_16k.wav"
    _write_tone_wav(audio, duration_sec=2.0)
    mock_timestamps.return_value = [{"start": 0.2, "end": 1.8}]

    regions = vad_silero.vad_step_silero(audio)

    assert regions == [{"start": 0.2, "end": 1.8}]
    mock_timestamps.assert_called_once()
    _, kwargs = mock_timestamps.call_args
    assert kwargs["sampling_rate"] == 16000
    assert kwargs["return_seconds"] is True


def test_vad_step_silero_mostly_silent_audio_has_few_regions(tmp_path: Path) -> None:
    vad_silero.reset_vad_model_cache()
    audio = tmp_path / "mostly_silent.wav"
    _write_silence_wav(audio, duration_sec=3.0)

    regions = vad_silero.vad_step_silero(audio, threshold=0.5)
    assert len(regions) <= 1
