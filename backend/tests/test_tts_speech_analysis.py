"""Unit tests for TTS speech envelope analysis."""

from __future__ import annotations

import array
import wave
from pathlib import Path

import pytest

from dv_backend.tts_speech_analysis import measure_speech_envelope


def _write_wav(path: Path, samples: list[float], rate: int = 16000, channels: int = 1) -> None:
    pcm = array.array("h", [int(max(-1.0, min(1.0, sample)) * 32767) for sample in samples])
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def test_leading_silence_detected(tmp_path: Path) -> None:
    rate = 16000
    silence = [0.0] * rate
    speech = [0.2] * rate
    path = tmp_path / "leading.wav"
    _write_wav(path, silence + speech, rate=rate)
    env = measure_speech_envelope(path)
    assert env.leading_silence >= 0.9
    assert env.speech_duration >= 0.8


def test_trailing_silence_detected(tmp_path: Path) -> None:
    rate = 16000
    speech = [0.2] * rate
    silence = [0.0] * rate
    path = tmp_path / "trailing.wav"
    _write_wav(path, speech + silence, rate=rate)
    env = measure_speech_envelope(path)
    assert env.trailing_silence >= 0.9


def test_internal_pause_not_removed(tmp_path: Path) -> None:
    rate = 16000
    frame = int(rate * 0.02)
    speech = [0.2] * frame
    pause = [0.0] * frame * 5
    path = tmp_path / "pause.wav"
    _write_wav(path, speech + pause + speech, rate=rate)
    env = measure_speech_envelope(path)
    assert env.speech_duration >= 0.08


def test_silent_clip(tmp_path: Path) -> None:
    rate = 16000
    path = tmp_path / "silent.wav"
    _write_wav(path, [0.0] * rate * 2, rate=rate)
    env = measure_speech_envelope(path)
    assert env.speech_duration == 0.0


def test_very_short_clip(tmp_path: Path) -> None:
    rate = 16000
    path = tmp_path / "short.wav"
    _write_wav(path, [0.2] * 100, rate=rate)
    env = measure_speech_envelope(path)
    assert env.raw_wav_duration > 0


def test_stereo_supported(tmp_path: Path) -> None:
    rate = 16000
    stereo = []
    for _ in range(rate):
        stereo.extend([0.2, 0.2])
    path = tmp_path / "stereo.wav"
    _write_wav(path, stereo, rate=rate, channels=2)
    env = measure_speech_envelope(path)
    assert env.speech_duration > 0


def test_different_sample_rate(tmp_path: Path) -> None:
    rate = 22050
    path = tmp_path / "22k.wav"
    _write_wav(path, [0.2] * rate, rate=rate)
    env = measure_speech_envelope(path)
    assert env.raw_wav_duration == pytest.approx(1.0, abs=0.05)
