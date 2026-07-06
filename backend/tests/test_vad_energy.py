"""Tests for vocal-energy false-positive filtering."""

from __future__ import annotations

import array
import wave
from pathlib import Path

from dv_backend.adapters.vad_energy import (
    filter_low_vocal_energy_segments,
    is_likely_low_vocal_energy,
    region_rms,
)


def _write_tone(path: Path, *, amplitude: float, duration: float = 1.0, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(duration * sample_rate)
    sample_value = max(-32767, min(32767, int(32767 * amplitude)))
    frame = int(sample_value).to_bytes(2, byteorder="little", signed=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frame * frame_count)


def test_region_rms_reflects_amplitude(tmp_path: Path) -> None:
    quiet = tmp_path / "quiet.wav"
    loud = tmp_path / "loud.wav"
    _write_tone(quiet, amplitude=0.01)
    _write_tone(loud, amplitude=0.2)
    assert region_rms(loud, 0.0, 1.0) > region_rms(quiet, 0.0, 1.0) * 5


def test_low_vocal_energy_detects_music_heavy_region(tmp_path: Path) -> None:
    vocals = tmp_path / "vocals.wav"
    bgm = tmp_path / "bgm.wav"
    _write_tone(vocals, amplitude=0.01)
    _write_tone(bgm, amplitude=0.25)
    assert is_likely_low_vocal_energy(
        vocals_path=vocals,
        bgm_path=bgm,
        start_sec=0.0,
        end_sec=1.0,
        min_vocal_ratio=1.15,
    )


def test_filter_low_vocal_energy_segments_rejects_music_heavy(tmp_path: Path) -> None:
    vocals = tmp_path / "vocals.wav"
    bgm = tmp_path / "bgm.wav"
    sample_rate = 16000

    def _segment_frames(amplitude: float) -> array.array:
        value = int(32767 * amplitude)
        return array.array("h", [value] * sample_rate)

    vocals.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(vocals), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((_segment_frames(0.01) + _segment_frames(0.3)).tobytes())
    with wave.open(str(bgm), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((_segment_frames(0.25) + _segment_frames(0.05)).tobytes())

    segments = [
        {"start": 0.0, "end": 1.0, "text": "nhạc"},
        {"start": 1.0, "end": 2.0, "text": "speech"},
    ]
    kept, rejected = filter_low_vocal_energy_segments(
        segments,
        vocals_path=vocals,
        bgm_path=bgm,
        enabled=True,
    )
    assert len(kept) == 1
    assert len(rejected) == 1
    assert kept[0]["text"] == "speech"
    assert rejected[0]["vad_false_positive_reason"] == "low_vocal_energy"
