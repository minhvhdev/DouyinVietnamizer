"""Validate TTS WAV candidates before promoting them to canonical paths."""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

from .tts_speech_analysis import measure_speech_envelope


@dataclass(frozen=True)
class WavValidationResult:
    ok: bool
    reason: str | None
    duration: float = 0.0
    sample_count: int = 0
    frame_rate: int = 0
    voiced_ratio: float = 0.0
    peak_abs: float = 0.0


def validate_canonical_wav_candidate(
    path: Path,
    *,
    min_duration: float = 0.08,
    max_duration: float = 180.0,
    min_voiced_ratio: float = 0.05,
    min_peak_abs: float = 0.002,
    max_samples_to_scan: int = 5_000_000,
) -> WavValidationResult:
    """Full-decode WAV validation for atomic promote (temp → canonical).

    Checks decode, sample integrity, duration bounds, non-silence / voiced ratio.
    Does not run ASR fidelity checks (those belong downstream QC).
    """
    if not path.is_file():
        return WavValidationResult(False, "missing_file")
    if path.stat().st_size < 44:
        return WavValidationResult(False, "too_small")

    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            rate = handle.getframerate()
            nframes = handle.getnframes()
            if channels < 1 or sample_width < 1 or rate <= 0 or nframes <= 0:
                return WavValidationResult(False, "invalid_header", frame_rate=rate, sample_count=nframes)
            # Full decode: read all frames (wave validates RIFF structure).
            frames = handle.readframes(nframes)
    except Exception:
        return WavValidationResult(False, "decode_failed")

    expected = nframes * channels * sample_width
    if len(frames) < expected:
        return WavValidationResult(
            False,
            "truncated_frames",
            duration=nframes / float(rate) if rate else 0.0,
            sample_count=nframes,
            frame_rate=rate,
        )

    duration = nframes / float(rate)
    if duration < min_duration:
        return WavValidationResult(False, "duration_too_short", duration=duration, sample_count=nframes, frame_rate=rate)
    if duration > max_duration:
        return WavValidationResult(False, "duration_too_long", duration=duration, sample_count=nframes, frame_rate=rate)

    # Spot-check sample integrity / peak without materializing huge float lists twice.
    import array

    if sample_width == 2:
        samples = array.array("h")
        samples.frombytes(frames)
        scale = 32768.0
    elif sample_width == 4:
        samples = array.array("i")
        samples.frombytes(frames)
        scale = 2147483648.0
    else:
        samples = array.array("B")
        samples.frombytes(frames)
        scale = 128.0

    if not samples:
        return WavValidationResult(False, "empty_samples", duration=duration, sample_count=0, frame_rate=rate)

    step = max(1, len(samples) // max_samples_to_scan)
    peak = 0.0
    for index in range(0, len(samples), step):
        value = float(samples[index]) / scale
        if math.isnan(value) or math.isinf(value):
            return WavValidationResult(
                False,
                "corrupt_samples",
                duration=duration,
                sample_count=nframes,
                frame_rate=rate,
            )
        abs_value = abs(value)
        if abs_value > peak:
            peak = abs_value

    if peak < min_peak_abs:
        return WavValidationResult(
            False,
            "near_silent_peak",
            duration=duration,
            sample_count=nframes,
            frame_rate=rate,
            peak_abs=peak,
        )

    try:
        envelope = measure_speech_envelope(path)
    except Exception:
        return WavValidationResult(
            False,
            "speech_measure_failed",
            duration=duration,
            sample_count=nframes,
            frame_rate=rate,
            peak_abs=peak,
        )

    voiced_ratio = envelope.speech_duration / max(duration, 0.01)
    if voiced_ratio < min_voiced_ratio:
        return WavValidationResult(
            False,
            "voiced_ratio_too_low",
            duration=duration,
            sample_count=nframes,
            frame_rate=rate,
            voiced_ratio=round(voiced_ratio, 4),
            peak_abs=peak,
        )

    return WavValidationResult(
        True,
        None,
        duration=round(duration, 4),
        sample_count=nframes,
        frame_rate=rate,
        voiced_ratio=round(voiced_ratio, 4),
        peak_abs=round(peak, 5),
    )
