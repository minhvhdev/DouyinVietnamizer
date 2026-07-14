"""Measure outer silence and speech duration inside TTS WAV clips."""

from __future__ import annotations

import array
import math
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpeechEnvelope:
    raw_wav_duration: float
    leading_silence: float
    trailing_silence: float
    speech_duration: float
    speech_start: float
    speech_end: float
    internal_pause_duration: float = 0.0
    active_speech_duration: float = 0.0
    measurement_confidence: float = 1.0


def _read_mono_samples(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if not frames:
        return [], rate

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
        return [], rate

    if channels > 1:
        mono = [float(samples[index]) / scale for index in range(0, len(samples), channels)]
    else:
        mono = [float(sample) / scale for sample in samples]
    return mono, rate


def _frame_rms(frame: list[float]) -> float:
    if not frame:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in frame) / len(frame))


def _estimate_noise_floor(mono: list[float], frame_size: int) -> float:
    if not mono:
        return 0.012
    rms_values = [
        _frame_rms(mono[offset : offset + frame_size])
        for offset in range(0, len(mono), frame_size)
    ]
    if not rms_values:
        return 0.012
    sorted_rms = sorted(rms_values)
    quiet = sorted_rms[: max(1, len(sorted_rms) // 5)]
    floor = statistics_median(quiet) * 2.5
    return max(0.008, min(0.05, floor))


def statistics_median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def measure_speech_envelope(
    wav_path: Path,
    *,
    noise_floor: float | None = None,
    frame_ms: int = 20,
    min_speech_ms: int = 40,
) -> SpeechEnvelope:
    mono, rate = _read_mono_samples(wav_path)
    if not mono or rate <= 0:
        return SpeechEnvelope(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    frame_size = max(1, int(rate * frame_ms / 1000))
    min_speech_frames = max(1, int(min_speech_ms / frame_ms))
    raw_duration = len(mono) / rate
    threshold = noise_floor if noise_floor is not None else _estimate_noise_floor(mono, frame_size)

    speech_flags: list[bool] = []
    for offset in range(0, len(mono), frame_size):
        frame = mono[offset : offset + frame_size]
        speech_flags.append(_frame_rms(frame) >= threshold)

    if not any(speech_flags):
        return SpeechEnvelope(
            round(raw_duration, 4),
            round(raw_duration, 4),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.2,
        )

    first = next(index for index, flag in enumerate(speech_flags) if flag)
    last = len(speech_flags) - 1 - next(
        index for index, flag in enumerate(reversed(speech_flags)) if flag
    )

    run = 0
    start_frame = first
    for index in range(first, last + 1):
        if speech_flags[index]:
            run += 1
            if run >= min_speech_frames:
                start_frame = index - run + 1
                break
        else:
            run = 0

    run = 0
    end_frame = last
    for index in range(last, start_frame - 1, -1):
        if speech_flags[index]:
            run += 1
            if run >= min_speech_frames:
                end_frame = index
                break
        else:
            run = 0

    speech_start = (start_frame * frame_size) / rate
    speech_end = min(raw_duration, ((end_frame + 1) * frame_size) / rate)
    speech_duration = max(0.0, speech_end - speech_start)
    leading = max(0.0, speech_start)
    trailing = max(0.0, raw_duration - speech_end)

    internal_pause = 0.0
    active_speech = 0.0
    in_pause = False
    pause_start = 0.0
    for index in range(start_frame, end_frame + 1):
        frame_start = (index * frame_size) / rate
        frame_end = min(raw_duration, ((index + 1) * frame_size) / rate)
        if speech_flags[index]:
            if in_pause:
                internal_pause += max(0.0, frame_start - pause_start)
                in_pause = False
            active_speech += max(0.0, frame_end - frame_start)
        elif not in_pause:
            in_pause = True
            pause_start = frame_start

    confidence = 1.0
    if raw_duration < 0.2:
        confidence = 0.35
    elif speech_duration / max(raw_duration, 0.01) < 0.15:
        confidence = 0.45
    elif internal_pause > speech_duration * 0.6:
        confidence = 0.7

    return SpeechEnvelope(
        round(raw_duration, 4),
        round(leading, 4),
        round(trailing, 4),
        round(speech_duration, 4),
        round(speech_start, 4),
        round(speech_end, 4),
        round(internal_pause, 4),
        round(active_speech, 4),
        round(confidence, 2),
    )


def attach_speech_metrics(segment: dict, envelope: SpeechEnvelope) -> None:
    segment["tts_speech_duration"] = envelope.speech_duration
    segment["tts_leading_silence"] = envelope.leading_silence
    segment["tts_trailing_silence"] = envelope.trailing_silence
    segment["tts_raw_wav_duration"] = envelope.raw_wav_duration
    segment["tts_internal_pause_duration"] = envelope.internal_pause_duration
    segment["tts_active_speech_duration"] = envelope.active_speech_duration
    segment["tts_speech_measurement_confidence"] = envelope.measurement_confidence
