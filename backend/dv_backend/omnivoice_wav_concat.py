"""Concatenate OmniVoice chunk WAVs with configurable inter-chunk pauses."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .adapters.tts import _read_wav_mono_float


def concat_omnivoice_chunks(
    chunk_paths: list[Path],
    *,
    pause_ms_list: list[int],
    output_path: Path,
    trailing_silence_ms: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Join chunk WAVs mono-normalized; return per-chunk timeline metadata."""
    if not chunk_paths:
        raise ValueError("No chunk WAV files to concatenate.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(chunk_paths) == 1:
        import shutil

        shutil.copy2(chunk_paths[0], output_path)
        duration = _wav_duration(chunk_paths[0])
        return [
            {
                "chunk_index": 0,
                "audio_start": 0.0,
                "audio_end": round(duration, 4),
                "concat_pause_ms": 0,
            }
        ]

    arrays: list[np.ndarray] = []
    sample_rate: int | None = None
    for path in chunk_paths:
        samples, rate, _params = _read_wav_mono_float(path)
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            samples = _resample_linear(samples, rate, sample_rate)
        arrays.append(samples.astype(np.float32))

    assert sample_rate is not None
    timeline: list[dict[str, Any]] = []
    merged = arrays[0]
    cursor_sec = len(arrays[0]) / sample_rate
    timeline.append(
        {
            "chunk_index": 0,
            "audio_start": 0.0,
            "audio_end": round(cursor_sec, 4),
            "concat_pause_ms": 0,
        }
    )
    trailing = trailing_silence_ms or [0] * len(chunk_paths)

    for index in range(1, len(arrays)):
        pause_ms = pause_ms_list[index - 1] if index - 1 < len(pause_ms_list) else 0
        natural_trailing_ms = trailing[index - 1] if index - 1 < len(trailing) else 0
        effective_pause_ms = max(0, pause_ms - min(pause_ms, natural_trailing_ms // 2))
        gap_samples = int(sample_rate * effective_pause_ms / 1000)
        if gap_samples > 0:
            merged = np.concatenate([merged, np.zeros(gap_samples, dtype=np.float32)])
        start_sec = len(merged) / sample_rate
        merged = np.concatenate([merged, arrays[index]])
        end_sec = len(merged) / sample_rate
        timeline.append(
            {
                "chunk_index": index,
                "audio_start": round(start_sec, 4),
                "audio_end": round(end_sec, 4),
                "concat_pause_ms": effective_pause_ms,
            }
        )

    pcm16 = np.clip(merged, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    import wave

    with wave.open(str(output_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm16.tobytes())
    return timeline


def _wav_duration(path: Path) -> float:
    import wave

    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or len(samples) == 0:
        return samples
    duration = len(samples) / source_rate
    target_length = max(1, int(duration * target_rate))
    source_times = np.linspace(0.0, duration, num=len(samples), endpoint=False)
    target_times = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(target_times, source_times, samples).astype(np.float32)
