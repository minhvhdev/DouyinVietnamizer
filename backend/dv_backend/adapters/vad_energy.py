"""Energy-based helpers for rejecting likely music-only VAD/ASR regions."""

from __future__ import annotations

import array
import wave
from pathlib import Path
from typing import Any


def region_rms(path: str | Path, start_sec: float, end_sec: float) -> float:
    """Root-mean-square amplitude for a mono/stereo PCM WAV time slice."""
    start = max(0.0, float(start_sec))
    end = max(start, float(end_sec))
    if end <= start:
        return 0.0

    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            start_frame = int(start * sample_rate)
            end_frame = min(handle.getnframes(), int(end * sample_rate))
            if end_frame <= start_frame:
                return 0.0
            handle.setpos(start_frame)
            frames = handle.readframes(end_frame - start_frame)
    except Exception:
        return 0.0

    samples = array.array("h")
    samples.frombytes(frames)
    if not samples:
        return 0.0

    if channels > 1:
        mono = [float(samples[index]) / 32768.0 for index in range(0, len(samples), channels)]
    else:
        mono = [float(sample) / 32768.0 for sample in samples]
    if not mono:
        return 0.0
    return (sum(value * value for value in mono) / len(mono)) ** 0.5


def vocal_energy_ratio(
    vocals_path: str | Path,
    bgm_path: str | Path,
    *,
    start_sec: float,
    end_sec: float,
) -> float:
    vocals_rms = region_rms(vocals_path, start_sec, end_sec)
    bgm_rms = region_rms(bgm_path, start_sec, end_sec)
    return vocals_rms / max(bgm_rms, 1e-6)


def is_likely_low_vocal_energy(
    *,
    vocals_path: str | Path,
    bgm_path: str | Path,
    start_sec: float,
    end_sec: float,
    min_vocal_ratio: float = 1.15,
    min_vocals_rms: float = 0.002,
) -> bool:
    vocals_rms = region_rms(vocals_path, start_sec, end_sec)
    if vocals_rms < min_vocals_rms:
        return True
    ratio = vocal_energy_ratio(
        vocals_path,
        bgm_path,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    return ratio < min_vocal_ratio


def filter_low_vocal_energy_segments(
    segments: list[dict[str, Any]],
    *,
    vocals_path: str | Path | None,
    bgm_path: str | Path | None,
    enabled: bool = True,
    min_vocal_ratio: float = 1.15,
    min_vocals_rms: float = 0.002,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not enabled or not segments or not vocals_path or not bgm_path:
        return list(segments), []

    vocals_file = Path(vocals_path)
    bgm_file = Path(bgm_path)
    if not vocals_file.is_file() or not bgm_file.is_file():
        return list(segments), []

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for segment in sorted(segments, key=lambda item: float(item.get("start", 0.0) or 0.0)):
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", start) or start)
        if is_likely_low_vocal_energy(
            vocals_path=vocals_file,
            bgm_path=bgm_file,
            start_sec=start,
            end_sec=end,
            min_vocal_ratio=min_vocal_ratio,
            min_vocals_rms=min_vocals_rms,
        ):
            rejected.append(
                {
                    **segment,
                    "vad_false_positive_reason": "low_vocal_energy",
                }
            )
            continue
        kept.append(segment)
    return kept, rejected
