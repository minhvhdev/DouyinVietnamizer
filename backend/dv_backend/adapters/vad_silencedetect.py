"""FFmpeg silencedetect-based voice activity detection (legacy rollback path)."""

from __future__ import annotations

import re
from pathlib import Path


def parse_silencedetect_stderr(stderr: str, total_duration: float) -> list[dict[str, float]]:
    """Invert FFmpeg silencedetect markers into speech regions."""
    starts = [float(value) for value in re.findall(r"silence_start:\s*(\d+\.?\d*)", stderr)]
    ends = [float(value) for value in re.findall(r"silence_end:\s*(\d+\.?\d*)", stderr)]

    silences: list[tuple[float, float]] = []
    for index in range(min(len(starts), len(ends))):
        silences.append((starts[index], ends[index]))
    if len(starts) > len(ends):
        silences.append((starts[-1], total_duration))

    silences.sort()

    speech_regions: list[dict[str, float]] = []
    current_time = 0.0
    for sil_start, sil_end in silences:
        if sil_start > current_time + 0.1:
            speech_regions.append(
                {
                    "start": round(current_time, 2),
                    "end": round(sil_start, 2),
                }
            )
        current_time = sil_end

    if total_duration > current_time + 0.1:
        speech_regions.append(
            {
                "start": round(current_time, 2),
                "end": round(total_duration, 2),
            }
        )
    return speech_regions


def silencedetect_filter(noise_db: float, min_silence_sec: float) -> str:
    return f"silencedetect=n={noise_db}dB:d={min_silence_sec}"


def model_config_label(noise_db: float, min_silence_sec: float) -> str:
    return f"ffmpeg_silencedetect:n={noise_db}dB:d={min_silence_sec}"


def detect_speech_regions_silencedetect(
    audio_16k_path: str | Path,
    *,
    total_duration: float,
    stderr: str,
) -> list[dict[str, float]]:
    del audio_16k_path
    return parse_silencedetect_stderr(stderr, total_duration)
