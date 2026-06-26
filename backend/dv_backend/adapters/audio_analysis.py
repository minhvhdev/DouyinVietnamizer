"""Lightweight audio heuristics for pipeline decisions."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def audio_has_prominent_bgm(audio_path: Path, *, sample_rate: int = 16000) -> bool:
    """Return True when sustained non-speech energy suggests background music."""
    if not audio_path.is_file():
        return False

    try:
        import librosa
    except ImportError:
        return True

    try:
        samples, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    except Exception:
        return True

    if samples.size < sample_rate:
        return False

    frame_length = sample_rate // 2
    hop_length = frame_length
    rms = librosa.feature.rms(
        y=samples,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    if rms.size == 0:
        return False

    floor = float(np.percentile(rms, 20))
    active_ratio = float(np.mean(rms > max(floor * 1.8, 0.008)))
    dynamic_range = float(np.percentile(rms, 90) - floor)
    return active_ratio > 0.35 and dynamic_range > 0.015
