"""Silero VAD adapter — CPU-only, thread-safe model cache."""

from __future__ import annotations

import logging
import threading
import wave
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_vad_lock = threading.Lock()
_vad_model: Any = None


def _load_model() -> Any:
    global _vad_model
    with _vad_lock:
        if _vad_model is None:
            from silero_vad import load_silero_vad

            # Silero defaults to CPU; never call .cuda() here (macOS has no CUDA).
            _vad_model = load_silero_vad(onnx=False)
            logger.info("Loaded Silero VAD model (CPU)")
        return _vad_model


def reset_vad_model_cache() -> None:
    """Clear cached Silero model (useful in tests)."""
    global _vad_model
    with _vad_lock:
        _vad_model = None


def _read_audio_16k(audio_16k_path: str | Path):
    """Load mono 16 kHz PCM WAV without torchaudio (cross-platform)."""
    import torch

    with wave.open(str(audio_16k_path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM audio, got sample width {sample_width}")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sample_rate} Hz")
    return torch.from_numpy(samples)


def vad_step_silero(
    audio_16k_path: str | Path,
    *,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 300,
    speech_pad_ms: int = 150,
) -> list[dict[str, float]]:
    """Detect speech regions with Silero VAD at 16 kHz."""
    from silero_vad import get_speech_timestamps

    wav = _read_audio_16k(audio_16k_path)
    model = _load_model()
    raw = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True,
    )
    return [
        {"start": round(float(region["start"]), 2), "end": round(float(region["end"]), 2)}
        for region in raw
    ]


def model_config_label(
    *,
    threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
) -> str:
    return (
        "silero_vad:"
        f"threshold={threshold},"
        f"min_speech_ms={min_speech_duration_ms},"
        f"min_silence_ms={min_silence_duration_ms},"
        f"speech_pad_ms={speech_pad_ms}"
    )
