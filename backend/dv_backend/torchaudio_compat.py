"""Compatibility shims for pyannote.audio 3.x on torchaudio 2.9+.

Torchaudio removed ``AudioMetaData``, ``info``, and ``list_audio_backends`` in
maintenance-mode releases while pyannote.audio 3.x still imports them at module
load time. Apply this patch before importing pyannote.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, NamedTuple


class AudioMetaData(NamedTuple):
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int = 16
    encoding: str = "PCM_S"


def _torchaudio_info(path: Any, backend: str | None = None) -> AudioMetaData:
    import soundfile as sf

    with sf.SoundFile(str(path)) as handle:
        return AudioMetaData(
            sample_rate=int(handle.samplerate),
            num_frames=int(handle.frames),
            num_channels=int(handle.channels),
        )


def apply_torchaudio_compat() -> None:
    import torchaudio

    if hasattr(torchaudio, "AudioMetaData"):
        return

    torchaudio.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]

    if not hasattr(torchaudio, "info"):
        torchaudio.info = _torchaudio_info  # type: ignore[attr-defined]


@contextmanager
def bypass_lightning_inspect_stack():
    """Work around lightning calling inspect.stack() during checkpoint load.

    On some Windows setups this walks speechbrain lazy-import modules and fails
    because optional deps like k2/flair are not installed.
    """
    import inspect

    original_stack = inspect.stack
    inspect.stack = lambda *args, **kwargs: []
    try:
        yield
    finally:
        inspect.stack = original_stack
