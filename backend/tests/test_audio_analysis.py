from pathlib import Path
from unittest.mock import patch

import numpy as np

from dv_backend.adapters.audio_analysis import audio_has_prominent_bgm


def test_audio_has_prominent_bgm_detects_sustained_energy(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"wav")

    samples = np.zeros(32000, dtype=np.float32)
    samples[8000:24000] = 0.25

    with patch("librosa.load", return_value=(samples, 16000)):
        assert audio_has_prominent_bgm(audio_path) is True


def test_audio_has_prominent_bgm_returns_false_for_short_audio(tmp_path: Path) -> None:
    audio_path = tmp_path / "short.wav"
    audio_path.write_bytes(b"wav")

    samples = np.zeros(8000, dtype=np.float32)

    with patch("librosa.load", return_value=(samples, 16000)):
        assert audio_has_prominent_bgm(audio_path) is False


def test_audio_has_prominent_bgm_returns_false_for_silent_audio(tmp_path: Path) -> None:
    audio_path = tmp_path / "silent.wav"
    audio_path.write_bytes(b"wav")

    samples = np.zeros(32000, dtype=np.float32)

    with patch("librosa.load", return_value=(samples, 16000)):
        assert audio_has_prominent_bgm(audio_path) is False
