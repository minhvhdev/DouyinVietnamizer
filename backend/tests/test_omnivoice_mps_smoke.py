from pathlib import Path
import wave

import numpy as np

from scripts.smoke_omnivoice_mps import analyze_wav, resolved_snapshot_revision


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 24_000) -> None:
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def test_analyze_wav_accepts_finite_non_silent_audio(tmp_path: Path) -> None:
    timeline = np.arange(24_000, dtype=np.float32) / 24_000
    samples = 0.2 * np.sin(2 * np.pi * 220 * timeline)
    path = tmp_path / "valid.wav"
    _write_wav(path, samples)

    analysis = analyze_wav(path)

    assert analysis["errors"] == []
    assert analysis["sample_rate"] == 24_000
    assert analysis["finite"] is True


def test_analyze_wav_rejects_silence(tmp_path: Path) -> None:
    path = tmp_path / "silent.wav"
    _write_wav(path, np.zeros(24_000, dtype=np.float32))

    analysis = analyze_wav(path)

    assert any(error.startswith("audio_silent_rms=") for error in analysis["errors"])


def test_resolved_snapshot_revision_extracts_huggingface_hash() -> None:
    source = "/Users/test/.cache/huggingface/hub/models--k2-fsa--OmniVoice/snapshots/abc123"

    assert resolved_snapshot_revision(source) == "abc123"
