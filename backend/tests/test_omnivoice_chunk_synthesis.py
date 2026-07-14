"""Tests for chunked OmniVoice synthesis orchestration."""
from __future__ import annotations

import array
import wave
from pathlib import Path

import pytest

from dv_backend.omnivoice_chunk_synthesis import synthesize_omnivoice_with_chunking
from dv_backend.omnivoice_wav_concat import concat_omnivoice_chunks
from dv_backend.tts_cache import segment_wav_cache_valid, build_tts_cache_identity, write_tts_sidecar
from dv_backend.tts_fidelity import text_similarity


def _write_tone_wav(path: Path, *, duration_sec: float = 0.4, rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(rate * duration_sec)
    samples = array.array("h", [8000 if (index // 100) % 2 == 0 else -8000 for index in range(frames)])
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


def _truncating_synthesize(truncate_at: int = 250):
    def synthesize(text: str, output_path: Path) -> None:
        spoken = text[:truncate_at]
        duration = max(0.15, len(spoken) * 0.01)
        _write_tone_wav(output_path, duration_sec=duration)
        output_path.with_suffix(".spoken.txt").write_text(spoken, encoding="utf-8")

    return synthesize


def _spoken_transcribe(path: Path) -> str:
    sidecar = path.with_suffix(".spoken.txt")
    return sidecar.read_text(encoding="utf-8") if sidecar.is_file() else ""


def test_concat_preserves_order_and_duration(tmp_path: Path) -> None:
    paths = []
    for index in range(3):
        wav = tmp_path / f"c{index}.wav"
        _write_tone_wav(wav, duration_sec=0.2)
        paths.append(wav)
    out = tmp_path / "merged.wav"
    timeline = concat_omnivoice_chunks(paths, pause_ms_list=[100, 100], output_path=out)
    assert len(timeline) == 3
    with wave.open(str(out), "rb") as handle:
        merged_duration = handle.getnframes() / handle.getframerate()
    assert merged_duration >= 0.6


def test_chunking_improves_fidelity_vs_truncating(tmp_path: Path) -> None:
    settings = {
        "omnivoice_chunk_max_chars": 80,
        "omnivoice_fidelity_check_enabled": True,
        "omnivoice_fidelity_check_min_chars": 100,
        "omnivoice_chunk_max_retries": 0,
    }
    long_text = " ".join([f"từ{index}" for index in range(100)])
    segment: dict = {"index": 7}

    mono_out = tmp_path / "mono.wav"
    _truncating_synthesize(40)(long_text, mono_out)
    mono_sim = text_similarity(long_text, _spoken_transcribe(mono_out))

    chunked_out = tmp_path / "tts_raw_7.wav"
    result = synthesize_omnivoice_with_chunking(
        text=long_text,
        output_path=chunked_out,
        synthesize_fn=_truncating_synthesize(500),
        settings=settings,
        segment=segment,
        transcribe_fn=_spoken_transcribe,
    )
    assert result["tts_chunk_count"] >= 2
    assert result["tts_text_similarity"] is not None
    assert result["tts_text_similarity"] >= 0.9
    assert mono_sim < result["tts_text_similarity"]


def test_chunk_cache_requires_manifest_for_long_text(tmp_path: Path) -> None:
    settings = {"omnivoice_long_text_threshold": 240}
    text = "a" * 300
    wav = tmp_path / "tts_raw_0.wav"
    _write_tone_wav(wav)
    identity = build_tts_cache_identity(settings, text=text, language="vi")
    write_tts_sidecar(wav, identity)
    assert not segment_wav_cache_valid(
        wav,
        identity,
        text=text,
        settings=settings,
        tts_dir=tmp_path,
        segment_index=0,
    )


def test_single_chunk_change_only_resynthesizes_that_chunk(tmp_path: Path) -> None:
    settings = {
        "omnivoice_chunk_max_chars": 80,
        "omnivoice_fidelity_check_enabled": False,
        "omnivoice_chunk_max_retries": 0,
    }
    text = ("Cau mot. " * 40).strip()
    calls: list[str] = []

    def synthesize(chunk_text: str, output_path: Path) -> None:
        calls.append(chunk_text)
        _write_tone_wav(output_path, duration_sec=0.2)

    out = tmp_path / "tts_raw_1.wav"
    segment = {"index": 1}
    synthesize_omnivoice_with_chunking(
        text=text,
        output_path=out,
        synthesize_fn=synthesize,
        settings=settings,
        segment=segment,
    )
    first_call_count = len(calls)
    assert first_call_count >= 2
    calls.clear()
    synthesize_omnivoice_with_chunking(
        text=text,
        output_path=out,
        synthesize_fn=synthesize,
        settings=settings,
        segment=segment,
    )
    assert len(calls) == 0
