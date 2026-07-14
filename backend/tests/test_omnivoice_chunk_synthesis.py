"""Tests for chunked OmniVoice synthesis orchestration."""
from __future__ import annotations

import array
import wave
from pathlib import Path

import pytest

from dv_backend.omnivoice_chunk_synthesis import (
    synthesize_omnivoice_with_chunking,
    synthesize_short_or_chunked,
)
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
        "omnivoice_chunk_fidelity_fallback_full_segment": False,
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


def test_routing_long_text_chunked_no_generate_over_max(tmp_path: Path) -> None:
    max_chars = 80
    settings = {
        "omnivoice_external_chunking_enabled": True,
        "omnivoice_chunk_max_chars": max_chars,
        "omnivoice_fidelity_check_enabled": False,
        "omnivoice_chunk_max_retries": 0,
    }
    long_text = " ".join([f"từ{index}" for index in range(120)])
    calls: list[str] = []

    def synthesize(chunk_text: str, output_path: Path) -> None:
        calls.append(chunk_text)
        assert len(chunk_text) <= max_chars
        _write_tone_wav(output_path, duration_sec=0.15)

    out = tmp_path / "long.wav"
    result = synthesize_short_or_chunked(
        text=long_text,
        output_path=out,
        synthesize_fn=synthesize,
        settings=settings,
        segment={"index": 2},
    )
    assert result["tts_chunking_used"] is True
    assert result["tts_chunk_count"] >= 2
    assert calls
    assert all(len(item) <= max_chars for item in calls)


def test_routing_short_text_single_generate(tmp_path: Path) -> None:
    settings = {
        "omnivoice_external_chunking_enabled": True,
        "omnivoice_fidelity_check_enabled": False,
    }
    short = "Khách đến sao tôi cản?"
    calls: list[str] = []

    def synthesize(chunk_text: str, output_path: Path) -> None:
        calls.append(chunk_text)
        _write_tone_wav(output_path, duration_sec=0.15)

    out = tmp_path / "short.wav"
    result = synthesize_short_or_chunked(
        text=short,
        output_path=out,
        synthesize_fn=synthesize,
        settings=settings,
        segment={"index": 3},
    )
    assert result["tts_chunking_used"] is False
    assert len(calls) == 1
    assert calls[0] == short


def test_routing_disabled_keeps_long_text_single_shot(tmp_path: Path) -> None:
    settings = {
        "omnivoice_external_chunking_enabled": False,
        "omnivoice_fidelity_check_enabled": False,
    }
    long_text = " ".join(["Đây là một câu tiếng Việt đủ dài."] * 30)
    calls: list[str] = []

    def synthesize(chunk_text: str, output_path: Path) -> None:
        calls.append(chunk_text)
        _write_tone_wav(output_path, duration_sec=0.2)

    out = tmp_path / "disabled.wav"
    result = synthesize_short_or_chunked(
        text=long_text,
        output_path=out,
        synthesize_fn=synthesize,
        settings=settings,
        segment={"index": 4},
    )
    assert result["tts_chunking_used"] is False
    assert len(calls) == 1
    assert calls[0] == long_text


def test_fidelity_truncate_falls_back_to_smaller_chunks_without_redoing_ok(tmp_path: Path) -> None:
    """Truncate-at-N on large chunks → split only the failed piece; siblings stay cached."""
    truncate_at = 60
    settings = {
        "omnivoice_external_chunking_enabled": True,
        "omnivoice_chunk_max_chars": 120,
        "omnivoice_chunk_retry_max_chars_1": 120,
        "omnivoice_chunk_retry_max_chars_2": 50,
        "omnivoice_chunk_retry_max_chars_3": 40,
        "omnivoice_chunk_max_retries": 2,
        "omnivoice_chunk_retry_on_fidelity_failure": True,
        "omnivoice_chunk_fidelity_fallback_full_segment": False,
        "omnivoice_fidelity_check_enabled": True,
        "omnivoice_fidelity_check_all_segments": True,
        "omnivoice_fidelity_check_min_chars": 1,
        "omnivoice_fidelity_good_threshold": 0.85,
        "omnivoice_fidelity_review_threshold": 0.70,
        "omnivoice_fidelity_critical_threshold": 0.55,
    }
    # First sentence short enough to survive truncate; later sentence truncates.
    short_ok = "Câu ngắn ổn."
    long_fail = " ".join(["từdài"] * 40)
    text = f"{short_ok} {long_fail}"
    call_counts: dict[str, int] = {}

    def synthesize(chunk_text: str, output_path: Path) -> None:
        call_counts[chunk_text] = call_counts.get(chunk_text, 0) + 1
        spoken = chunk_text[:truncate_at]
        duration = max(0.15, len(spoken) * 0.01)
        _write_tone_wav(output_path, duration_sec=duration)
        output_path.with_suffix(".spoken.txt").write_text(spoken, encoding="utf-8")

    out = tmp_path / "fallback.wav"
    result = synthesize_omnivoice_with_chunking(
        text=text,
        output_path=out,
        synthesize_fn=synthesize,
        settings=settings,
        segment={"index": 9},
        transcribe_fn=_spoken_transcribe,
    )
    assert result["tts_chunk_count"] >= 2
    assert result["tts_chunk_retry_count"] >= 1
    # Successful short chunk should not be resynthesized after the first pass.
    assert call_counts.get(short_ok, 0) == 1
    assert all(len(key) <= 120 for key in call_counts)
    # After smaller split, no generate text should exceed truncate threshold (passes fidelity).
    assert any(len(key) <= truncate_at for key in call_counts if key != short_ok)


def test_bounded_retries_no_infinite_loop(tmp_path: Path) -> None:
    settings = {
        "omnivoice_chunk_max_chars": 100,
        "omnivoice_chunk_retry_max_chars_1": 100,
        "omnivoice_chunk_retry_max_chars_2": 80,
        "omnivoice_chunk_retry_max_chars_3": 60,
        "omnivoice_chunk_max_retries": 2,
        "omnivoice_chunk_retry_on_fidelity_failure": True,
        "omnivoice_chunk_fidelity_fallback_full_segment": False,
        "omnivoice_fidelity_check_enabled": True,
        "omnivoice_fidelity_check_all_segments": True,
        "omnivoice_fidelity_check_min_chars": 1,
        "omnivoice_fidelity_good_threshold": 0.99,
        "omnivoice_fidelity_review_threshold": 0.98,
        "omnivoice_fidelity_critical_threshold": 0.97,
    }
    text = " ".join([f"abc{index}" for index in range(60)])
    calls = 0

    def always_bad(chunk_text: str, output_path: Path) -> None:
        nonlocal calls
        calls += 1
        # Always speak something unrelated → perpetual fidelity failure.
        _write_tone_wav(output_path, duration_sec=0.2)
        output_path.with_suffix(".spoken.txt").write_text("zzz", encoding="utf-8")

    out = tmp_path / "bounded.wav"
    result = synthesize_omnivoice_with_chunking(
        text=text,
        output_path=out,
        synthesize_fn=always_bad,
        settings=settings,
        segment={"index": 11},
        transcribe_fn=_spoken_transcribe,
    )
    assert result["tts_chunk_retry_count"] <= 2
    # Hard bound: retries shouldn't explode into dozens of leaf attempts.
    assert calls < 40


def test_final_fidelity_failure_logs_diagnostics(tmp_path: Path, caplog) -> None:
    import logging

    settings = {
        "omnivoice_chunk_max_chars": 80,
        "omnivoice_chunk_retry_max_chars_1": 80,
        "omnivoice_chunk_retry_max_chars_2": 40,
        "omnivoice_chunk_retry_max_chars_3": 20,
        "omnivoice_chunk_max_retries": 0,
        "omnivoice_chunk_retry_on_fidelity_failure": True,
        "omnivoice_chunk_fidelity_fallback_full_segment": False,
        "omnivoice_fidelity_check_enabled": True,
        "omnivoice_fidelity_check_all_segments": True,
        "omnivoice_fidelity_check_min_chars": 1,
        "omnivoice_fidelity_good_threshold": 0.99,
    }
    text = " ".join([f"từ{index}" for index in range(40)])

    def always_bad(chunk_text: str, output_path: Path) -> None:
        _write_tone_wav(output_path, duration_sec=0.2)
        output_path.with_suffix(".spoken.txt").write_text("zzz", encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="dv_backend.omnivoice_chunk_synthesis"):
        synthesize_omnivoice_with_chunking(
            text=text,
            output_path=tmp_path / "diag.wav",
            synthesize_fn=always_bad,
            settings=settings,
            segment={"index": 42},
            transcribe_fn=_spoken_transcribe,
        )
    messages = [record.getMessage() for record in caplog.records]
    assert any("omnivoice fidelity final failure" in message for message in messages)
    joined = "\n".join(messages)
    assert "chunk_index=" in joined
    assert "max_chars=" in joined
    assert "deletion_span=" in joined
    assert "reason=" in joined
    # Must not dump the entire source text at INFO/ERROR.
    assert text not in joined
