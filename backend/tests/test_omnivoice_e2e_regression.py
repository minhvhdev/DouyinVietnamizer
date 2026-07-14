"""Omnivoice E2E: fake truncating engine through public synthesize_short_or_chunked path."""
from __future__ import annotations

import array
import wave
from pathlib import Path

from dv_backend.omnivoice_chunk_synthesis import synthesize_short_or_chunked
from dv_backend.omnivoice_chunking import validate_chunk_reconstruction


def _write_tone_wav(path: Path, *, duration_sec: float = 0.3, rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(rate * duration_sec)
    samples = array.array("h", [6000 if (i // 80) % 2 == 0 else -6000 for i in range(frames)])
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


def _spoken_transcribe(path: Path) -> str:
    sidecar = path.with_suffix(".spoken.txt")
    return sidecar.read_text(encoding="utf-8") if sidecar.is_file() else ""


def _adaptive_settings(**overrides) -> dict:
    base = {
        "omnivoice_external_chunking_enabled": True,
        "omnivoice_chunk_max_chars": 120,
        "omnivoice_chunk_retry_max_chars_1": 120,
        "omnivoice_chunk_retry_max_chars_2": 55,
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
        "omnivoice_pause_comma_ms": 40,
        "omnivoice_pause_sentence_ms": 80,
        "omnivoice_pause_hard_ms": 20,
    }
    base.update(overrides)
    return base


def test_omnivoice_e2e_truncate_resynth_cache_and_bound(tmp_path: Path) -> None:
    truncate_at = 55
    short_ok = "Câu ngắn hoàn chỉnh."
    long_fail = " ".join(["từkhác"] * 50)
    text = f"{short_ok} {long_fail}."
    call_counts: dict[str, int] = {}

    def synthesize(chunk_text: str, output_path: Path) -> None:
        call_counts[chunk_text] = call_counts.get(chunk_text, 0) + 1
        spoken = chunk_text[:truncate_at]
        _write_tone_wav(output_path, duration_sec=max(0.12, len(spoken) * 0.008))
        output_path.with_suffix(".spoken.txt").write_text(spoken, encoding="utf-8")

    settings = _adaptive_settings()
    out1 = tmp_path / "pass1.wav"
    result1 = synthesize_short_or_chunked(
        text=text,
        output_path=out1,
        synthesize_fn=synthesize,
        settings=settings,
        segment={"index": 3},
        transcribe_fn=_spoken_transcribe,
    )
    assert result1["tts_chunking_used"] is True
    assert result1["tts_chunk_retry_count"] >= 1
    assert call_counts.get(short_ok, 0) == 1
    assert any(len(key) <= truncate_at for key in call_counts if key != short_ok)
    leaf_texts = [
        str(chunk.get("text") or chunk.get("chunk_text") or "")
        for chunk in (result1.get("tts_chunks") or [])
    ]
    leaf_texts = [t for t in leaf_texts if t]
    if leaf_texts:
        validate_chunk_reconstruction(text, leaf_texts)

    first_calls = sum(call_counts.values())
    # Second call on same path → adaptive chunk cache hits successful leaves.
    result2 = synthesize_short_or_chunked(
        text=text,
        output_path=out1,
        synthesize_fn=synthesize,
        settings=settings,
        segment={"index": 3},
        transcribe_fn=_spoken_transcribe,
    )
    second_calls = sum(call_counts.values())
    assert result2["tts_chunk_cache_hits"] >= 1
    # Failed leaf may regenerate (not cacheable); successful sibling must stay cached.
    assert second_calls >= first_calls
    assert call_counts.get(short_ok, 0) == 1  # sibling never redone
    assert second_calls - first_calls < first_calls  # not a full redo

    # Bounded failure when always truncating (never recovers).
    always_counts = {"n": 0}

    def always_truncate(chunk_text: str, output_path: Path) -> None:
        always_counts["n"] += 1
        spoken = chunk_text[:20]
        _write_tone_wav(output_path, duration_sec=0.15)
        output_path.with_suffix(".spoken.txt").write_text(spoken, encoding="utf-8")

    out3 = tmp_path / "always_bad.wav"
    result3 = synthesize_short_or_chunked(
        text=" ".join([f"word{i}" for i in range(80)]) + ".",
        output_path=out3,
        synthesize_fn=always_truncate,
        settings=_adaptive_settings(
            omnivoice_fidelity_good_threshold=0.99,
            omnivoice_chunk_max_retries=2,
        ),
        segment={"index": 8},
        transcribe_fn=_spoken_transcribe,
    )
    assert result3["tts_chunk_retry_count"] <= 2
    assert always_counts["n"] < 50
