"""Residual safety guards A+B+C: ASS conservation, atomic WAV, release_eligible."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from dv_backend.errors import AppError
from dv_backend.release_eligibility import assert_formal_release_allowed
from dv_backend.subtitle_timing import (
    assert_cues_conserve_spoken,
    normalize_spoken_for_conservation,
    _rebase_cue_texts_to_spoken,
)
from dv_backend.wav_canonical_validate import validate_canonical_wav_candidate


def _write_wav(path: Path, *, frames: int, rate: int = 16000, amplitude: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        payload = bytearray()
        for index in range(frames):
            # Simple tone-ish oscillation so peak/voiced detection succeeds.
            sample = int(amplitude * math.sin(2 * math.pi * 220 * index / rate))
            payload.extend(struct.pack("<h", sample))
        handle.writeframes(bytes(payload))


def _write_silent_wav(path: Path, *, frames: int = 16000, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * frames)


def test_normalize_spoken_for_conservation_drops_punct_keeps_digits() -> None:
    assert normalize_spoken_for_conservation("Xin chào, thế giới!") == normalize_spoken_for_conservation(
        "xin chào thế giới"
    )
    assert "12" in normalize_spoken_for_conservation("Mã số 12.")


def test_ass_conservation_passes_when_cues_match_spoken() -> None:
    cues = [
        {"start": 0.0, "end": 1.0, "text": "Xin chào"},
        {"start": 1.0, "end": 2.0, "text": "thế giới"},
    ]
    assert_cues_conserve_spoken(cues, "Xin chào, thế giới!")


def test_ass_conservation_fails_on_token_loss() -> None:
    cues = [{"start": 0.0, "end": 1.0, "text": "Xin chào"}]
    with pytest.raises(AppError) as raised:
        assert_cues_conserve_spoken(cues, "Xin chào thế giới", segment_index=7)
    assert raised.value.status_code == 409
    assert raised.value.info.code == "SUBTITLE_CONTENT_CONSERVATION_FAILED"


def test_rebase_multi_cue_conserves_spoken_tokens() -> None:
    cues = [
        {"start": 1.0, "end": 1.5, "text": "asr một"},
        {"start": 1.5, "end": 2.0, "text": "asr hai"},
        {"start": 2.0, "end": 2.8, "text": "asr ba"},
    ]
    spoken = "Xin chào các bạn thân mến"
    out = _rebase_cue_texts_to_spoken(
        cues,
        spoken,
        language="vi",
        settings={"subtitle_max_chars_per_cue": 12},
    )
    assert_cues_conserve_spoken(out, spoken, segment_index=3)
    joined = " ".join(c["text"] for c in out)
    assert "asr" not in joined.lower()


def test_validate_rejects_silent_and_accepts_voiced(tmp_path: Path) -> None:
    silent = tmp_path / "silent.wav"
    voiced = tmp_path / "voiced.wav"
    _write_silent_wav(silent, frames=16000)
    _write_wav(voiced, frames=16000, amplitude=12000)

    silent_result = validate_canonical_wav_candidate(silent)
    voiced_result = validate_canonical_wav_candidate(voiced)
    assert silent_result.ok is False
    assert silent_result.reason in {"near_silent_peak", "voiced_ratio_too_low"}
    assert voiced_result.ok is True


def test_validate_rejects_truncated_header(tmp_path: Path) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFF....WAVE")
    result = validate_canonical_wav_candidate(bad)
    assert result.ok is False


def test_release_eligible_false_blocks_formal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dv_backend import release_eligibility as mod

    monkeypatch.setattr(
        mod,
        "resolve_release_eligible",
        lambda *_a, **_k: {
            "release_eligible": False,
            "remaining_count": 2,
            "overlap_count": 0,
            "source": "test",
        },
    )

    class _Cfg:
        data_dir = tmp_path

    with pytest.raises(AppError) as raised:
        assert_formal_release_allowed(_Cfg(), "job-x", stage="mix")  # type: ignore[arg-type]
    assert raised.value.info.code == "RELEASE_ELIGIBLE_BLOCKED"


def test_resolve_ignores_stale_false_when_remaining_and_overlaps_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Critical: remaining=0 overlaps=0 must not stay blocked by stale checkpoint false."""
    from dv_backend import release_eligibility as mod
    from dv_backend.release_eligibility import resolve_release_eligible

    segments = [
        {
            "index": 1,
            "tts_spoken_text": "đã vừa khung",
            "timing_overflow_sec": 0.0,
            "timing_available_duration": 3.77,
            "repaired_duration": 2.84,
            "soft_speed_factor": 1.2,
            "start": 0.0,
            "end": 3.0,
            "placement_start": 0.0,
            "placement_end": 2.84,
        }
    ]
    monkeypatch.setattr(
        mod,
        "load_checkpoint",
        lambda *_a, **_k: {"release_eligible": False, "segments": segments},
    )

    class _Cfg:
        data_dir = tmp_path

    info = resolve_release_eligible(_Cfg(), "job-stale-false")  # type: ignore[arg-type]
    assert info["remaining_count"] == 0
    assert info["overlap_count"] == 0
    assert info["release_eligible"] is True


def test_resolve_empty_segments_trusts_checkpoint_true(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dv_backend import release_eligibility as mod
    from dv_backend.release_eligibility import resolve_release_eligible

    monkeypatch.setattr(
        mod,
        "load_checkpoint",
        lambda data_dir, job_id, step: (
            {"release_eligible": True, "segments": []}
            if step == "duration_repair"
            else None
        ),
    )

    class _Cfg:
        data_dir = tmp_path

    info = resolve_release_eligible(_Cfg(), "job-empty")  # type: ignore[arg-type]
    assert info["release_eligible"] is True
    assert info["remaining_count"] == 0
