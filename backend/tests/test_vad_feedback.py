"""Tests for ASR → VAD false-positive feedback helpers."""

from __future__ import annotations

from dv_backend.adapters.vad_feedback import filter_asr_false_positives, is_likely_vad_false_positive


def test_is_likely_vad_false_positive_empty_text() -> None:
    assert is_likely_vad_false_positive("   ", None) is True
    assert is_likely_vad_false_positive("", "previous") is True


def test_is_likely_vad_false_positive_duplicate_previous() -> None:
    assert is_likely_vad_false_positive("hello", "hello", prev_end=1.0, current_start=1.1) is True
    assert is_likely_vad_false_positive(" hello ", "hello", prev_end=1.0, current_start=1.3) is True


def test_is_likely_vad_false_positive_allows_distant_duplicate() -> None:
    assert is_likely_vad_false_positive("hello", "hello", prev_end=1.0, current_start=2.0) is False


def test_is_likely_vad_false_positive_valid_segment() -> None:
    assert is_likely_vad_false_positive("你好世界", None) is False
    assert is_likely_vad_false_positive("第二句", "第一句") is False


def test_filter_asr_false_positives_counts_rejected() -> None:
    segments = [
        {"start": 0.0, "end": 1.0, "text": "noise"},
        {"start": 1.0, "end": 2.0, "text": "noise"},
        {"start": 2.0, "end": 3.0, "text": "real speech"},
        {"start": 3.0, "end": 4.0, "text": "   "},
        {"start": 5.0, "end": 6.0, "text": "real speech"},
    ]
    kept, rejected = filter_asr_false_positives(segments)
    assert len(kept) == 3
    assert len(rejected) == 2
    assert rejected[0]["vad_false_positive_reason"] == "duplicate_asr"
    assert rejected[1]["vad_false_positive_reason"] == "empty_asr"
