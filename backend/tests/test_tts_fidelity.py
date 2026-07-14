"""Tests for TTS fidelity scoring."""
from __future__ import annotations

from dv_backend.tts_fidelity import (
    compact_text_similarity,
    content_coverage,
    evaluate_tts_fidelity,
    fidelity_status_from_scores,
    normalize_fidelity_text,
    text_similarity,
)


def test_similarity_ignores_punctuation_and_case() -> None:
    expected = "Khách đến, sao tôi cản?"
    heard = "khach den sao toi can"
    assert text_similarity(expected, heard) >= 0.75


def test_compact_similarity_handles_vietnamese_asr_without_spaces() -> None:
    expected = "Vào đi, dọn cho sạch. Tại sao đột nhiên ra tay?"
    heard = "VàođidọnchosạchTạisaođộtnhiênratay"
    assert compact_text_similarity(expected, heard) >= text_similarity(expected, heard)
    assert compact_text_similarity(expected, heard) >= 0.95

    result = evaluate_tts_fidelity(expected_text=expected, heard_text=heard, settings={})
    assert result["tts_fidelity_status"] == "good"
    assert result["tts_compact_text_similarity"] >= 0.95


def test_coverage_counts_tokens() -> None:
    expected = "mot hai ba bon"
    heard = "mot hai ba"
    assert content_coverage(expected, heard) == 0.75


def test_status_thresholds() -> None:
    cfg = {
        "fidelity_threshold": 0.85,
        "fidelity_review_threshold": 0.70,
        "fidelity_critical_threshold": 0.55,
    }
    assert fidelity_status_from_scores(0.90, cfg=cfg) == "good"
    assert fidelity_status_from_scores(0.75, cfg=cfg) == "review"
    assert fidelity_status_from_scores(0.60, cfg=cfg) == "poor"
    assert fidelity_status_from_scores(0.40, cfg=cfg) == "failed"


def test_evaluate_fidelity_does_not_mutate_canonical() -> None:
    canonical = "Bản dịch gốc không đổi."
    result = evaluate_tts_fidelity(
        expected_text=canonical,
        heard_text="ban dich goc",
        settings={"omnivoice_fidelity_good_threshold": 0.85},
    )
    assert canonical == "Bản dịch gốc không đổi."
    assert "tts_fidelity_status" in result


def test_normalize_unicode_vietnamese() -> None:
    assert normalize_fidelity_text("Tôi  nói.") == "tôi nói"
