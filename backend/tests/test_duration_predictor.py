"""Unit tests for duration predictor."""

from __future__ import annotations

import unicodedata

import pytest

from dv_backend.duration_predictor import (
    count_vietnamese_syllables,
    default_voice_profile,
    estimate_vietnamese_spoken_duration,
    normalize_text_nfc,
    predict_spoken_duration,
)


def test_short_vietnamese_sentence_positive_duration() -> None:
    result = predict_spoken_duration("Xin chào.", "vi")
    assert result["predicted_seconds"] > 0
    assert result["syllable_count"] >= 2
    assert "speech_unit_count" in result["debug"]
    assert result["debug"]["profile_source"] == "default"


def test_many_commas_add_pause() -> None:
    plain = predict_spoken_duration("Một, hai, ba.", "vi")
    many = predict_spoken_duration("Một, hai, ba, bốn, năm, sáu.", "vi")
    assert many["pause_seconds"] > plain["pause_seconds"]
    assert many["predicted_seconds"] > plain["predicted_seconds"]


def test_numbers_and_percent_increase_duration() -> None:
    base = predict_spoken_duration("Giá tăng mạnh.", "vi")
    with_numbers = predict_spoken_duration("Giá tăng 25% lên 100 USD.", "vi")
    assert with_numbers["predicted_seconds"] >= base["predicted_seconds"]


def test_proper_noun_and_acronym_features() -> None:
    result = predict_spoken_duration("Apple và API hoạt động.", "vi")
    assert result["punctuation_features"]["latin_tokens"] >= 2


def test_empty_text_zero_duration() -> None:
    result = predict_spoken_duration("   ", "vi")
    assert result["predicted_seconds"] == 0.0
    assert result["confidence"] == 0.0


def test_unicode_nfc_nfd_equivalent() -> None:
    nfc = "Tiếng Việt"
    nfd = unicodedata.normalize("NFD", nfc)
    assert count_vietnamese_syllables(nfc) == count_vietnamese_syllables(normalize_text_nfc(nfd))


def test_voice_profile_override_faster_voice() -> None:
    profile = {**default_voice_profile("vi"), "syllables_per_second": 5.5, "samples": 10}
    fast = predict_spoken_duration("Hôm nay chúng ta thử món này.", "vi", voice_profile=profile)
    slow = predict_spoken_duration("Hôm nay chúng ta thử món này.", "vi")
    assert fast["predicted_seconds"] < slow["predicted_seconds"]
    assert fast["predictor_method"] == "voice_calibrated_vi_v1"


def test_estimate_vietnamese_spoken_duration_wrapper() -> None:
    assert estimate_vietnamese_spoken_duration("Xin chào bạn.") > 0


@pytest.mark.parametrize("text", ["A", "Hi!", "123"])
def test_non_empty_text_never_zero(text: str) -> None:
    assert predict_spoken_duration(text, "vi")["predicted_seconds"] > 0
