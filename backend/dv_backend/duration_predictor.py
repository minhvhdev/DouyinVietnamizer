"""Language-aware spoken duration prediction for dubbing candidates."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .dubbing_languages import default_speaking_rate_wps, dub_language_from_settings

_VI_VOWEL = re.compile(
    r"[aeiouyăâêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)
_WORD = re.compile(r"\w+", re.UNICODE)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?%?")
_LATIN = re.compile(r"[A-Za-z]{2,}")
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")
_ELLIPSIS = re.compile(r"\.{3}|…")
_COMMA = re.compile(r",|，")
_SENTENCE_END = re.compile(r"[.!?。！？]")


def normalize_text_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def count_vietnamese_syllables(text: str) -> int:
    cleaned = normalize_text_nfc(re.sub(r"\s+", " ", (text or "").strip()))
    if not cleaned:
        return 0
    vowels = _VI_VOWEL.findall(cleaned)
    if vowels:
        return max(1, len(vowels))
    words = _WORD.findall(cleaned)
    return max(1, len(words)) if words else 0


_THAI_CHAR = re.compile(r"[\u0E00-\u0E7F]")


def count_thai_speech_units(text: str) -> int:
    """Estimate Thai speech units (words/syllable-like) for duration calibration."""
    cleaned = normalize_text_nfc((text or "").strip())
    if not cleaned:
        return 0
    try:
        from .subtitle_timing import _thai_word_tokens

        tokens = [tok for tok in _thai_word_tokens(cleaned) if tok.strip()]
    except Exception:
        tokens = []
    if len(tokens) >= 2:
        return len(tokens)
    thai_chars = len(_THAI_CHAR.findall(cleaned))
    if thai_chars:
        # Rough syllable estimate when tokenization returns a single blob.
        return max(2, round(thai_chars / 2.4))
    words = _WORD.findall(cleaned)
    return max(1, len(words)) if words else 0


def count_speech_units(text: str, language: str = "vi") -> int:
    lang = str(language or "vi").strip().lower()
    if lang in {"th", "thai", "thailand"}:
        return count_thai_speech_units(text)
    return count_vietnamese_syllables(text)


def extract_punctuation_features(text: str) -> dict[str, int]:
    cleaned = normalize_text_nfc(text or "")
    return {
        "comma_count": len(_COMMA.findall(cleaned)),
        "sentence_end_count": len(_SENTENCE_END.findall(cleaned)),
        "ellipsis_count": len(_ELLIPSIS.findall(cleaned)),
        "number_tokens": len(_NUMBER.findall(cleaned)),
        "latin_tokens": len(_LATIN.findall(cleaned)),
        "acronym_tokens": len(_ACRONYM.findall(cleaned)),
    }


def default_voice_profile(language: str = "vi", *, speaking_rate_wps: float | None = None) -> dict[str, Any]:
    rate_wps = speaking_rate_wps or default_speaking_rate_wps(language)
    syllables_per_second = max(2.5, min(6.0, rate_wps * 1.15))
    return {
        "voice_id": "default",
        "language": language,
        "samples": 0,
        "syllables_per_second": round(syllables_per_second, 3),
        "comma_pause_ms": 145,
        "sentence_pause_ms": 270,
        "ellipsis_pause_ms": 320,
        "number_pause_ms": 80,
        "latin_token_pause_ms": 60,
        "prediction_error_mae_ms": None,
    }


def predict_spoken_duration(
    text: str,
    language: str = "vi",
    *,
    voice_profile: dict[str, Any] | None = None,
    punctuation_profile: dict[str, Any] | None = None,
    speaking_rate_wps: float | None = None,
) -> dict[str, Any]:
    cleaned = normalize_text_nfc(re.sub(r"\s+", " ", (text or "").strip()))
    if not cleaned:
        return {
            "predicted_seconds": 0.0,
            "speech_seconds": 0.0,
            "pause_seconds": 0.0,
            "syllable_count": 0,
            "confidence": 0.0,
            "predictor_method": "empty_text",
        }

    profile = voice_profile or default_voice_profile(language, speaking_rate_wps=speaking_rate_wps)
    punct = punctuation_profile or extract_punctuation_features(cleaned)
    syllables = count_vietnamese_syllables(cleaned)
    sps = max(2.0, float(profile.get("syllables_per_second") or 4.0))
    speech_seconds = syllables / sps

    comma_ms = float(profile.get("comma_pause_ms", 145))
    sentence_ms = float(profile.get("sentence_pause_ms", 270))
    ellipsis_ms = float(profile.get("ellipsis_pause_ms", 320))
    number_ms = float(profile.get("number_pause_ms", 80))
    latin_ms = float(profile.get("latin_token_pause_ms", 60))

    pause_seconds = (
        punct["comma_count"] * comma_ms
        + punct["sentence_end_count"] * sentence_ms
        + punct["ellipsis_count"] * ellipsis_ms
        + punct["number_tokens"] * number_ms
        + punct["latin_tokens"] * latin_ms
        + punct["acronym_tokens"] * latin_ms
    ) / 1000.0

    predicted = max(0.05, speech_seconds + pause_seconds)
    samples = int(profile.get("samples") or 0)
    confidence = min(0.95, 0.55 + min(samples, 40) * 0.01)

    method = "voice_calibrated_vi_v1" if samples >= 5 else "default_vi_v1"
    profile_source = "voice_profile" if samples >= 5 else "default"
    if language != "vi":
        method = f"default_{language}_v1"
        profile_source = "default"

    base_rate = round(syllables / max(predicted, 0.05), 2)
    speech_units = max(syllables, len(_WORD.findall(cleaned)))

    return {
        "predicted_seconds": round(predicted, 3),
        "speech_seconds": round(speech_seconds, 3),
        "pause_seconds": round(pause_seconds, 3),
        "syllable_count": syllables,
        "confidence": round(confidence, 2),
        "predictor_method": method,
        "punctuation_features": punct,
        "debug": {
            "speech_unit_count": speech_units,
            "number_unit_count": punct["number_tokens"],
            "acronym_unit_count": punct["acronym_tokens"],
            "punctuation_pause_seconds": round(pause_seconds, 3),
            "base_rate": base_rate,
            "profile_source": profile_source,
        },
    }


def voice_profile_from_settings(settings: dict[str, Any]) -> dict[str, Any]:
    from .voice_duration_profile import resolve_voice_profile

    language = dub_language_from_settings(settings)
    return resolve_voice_profile(settings, language=language)


def estimate_vietnamese_spoken_duration(text: str, *, speaking_rate_wps: float = 3.2) -> float:
    """Backward-compatible wrapper."""
    result = predict_spoken_duration(
        text,
        "vi",
        speaking_rate_wps=speaking_rate_wps,
    )
    return float(result["predicted_seconds"])
