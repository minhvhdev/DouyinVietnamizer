"""Short and long utterance classification for timing policy."""

from __future__ import annotations

from typing import Any

from .duration_predictor import count_vietnamese_syllables

SHORT_SPEECH_TARGET_SEC = 1.0
SHORT_SYLLABLE_MAX = 3
LONG_SPEECH_TARGET_SEC = 6.0


def classify_utterance_length(segment: dict[str, Any]) -> str:
    profile = segment.get("timing_profile") or {}
    speech_target = float(profile.get("speech_target_duration") or segment.get("original_duration") or 0.0)
    text = str(segment.get("translation") or segment.get("text") or "")
    syllables = count_vietnamese_syllables(text)
    if speech_target > 0 and speech_target < SHORT_SPEECH_TARGET_SEC:
        return "short"
    if syllables <= SHORT_SYLLABLE_MAX and speech_target <= SHORT_SPEECH_TARGET_SEC + 0.2:
        return "short"
    if speech_target >= LONG_SPEECH_TARGET_SEC:
        return "long"
    return "normal"


def short_utterance_abs_tolerance(base_abs: float) -> float:
    return max(base_abs, 0.35)


def should_skip_rewrite_for_short_utterance(segment: dict[str, Any], fit: str) -> bool:
    if classify_utterance_length(segment) != "short":
        return False
    return fit in {"good", "slightly_short", "slightly_long"}


def long_segment_review_required(segment: dict[str, Any]) -> bool:
    return classify_utterance_length(segment) == "long"
