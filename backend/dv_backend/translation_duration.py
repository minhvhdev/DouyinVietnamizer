from __future__ import annotations

import re
from typing import Any

from .duration_predictor import estimate_vietnamese_spoken_duration, predict_spoken_duration

def duration_fit_prediction(estimated_duration: float, budget: float) -> str:
    if budget <= 0:
        return "unknown"
    if estimated_duration <= budget * 0.95:
        return "fits"
    if estimated_duration <= budget * 1.15:
        return "risky"
    return "over_budget"


def annotate_translation_duration(
    segment: dict[str, Any],
    *,
    speaking_rate_wps: float = 3.2,
    voice_profile: dict[str, Any] | None = None,
    language: str = "vi",
) -> dict[str, Any]:
    updated = dict(segment)
    profile = updated.get("timing_profile") or {}
    budget = float(
        profile.get("speech_target_duration")
        or updated.get("repair_target_duration")
        or updated.get("duration_budget")
        or 0.0
    )
    # Legacy vietnamese_speaking_rate_wps is only used when no calibrated voice profile is available.
    effective_profile = voice_profile
    prediction = predict_spoken_duration(
        str(updated.get("translation") or ""),
        language,
        voice_profile=effective_profile,
        speaking_rate_wps=speaking_rate_wps if effective_profile is None else None,
    )
    estimate = float(prediction["predicted_seconds"])
    updated["estimated_translation_duration"] = estimate
    updated["translation_duration_prediction"] = prediction
    updated["duration_fit_prediction"] = duration_fit_prediction(estimate, budget)
    updated["translation_was_duration_constrained"] = budget > 0
    return updated


def duration_prompt_suffix(duration_budget: float, *, language_label: str = "Vietnamese") -> str:
    if duration_budget <= 0:
        return ""
    return (
        f" Keep the {language_label} line natural and concise enough for about {duration_budget:.2f} seconds of speech."
        " Stay within the syllable/word range implied by that timing."
        " Preserve names, numbers, core meaning, and causal relationships."
    )


def count_source_speech_units(text: str, *, aligned_units: list[dict[str, Any]] | None = None) -> int:
    if aligned_units:
        timed_units = [
            unit for unit in aligned_units
            if str(unit.get("text") or unit.get("word") or unit.get("token") or "").strip()
        ]
        if timed_units:
            return len(timed_units)

    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return 0

    cjk_units = re.findall(r"[\u3400-\u9fff]", cleaned)
    if cjk_units:
        return len(cjk_units)

    return len(re.findall(r"\w+", cleaned, flags=re.UNICODE))


def target_vietnamese_syllable_count(
    duration_budget: float,
    *,
    source_speech_units: int = 0,
    speaking_rate_wps: float = 3.2,
) -> int:
    if duration_budget <= 0:
        return max(1, source_speech_units) if source_speech_units else 0

    budget_target = max(1, round(float(duration_budget) * max(1.0, float(speaking_rate_wps))))
    if source_speech_units <= 0:
        return budget_target

    source_cap = max(source_speech_units + 2, round(source_speech_units * 1.35))
    return max(1, min(budget_target, source_cap))


def timing_translate_prompt_rules() -> str:
    """Shared LLM rules: priority order + syllable range as soft fit target."""
    return (
        "Priority order (highest first): "
        "(1) preserve meaning and causal links; "
        "(2) keep exact segment/slot count and never invent or change timing; "
        "(3) produce complete speakable thoughts — no hanging fragments across adjacent slots; "
        "(4) lock entity/terminology consistency across the whole batch; "
        "(5) prefer target_vi_syllable_range [min, max] when present "
        "(translate with no fewer than min and no more than max Vietnamese syllables/words when feasible); "
        "(6) style/concision. "
        "Syllable range is a soft fit target: shorten or expand naturally, but do not cut mid-thought "
        "or spill leftover words into the next timing slot just to hit the count. "
        "duration_budget_sec is the speaking time used to derive that range. "
        "Treat source_speech_units as source-side context only, not a literal word-by-word target. "
        "Preserve names, numbers, core meaning, and causal relationships."
    )


def build_translation_timing_guidance(
    segment: dict[str, Any],
    *,
    aligned_units: list[dict[str, Any]] | None = None,
    speaking_rate_wps: float = 3.2,
) -> dict[str, Any]:
    profile = segment.get("timing_profile") or {}
    duration_budget = float(
        profile.get("speech_target_duration")
        or segment.get("repair_target_duration")
        or segment.get("duration_budget")
        or 0.0
    )
    source_units = count_source_speech_units(
        str(segment.get("text") or ""),
        aligned_units=aligned_units,
    )
    target_syllables = target_vietnamese_syllable_count(
        duration_budget,
        source_speech_units=source_units,
        speaking_rate_wps=speaking_rate_wps,
    )
    range_radius = 1 if target_syllables <= 6 else 2
    lower = max(1, target_syllables - range_radius) if target_syllables else 0
    upper = max(lower, target_syllables + range_radius) if target_syllables else 0
    return {
        "source_speech_units": source_units,
        "target_vi_syllables": target_syllables,
        "target_vi_syllable_range": [lower, upper] if target_syllables else None,
    }
