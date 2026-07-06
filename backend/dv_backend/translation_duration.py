from __future__ import annotations

import re
from typing import Any


def estimate_vietnamese_spoken_duration(text: str, *, speaking_rate_wps: float = 3.2) -> float:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return 0.0
    words = re.findall(r"\w+", cleaned, flags=re.UNICODE)
    punctuation_pause = len(re.findall(r"[,.!?;:…]", cleaned)) * 0.12
    rate = max(1.0, float(speaking_rate_wps))
    return round((len(words) / rate) + punctuation_pause, 3)


def duration_fit_prediction(estimated_duration: float, budget: float) -> str:
    if budget <= 0:
        return "unknown"
    if estimated_duration <= budget * 0.95:
        return "fits"
    if estimated_duration <= budget * 1.15:
        return "risky"
    return "over_budget"


def annotate_translation_duration(segment: dict[str, Any], *, speaking_rate_wps: float = 3.2) -> dict[str, Any]:
    updated = dict(segment)
    estimate = estimate_vietnamese_spoken_duration(str(updated.get("translation") or ""), speaking_rate_wps=speaking_rate_wps)
    budget = float(updated.get("repair_target_duration") or updated.get("duration_budget") or 0.0)
    updated["estimated_translation_duration"] = estimate
    updated["duration_fit_prediction"] = duration_fit_prediction(estimate, budget)
    updated["translation_was_duration_constrained"] = budget > 0
    return updated


def duration_prompt_suffix(duration_budget: float) -> str:
    if duration_budget <= 0:
        return ""
    return (
        f" Keep the Vietnamese line natural and concise enough for about {duration_budget:.2f} seconds of speech."
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


def build_translation_timing_guidance(
    segment: dict[str, Any],
    *,
    aligned_units: list[dict[str, Any]] | None = None,
    speaking_rate_wps: float = 3.2,
) -> dict[str, Any]:
    duration_budget = float(segment.get("repair_target_duration") or segment.get("duration_budget") or 0.0)
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
