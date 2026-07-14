"""Deterministic ranking for timing-aware translation candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .duration_predictor import predict_spoken_duration
from .duration_fit_policy import DurationFitPolicy, acceptable_duration_fit, classify_duration_fit, policy_from_settings
from .semantic_safeguards import evaluate_semantic_safeguards

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class RankWeights:
    duration: float = 0.35
    semantic: float = 0.35
    terminology: float = 0.15
    naturalness: float = 0.15


def _repetition_penalty(text: str) -> float:
    words = [word.lower() for word in _WORD.findall(text or "")]
    if len(words) < 2:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio >= 0.85:
        return 0.0
    return min(0.4, (0.85 - unique_ratio) * 2.0)


def _style_penalty(style: str, *, target: float, predicted: float, natural_fits: bool) -> float:
    if predicted <= 0 or target <= 0:
        return 0.0
    ratio = predicted / target
    penalty = 0.0
    if style == "very_compact" and ratio > 1.05:
        penalty += 0.15
    if style == "expanded" and ratio < 0.92:
        penalty += 0.15
    if style == "compact" and ratio > 1.12:
        penalty += 0.1
    if natural_fits and style in {"compact", "very_compact"}:
        penalty += 0.12
    return penalty


def duration_fit_score(predicted: float, target: float, hard_max: float, *, confidence: float = 1.0) -> float:
    if target <= 0:
        return 0.5
    weight = max(0.35, min(1.0, confidence))
    if hard_max > 0 and predicted > hard_max:
        overrun = (predicted - hard_max) / max(hard_max, 0.05)
        return max(0.0, (0.2 - overrun) * weight)
    error = abs(predicted - target) / max(target, 0.05)
    return max(0.0, (1.0 - min(1.0, error)) * weight)


def score_candidate(
    candidate: dict[str, Any],
    *,
    timing_profile: dict[str, float],
    source_text: str,
    reference_text: str | None,
    language: str,
    voice_profile: dict[str, Any] | None,
    weights: RankWeights | None = None,
    natural_fits: bool = False,
) -> dict[str, Any]:
    weights = weights or RankWeights()
    text = str(candidate.get("text") or "").strip()
    style = str(candidate.get("style") or "natural")
    target = float(timing_profile.get("speech_target_duration") or 0.0)
    hard_max = float(timing_profile.get("hard_max_duration") or 0.0)

    prediction = predict_spoken_duration(text, language, voice_profile=voice_profile)
    predicted = float(prediction["predicted_seconds"])
    confidence = float(prediction.get("confidence") or 0.5)
    dur_score = duration_fit_score(predicted, target, hard_max, confidence=confidence)

    semantic_result = evaluate_semantic_safeguards(text, source_text=source_text, reference_text=reference_text)
    semantic = float(semantic_result["semantic_score"])
    penalties = list(semantic_result.get("penalties") or [])
    critical = bool(semantic_result.get("critical_violation"))

    naturalness = max(
        0.0,
        1.0 - _repetition_penalty(text) - _style_penalty(style, target=target, predicted=predicted, natural_fits=natural_fits),
    )

    total = (
        dur_score * weights.duration
        + semantic * weights.semantic
        + semantic * weights.terminology
        + naturalness * weights.naturalness
    )
    if critical:
        total -= 0.75

    return {
        "score": round(total, 4),
        "duration_fit_score": round(dur_score, 4),
        "semantic_score": round(semantic, 4),
        "naturalness_score": round(naturalness, 4),
        "penalties": penalties,
        "critical_violation": critical,
        "predicted_duration": predicted,
        "prediction": prediction,
        "style": style,
        "selected": False,
    }


def rank_translation_candidates(
    candidates: list[dict[str, Any]],
    *,
    timing_profile: dict[str, float],
    source_text: str,
    language: str = "vi",
    voice_profile: dict[str, Any] | None = None,
    reference_text: str | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = policy_from_settings(settings or {})
    valid = [c for c in candidates if str(c.get("text") or "").strip()]
    if not valid:
        return {
            "translation_candidates": candidates,
            "selected_candidate_index": -1,
            "selected_candidate_reason": "no_valid_candidate",
            "rankings": [],
        }

    natural_index = next((i for i, c in enumerate(candidates) if c.get("style") == "natural"), 0)
    natural_text = str(candidates[natural_index].get("text") or "").strip() if candidates else ""
    natural_pred = predict_spoken_duration(natural_text, language, voice_profile=voice_profile) if natural_text else {"predicted_seconds": 0}
    natural_fits = acceptable_duration_fit(
        classify_duration_fit(float(natural_pred["predicted_seconds"]), timing_profile, policy=policy)
    )

    rankings: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not str(candidate.get("text") or "").strip():
            rankings.append({"index": index, "score": -1.0, "skipped": True, "penalties": ["rejected_empty"]})
            continue
        scored = score_candidate(
            candidate,
            timing_profile=timing_profile,
            source_text=source_text,
            reference_text=reference_text,
            language=language,
            voice_profile=voice_profile,
            natural_fits=natural_fits,
        )
        fit = classify_duration_fit(scored["predicted_duration"], timing_profile, policy=policy)
        scored["duration_fit"] = fit
        scored["index"] = index
        rankings.append(scored)

    eligible = [item for item in rankings if not item.get("skipped") and not item.get("critical_violation")]
    if not eligible:
        fallback_index = natural_index if 0 <= natural_index < len(candidates) else 0
        return {
            "translation_candidates": candidates,
            "selected_candidate_index": fallback_index,
            "selected_candidate_reason": "semantic_fallback_to_natural",
            "selected_candidate_style": candidates[fallback_index].get("style") if candidates else None,
            "rankings": rankings,
        }

    ranked = sorted(eligible, key=lambda item: (-item["score"], item["index"]))
    best = ranked[0]
    best["selected"] = True
    target = float(timing_profile.get("speech_target_duration") or 0.0)
    reason = "highest_ranked"
    if target > 0 and abs(best["predicted_duration"] - target) <= 0.18:
        reason = "closest_to_speech_target"
    if best.get("style") == "natural" and natural_fits:
        reason = "natural_within_acceptable_range"

    return {
        "translation_candidates": candidates,
        "selected_candidate_index": best["index"],
        "selected_candidate_reason": reason,
        "selected_candidate_style": best.get("style"),
        "predicted_duration": best["predicted_duration"],
        "duration_error_ms": round(abs(best["predicted_duration"] - target) * 1000) if target > 0 else None,
        "rankings": rankings,
    }
