"""Deterministic per-segment dubbing quality scoring."""

from __future__ import annotations

from typing import Any

from ..semantic_safeguards import evaluate_semantic_safeguards
from ..utterance_policy import classify_utterance_length


def score_segment_quality(segment: dict[str, Any], *, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    del settings
    reasons: list[str] = []
    penalties: list[str] = []
    severity = "good"

    if segment.get("speech_trimmed"):
        severity = "danger"
        penalties.append("speech_trimmed")
        reasons.append("speech_trimmed")

    semantic = evaluate_semantic_safeguards(
        str(segment.get("translation") or ""),
        source_text=str(segment.get("text") or ""),
    )
    if semantic.get("critical_violation"):
        severity = "danger"
        penalties.extend(semantic.get("penalties") or [])
        reasons.append("semantic_critical_violation")

    if segment.get("subtitle_overlap"):
        severity = "danger"
        reasons.append("subtitle_overlap")

    if segment.get("subtitle_out_of_bounds") or segment.get("dub_words_out_of_bounds"):
        severity = "danger"
        reasons.append("subtitle_out_of_bounds")

    tts_path = segment.get("tts_path") or segment.get("tts_raw_path")
    if not tts_path and str(segment.get("translation") or "").strip():
        severity = "danger"
        reasons.append("audio_missing")

    auto_tempo = float(segment.get("automatic_tempo_factor") or segment.get("time_stretch_factor") or 1.0)
    effective = float(segment.get("effective_speed") or auto_tempo)
    if effective > 1.25 or effective < 0.85:
        if severity != "danger":
            severity = "danger"
        reasons.append("effective_tempo_danger")
    elif effective > 1.12 or effective < 0.9:
        if severity == "good":
            severity = "warning"
        reasons.append("effective_tempo_warning")
    else:
        reasons.append("tempo_within_preferred_range")

    predicted = segment.get("predicted_duration")
    actual = segment.get("tts_speech_duration") or segment.get("tts_duration")
    if predicted is not None and actual is not None:
        err_ms = abs(float(actual) - float(predicted)) * 1000.0
        if err_ms > 700:
            if severity == "good":
                severity = "warning"
            reasons.append("prediction_error_high")
        elif err_ms < 300:
            reasons.append("prediction_error_low")

    if segment.get("dub_alignment_status") in {"fallback", "skipped", "error"}:
        if severity == "good":
            severity = "warning"
        reasons.append("alignment_fallback")
    elif segment.get("dub_alignment_status") == "exact":
        reasons.append("alignment_exact")

    if int(segment.get("tts_attempt_count") or 1) > 1:
        if severity == "good":
            severity = "review"
        reasons.append("candidate_retry")

    if any(item.get("source") == "rewrite" for item in segment.get("tts_attempts") or []):
        if severity == "good":
            severity = "review"
        reasons.append("rewrite_used")

    if classify_utterance_length(segment) == "long" and segment.get("long_segment_review_required"):
        if severity == "good":
            severity = "review"
        reasons.append("long_segment_review_required")

    if severity == "danger":
        quality_score = max(0.0, 0.35 - 0.05 * len(penalties))
    elif severity == "warning":
        quality_score = max(0.4, 0.75 - 0.08 * len(penalties))
    elif severity == "review":
        quality_score = max(0.55, 0.82 - 0.05 * len(penalties))
    else:
        quality_score = min(0.98, 0.84 + 0.04 * len([r for r in reasons if "within" in r or "low" in r or "exact" in r]))

    return {
        "quality_score": round(quality_score, 3),
        "quality_severity": severity,
        "quality_reasons": reasons,
        "quality_penalties": penalties,
    }
