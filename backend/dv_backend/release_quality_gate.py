"""Release quality gate for experiment and production validation."""

from __future__ import annotations

from typing import Any

from .production_thresholds import DEFAULT_THRESHOLDS, thresholds_from_settings
from .timing_qc_metrics import compute_timing_qc_metrics


def evaluate_release_gate(
    segments: list[dict[str, Any]],
    *,
    metrics: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = thresholds or thresholds_from_settings(settings)
    metrics = metrics or compute_timing_qc_metrics(segments, settings=settings)
    blocking: list[str] = []
    warnings: list[str] = []

    if metrics.get("speech_trim_count", 0) > policy["speech_trim_count_max"]:
        blocking.append("speech_trim_count")
    if metrics.get("semantic_safeguard_critical_violations", 0) > policy["semantic_critical_violations_max"]:
        blocking.append("semantic_critical_violations")
    if metrics.get("subtitle_overlap_count", 0) > policy["subtitle_overlap_count_max"]:
        blocking.append("subtitle_overlap_count")

    out_of_bounds = metrics.get("subtitle_out_of_bounds_count")
    if out_of_bounds is None:
        out_of_bounds = sum(1 for s in segments if s.get("subtitle_out_of_bounds") or s.get("dub_words_out_of_bounds"))
    if out_of_bounds > policy["subtitle_out_of_bounds_count_max"]:
        blocking.append("subtitle_out_of_bounds_count")

    if metrics.get("danger_stretch_count", 0) > policy["danger_stretch_count_max"]:
        blocking.append("danger_stretch_count")

    if comparison and comparison.get("comparison_valid") is False:
        blocking.append("comparison_invalid")

    if metrics.get("warning_stretch_rate") is not None and metrics["warning_stretch_rate"] > policy["warning_stretch_rate_max"]:
        warnings.append("warning_stretch_rate_high")
    if metrics.get("rewrite_rate") is not None and metrics["rewrite_rate"] > policy["rewrite_rate_max"]:
        warnings.append("rewrite_rate_high")
    if metrics.get("candidate_retry_rate") is not None and metrics["candidate_retry_rate"] > policy["candidate_retry_rate_max"]:
        warnings.append("candidate_retry_rate_high")
    if metrics.get("p90_prediction_error_ms") is not None and metrics["p90_prediction_error_ms"] > policy["p90_prediction_error_ms_max"]:
        warnings.append("prediction_p90_high")
    if metrics.get("alignment_fallback_count", 0) > 0:
        denom = max(1, metrics.get("tts_segment_count") or 1)
        if metrics["alignment_fallback_count"] / denom > policy["alignment_fallback_rate_max"]:
            warnings.append("alignment_fallback_rate_high")

    return {
        "passed": not blocking,
        "blocking": blocking,
        "warnings": warnings,
        "metrics": metrics,
        "thresholds": policy,
    }
