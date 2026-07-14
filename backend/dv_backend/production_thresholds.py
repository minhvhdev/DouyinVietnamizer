"""Production acceptance thresholds for Phase 3 release validation."""

from __future__ import annotations

from typing import Any

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "speech_trim_count_max": 0,
    "semantic_critical_violations_max": 0,
    "subtitle_overlap_count_max": 0,
    "subtitle_out_of_bounds_count_max": 0,
    "danger_stretch_count_max": 0,
    "warning_stretch_rate_max": 0.10,
    "first_attempt_acceptance_rate_min": 0.70,
    "candidate_retry_rate_max": 0.30,
    "rewrite_rate_max": 0.15,
    "median_prediction_error_ms_max": 300.0,
    "p90_prediction_error_ms_max": 700.0,
    "median_effective_tempo_min": 0.95,
    "median_effective_tempo_max": 1.07,
    "p90_effective_tempo_min": 0.90,
    "p90_effective_tempo_max": 1.12,
    "alignment_fallback_rate_max": 0.10,
    "min_reviewed_segments_for_recommendation": 20,
    "min_videos_for_recommendation": 3,
}


def thresholds_from_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_THRESHOLDS)
    if settings:
        for key, value in settings.items():
            if key.startswith("production_") and key.endswith(("_max", "_min")):
                merged[key.removeprefix("production_")] = value
            elif key in merged:
                merged[key] = value
    return merged
