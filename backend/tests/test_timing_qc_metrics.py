"""Tests for shared timing QC metrics."""

from __future__ import annotations

import pytest

from dv_backend.timing_qc_metrics import compare_timing_metrics, compute_timing_qc_metrics


def test_first_attempt_denominator_uses_tts_segments_only() -> None:
    segments = [
        {"translation": "hello", "tts_duration": 2.0, "accepted_without_repair": True, "tts_attempt_count": 1},
        {"translation": "", "tts_duration": None},
    ]
    metrics = compute_timing_qc_metrics(segments)
    assert metrics["tts_segment_count"] == 1
    assert metrics["first_attempt_acceptance_rate"] == 1.0


def test_semantic_critical_counts_only_selected_candidate() -> None:
    segments = [
        {
            "translation": "selected translation",
            "tts_duration": 2.0,
            "selected_candidate_index": 0,
            "candidate_rankings": [
                {"index": 0, "selected": True, "critical_violation": False},
                {"index": 1, "selected": False, "critical_violation": True},
            ],
        }
    ]
    metrics = compute_timing_qc_metrics(segments)
    assert metrics["semantic_safeguard_critical_violations"] == 0

    segments[0]["candidate_rankings"][0]["critical_violation"] = True
    metrics = compute_timing_qc_metrics(segments)
    assert metrics["semantic_safeguard_critical_violations"] == 1


def test_compare_metrics_delta() -> None:
    baseline = {"first_attempt_acceptance_rate": 0.5, "speech_trim_count": 1}
    phase2 = {"first_attempt_acceptance_rate": 0.8, "speech_trim_count": 0}
    rows = compare_timing_metrics(baseline, phase2)
    by_name = {row["metric"]: row for row in rows}
    assert by_name["first_attempt_acceptance_rate"]["delta"] == pytest.approx(0.3)
    assert by_name["speech_trim_count"]["delta"] == -1
