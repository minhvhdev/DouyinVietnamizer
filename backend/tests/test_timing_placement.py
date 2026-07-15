"""Tests for narration placement timing."""

from __future__ import annotations

from dv_backend.timing_placement import (
    compute_placement_starts,
    schedule_soft_placements,
    segment_effective_end,
    segment_effective_start,
    segment_playback_interval,
    segment_timing_diagnostics,
)


def test_segment_effective_start_prefers_placement_including_zero() -> None:
    assert segment_effective_start({"placement_start": 0.0, "start": 4.2}) == 0.0
    assert segment_effective_start({"placement_start": 3.5, "start": 2.0}) == 3.5
    assert segment_effective_start({"start": 2.0}) == 2.0
    assert segment_effective_start({"placement_start": "nan", "start": 1.5}) == 1.5


def test_segment_effective_end_prefers_placement_end_then_repaired_duration() -> None:
    segment = {
        "placement_start": 5.0,
        "placement_end": 8.25,
        "repaired_duration": 2.0,
        "start": 1.0,
        "end": 3.0,
    }
    assert segment_effective_end(segment) == 8.25

    segment.pop("placement_end")
    assert segment_effective_end(segment) == 7.0

    segment.pop("repaired_duration")
    segment["tts_duration"] = 1.5
    assert segment_effective_end(segment) == 6.5


def test_repaired_duration_wins_over_budget_tts_and_original() -> None:
    """Playback interval must use repaired_duration, not planning/source fallbacks."""
    segment = {
        "placement_start": 4.0,
        "repaired_duration": 1.2,
        "tts_duration": 9.9,
        "duration_budget": 8.0,
        "original_duration": 7.0,
        "start": 0.0,
        "end": 7.0,
    }
    start, end = segment_playback_interval(segment)
    assert start == 4.0
    assert end == 5.2


def test_segment_playback_interval_never_negative() -> None:
    start, end = segment_playback_interval(
        {
            "placement_start": 4.0,
            "repaired_duration": -0.5,
            "start": 4.0,
            "end": 3.0,
        }
    )
    assert start == 4.0
    assert end >= start


def test_segment_timing_diagnostics_preserves_source_and_effective_fields() -> None:
    payload = segment_timing_diagnostics(
        {
            "start": 1.0,
            "end": 3.0,
            "placement_start": 2.5,
            "placement_end": 5.0,
            "repaired_duration": 2.5,
        },
        timing_stage="align_final_dub",
    )
    assert payload["source_start"] == 1.0
    assert payload["source_end"] == 3.0
    assert payload["placement_start"] == 2.5
    assert payload["effective_start"] == 2.5
    assert payload["effective_end"] == 5.0
    assert payload["timing_stage"] == "align_final_dub"


def test_segment_effective_end_prefers_source_interval_over_stale_original_duration() -> None:
    segment = {
        "placement_start": 2.0,
        "start": 1.0,
        "end": 4.0,
        "original_duration": 9.9,
    }
    assert segment_effective_end(segment) == 5.0


def test_compute_placement_starts_shifts_into_preceding_silence() -> None:
    segments = [
        {
            "index": 0,
            "start": 2.0,
            "end": 4.0,
            "repair_target_duration": 2.0,
            "repaired_duration": 2.6,
        },
        {
            "index": 1,
            "start": 6.0,
            "end": 8.0,
            "repair_target_duration": 2.0,
            "repaired_duration": 2.0,
        },
    ]

    compute_placement_starts(segments)

    assert segments[0]["placement_start"] < 2.0
    assert segments[0]["placement_start"] >= 0.0
    assert segments[1]["placement_start"] == 6.0


def test_schedule_soft_placements_pushes_next_within_hard_cap() -> None:
    segments = [
        {
            "index": 0,
            "start": 0.0,
            "end": 2.0,
            "repaired_duration": 4.0,
            "placement_start": 0.0,
            "preferred_placement_start": 0.0,
        },
        {
            "index": 1,
            "start": 2.5,
            "end": 4.0,
            "repaired_duration": 1.5,
            "placement_start": 2.5,
            "preferred_placement_start": 2.5,
        },
    ]
    schedule_soft_placements(segments)
    assert segments[0]["placement_start"] == 0.0
    # First segment overflows allocation → marked for speed/compact.
    assert float(segments[0]["timing_overflow_sec"]) > 0.15
    assert segments[0]["timing_needs_speed"] is True
    # Zero-overlap: next starts after previous audible end (no invented overlap).
    prev_end = float(segments[0]["placement_start"]) + float(segments[0]["repaired_duration"])
    assert float(segments[1]["placement_start"]) + 1e-6 >= prev_end
    # Forced shift beyond hard drift is flagged unresolved, not clipped.
    if float(segments[1]["placement_drift_sec"]) > 1.2:
        assert segments[1]["timing_status"] == "UNRESOLVED_TIMING"
