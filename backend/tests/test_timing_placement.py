"""Tests for narration placement timing."""

from __future__ import annotations

from dv_backend.timing_placement import compute_placement_starts, schedule_soft_placements


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
