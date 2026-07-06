"""Tests for narration placement timing."""

from __future__ import annotations

from dv_backend.timing_placement import compute_placement_starts


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
