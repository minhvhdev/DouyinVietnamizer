"""Tests for timing profile derivation."""

from __future__ import annotations

from dv_backend.timing_profile import attach_timing_profiles, build_timing_profile


def test_speech_target_uses_original_duration_not_full_window() -> None:
    profile = build_timing_profile(
        {
            "start": 1.0,
            "end": 3.5,
            "original_duration": 2.5,
            "duration_budget": 5.0,
        },
        next_segment_start=6.0,
    )
    assert profile["timeline_window"] == 5.0
    assert profile["speech_target_duration"] == 2.5
    assert profile["hard_max_duration"] > profile["speech_target_duration"]


def test_attach_timing_profiles_mutates_segments() -> None:
    segments = [{"start": 0.0, "end": 2.0, "original_duration": 2.0, "duration_budget": 4.0}]
    attach_timing_profiles(segments, total_duration=10.0)
    assert "timing_profile" in segments[0]


def test_last_segment_timeline_capped() -> None:
    profile = build_timing_profile(
        {"start": 90.0, "end": 92.0, "original_duration": 2.0},
        total_duration=120.0,
    )
    assert profile["speech_target_duration"] == 2.0
    assert profile["timeline_window"] <= 8.0 + 2.0
    assert "last_segment_timeline_capped" in (profile.get("timing_profile_warnings") or [])


def test_long_trailing_gap_does_not_inflate_speech_target() -> None:
    profile = build_timing_profile(
        {"start": 0.0, "end": 2.0, "original_duration": 2.0},
        next_segment_start=20.0,
    )
    assert profile["speech_target_duration"] == 2.0
    assert profile["timeline_window"] > profile["speech_target_duration"]


def test_soft_min_leq_target_leq_hard_max() -> None:
    profile = build_timing_profile({"start": 1.0, "end": 4.0, "original_duration": 3.0})
    assert profile["soft_min_duration"] <= profile["speech_target_duration"] <= profile["hard_max_duration"]


def test_negative_segment_normalized() -> None:
    profile = build_timing_profile({"start": 5.0, "end": 4.0, "original_duration": 1.0})
    assert profile["speech_target_duration"] >= 0.05
    assert "negative_segment_duration_normalized" in (profile.get("timing_profile_warnings") or [])
