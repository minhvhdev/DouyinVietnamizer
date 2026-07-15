"""Tests for gap-based duration fit classification (VẤN ĐỀ 1)."""

from __future__ import annotations

from dv_backend.duration_fit_policy import (
    acceptable_duration_fit,
    classify_duration_fit,
    decide_duration_repair,
    should_shorten_for_timing,
)
from dv_backend.timing_profile import build_timing_profile


def _profile(*, start: float, original: float, next_start: float) -> dict:
    built = build_timing_profile(
        {"start": start, "end": start + original, "original_duration": original},
        next_segment_start=next_start,
    )
    return {k: float(v) for k, v in built.items() if isinstance(v, (int, float))}


def test_longer_than_original_but_fits_free_window_is_accepted() -> None:
    # Original speech 2s, but the next segment does not start until 8s -> ~6s free space.
    profile = _profile(start=0.0, original=2.0, next_start=8.0)
    # A 5s dub is far longer than the 2s original yet still fits well before the next segment.
    fit = classify_duration_fit(5.0, profile)
    assert fit == "slightly_long"
    assert acceptable_duration_fit(fit) is True


def test_overflow_past_next_segment_is_too_long() -> None:
    profile = _profile(start=0.0, original=2.0, next_start=8.0)
    # hard_max ~= 8 - safety_buffer(0.12) = 7.88s. A 9s dub overflows into the next segment.
    fit = classify_duration_fit(9.0, profile)
    assert fit == "too_long"
    assert acceptable_duration_fit(fit) is False
    segment = {"start": 0.0, "end": 2.0, "original_duration": 2.0, "timing_profile": profile}
    assert should_shorten_for_timing(9.0, segment) is True


def test_much_shorter_than_original_is_warning_not_forced_repair() -> None:
    profile = _profile(start=0.0, original=2.0, next_start=8.0)
    # soft_min = 2.0 * 0.55 = 1.1s. A 0.4s dub leaves abnormal silence -> warn (too_short),
    # but this must NOT trigger a shorten/compress repair.
    fit = classify_duration_fit(0.4, profile)
    assert fit == "too_short"
    assert acceptable_duration_fit(fit) is False
    segment = {"start": 0.0, "end": 2.0, "original_duration": 2.0, "timing_profile": profile}
    assert should_shorten_for_timing(0.4, segment) is False


def test_close_to_original_is_good() -> None:
    profile = _profile(start=0.0, original=3.0, next_start=9.0)
    fit = classify_duration_fit(3.0, profile)
    assert fit == "good"
    assert acceptable_duration_fit(fit) is True


def test_slightly_short_within_window_is_accepted() -> None:
    profile = _profile(start=0.0, original=3.0, next_start=9.0)
    # 2.4s is below target 3.0 but above soft_min 1.65 -> acceptable (no forced lengthen).
    fit = classify_duration_fit(2.4, profile)
    assert fit == "slightly_short"
    assert acceptable_duration_fit(fit) is True


def test_decide_duration_repair_accepts_good_fit() -> None:
    profile = _profile(start=0.0, original=3.0, next_start=9.0)
    decision = decide_duration_repair(speech_duration=3.0, timing_profile=profile)
    assert decision["action"] == "accept"
    assert decision["duration_miss"] is False


def test_decide_duration_repair_placement_shift_only_does_not_rewrite() -> None:
    profile = _profile(start=0.0, original=3.0, next_start=9.0)
    segment = {
        "start": 0.0,
        "end": 3.0,
        "original_duration": 3.0,
        "timing_profile": profile,
        "placement_drift_sec": 0.35,
    }
    decision = decide_duration_repair(
        speech_duration=3.0,
        timing_profile=profile,
        segment=segment,
        allow_spoken_text_mutation=True,
        exact_timing_enabled=True,
    )
    assert decision["action"] == "accept"
    assert decision["placement_shift_only"] is True
    assert decision["reason"] == "placement_shift_only"


def test_decide_duration_repair_rewrite_when_too_long_and_mutation_allowed() -> None:
    profile = _profile(start=0.0, original=2.0, next_start=8.0)
    segment = {"start": 0.0, "end": 2.0, "original_duration": 2.0, "timing_profile": profile}
    decision = decide_duration_repair(
        speech_duration=9.0,
        timing_profile=profile,
        segment=segment,
        allow_spoken_text_mutation=True,
        settings={"timing_max_llm_rewrite_attempts": 1},
    )
    assert decision["action"] == "rewrite_shorten"
    assert decision["duration_miss"] is True


def test_decide_duration_repair_tempo_when_too_long_without_rewrite() -> None:
    profile = _profile(start=0.0, original=2.0, next_start=8.0)
    segment = {"start": 0.0, "end": 2.0, "original_duration": 2.0, "timing_profile": profile}
    decision = decide_duration_repair(
        speech_duration=9.0,
        timing_profile=profile,
        segment=segment,
        exact_timing_enabled=True,
        allow_spoken_text_mutation=False,
    )
    assert decision["action"] == "tempo"
    assert decision["tempo_factor"] is not None
    assert decision["tempo_factor"] >= 1.0


def test_decide_duration_repair_pad_when_too_short_exact_timing() -> None:
    profile = _profile(start=0.0, original=2.0, next_start=8.0)
    segment = {"start": 0.0, "end": 2.0, "original_duration": 2.0, "timing_profile": profile}
    decision = decide_duration_repair(
        speech_duration=0.4,
        timing_profile=profile,
        segment=segment,
        exact_timing_enabled=True,
        allow_spoken_text_mutation=False,
    )
    assert decision["action"] == "pad"
    assert decision["pad_target_duration"] > 0.0


def test_decide_duration_repair_unresolved_when_rewrite_exhausted() -> None:
    profile = _profile(start=0.0, original=2.0, next_start=8.0)
    segment = {"start": 0.0, "end": 2.0, "original_duration": 2.0, "timing_profile": profile}
    decision = decide_duration_repair(
        speech_duration=9.0,
        timing_profile=profile,
        segment=segment,
        allow_spoken_text_mutation=True,
        rewrite_attempts=1,
        max_rewrite_attempts=1,
        exact_timing_enabled=False,
    )
    assert decision["action"] == "unresolved"
