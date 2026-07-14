"""Timing profile: speech window vs full timeline window for dubbing segments."""

from __future__ import annotations

from typing import Any

DEFAULT_LEADING_SILENCE_ALLOWANCE = 0.2
DEFAULT_TRAILING_SILENCE_ALLOWANCE = 0.45
DEFAULT_HARD_MAX_SLACK = 0.15
DEFAULT_SOFT_MIN_RATIO = 0.85
DEFAULT_LAST_SEGMENT_CAP_SEC = 8.0
DEFAULT_LAST_SEGMENT_MAX_EXTRA_SEC = 1.5
DEFAULT_LAST_SEGMENT_MAX_WINDOW_SEC = 8.0
MIN_SPEECH_TARGET_SEC = 0.05

# Gap-based fit policy (VẤN ĐỀ 1 fix): the acceptable window for a dubbed segment is
# the real free space up to the next segment start, not the original speech duration.
DEFAULT_SAFETY_BUFFER_SEC = 0.12
DEFAULT_MIN_FILL_RATIO = 0.55


def last_segment_policy_from_settings(settings: dict[str, Any] | None) -> tuple[float, float]:
    settings = settings or {}
    max_window = float(settings.get("last_segment_max_window_seconds", DEFAULT_LAST_SEGMENT_MAX_WINDOW_SEC) or DEFAULT_LAST_SEGMENT_MAX_WINDOW_SEC)
    max_extra = float(settings.get("last_segment_max_extra_seconds", DEFAULT_LAST_SEGMENT_MAX_EXTRA_SEC) or DEFAULT_LAST_SEGMENT_MAX_EXTRA_SEC)
    return max_extra, max_window


def timing_fit_params_from_settings(settings: dict[str, Any] | None) -> tuple[float, float]:
    """Return (safety_buffer_sec, min_fill_ratio) for gap-based fit classification."""
    settings = settings or {}
    buffer_ms = float(settings.get("timing_safety_buffer_ms", DEFAULT_SAFETY_BUFFER_SEC * 1000) or DEFAULT_SAFETY_BUFFER_SEC * 1000)
    min_fill = float(settings.get("timing_min_fill_ratio", DEFAULT_MIN_FILL_RATIO) or DEFAULT_MIN_FILL_RATIO)
    return max(0.0, buffer_ms / 1000.0), max(0.0, min(1.0, min_fill))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_segment_bounds(segment: dict[str, Any]) -> tuple[float, float, float, list[str]]:
    warnings: list[str] = []
    start = max(0.0, _float(segment.get("start")))
    end = _float(segment.get("end"), start)
    if end < start:
        warnings.append("negative_segment_duration_normalized")
        end = start + MIN_SPEECH_TARGET_SEC

    original = segment.get("original_duration")
    if original is not None:
        speech_target = max(_float(original), MIN_SPEECH_TARGET_SEC)
    else:
        speech_target = max(end - start, MIN_SPEECH_TARGET_SEC)

    if speech_target <= 0:
        speech_target = MIN_SPEECH_TARGET_SEC
        warnings.append("zero_speech_target_normalized")

    return start, end, speech_target, warnings


def build_timing_profile(
    segment: dict[str, Any],
    *,
    next_segment_start: float | None = None,
    total_duration: float | None = None,
    leading_allowance: float = DEFAULT_LEADING_SILENCE_ALLOWANCE,
    trailing_allowance_cap: float = DEFAULT_TRAILING_SILENCE_ALLOWANCE,
    hard_max_slack: float = DEFAULT_HARD_MAX_SLACK,
    soft_min_ratio: float = DEFAULT_SOFT_MIN_RATIO,
    last_segment_cap_sec: float = DEFAULT_LAST_SEGMENT_CAP_SEC,
    last_segment_max_extra_sec: float = DEFAULT_LAST_SEGMENT_MAX_EXTRA_SEC,
    safety_buffer: float = DEFAULT_SAFETY_BUFFER_SEC,
    min_fill_ratio: float = DEFAULT_MIN_FILL_RATIO,
    settings: dict[str, Any] | None = None,
) -> dict[str, float | list[str]]:
    """Derive speech-centric timing profile while keeping legacy duration_budget."""
    if settings is not None:
        extra, cap = last_segment_policy_from_settings(settings)
        last_segment_max_extra_sec = extra
        last_segment_cap_sec = cap
        safety_buffer, min_fill_ratio = timing_fit_params_from_settings(settings)
    start, end, speech_target, warnings = _normalize_segment_bounds(segment)

    legacy_budget = _float(segment.get("duration_budget"))
    if legacy_budget <= 0:
        if next_segment_start is not None:
            timeline_window = max(next_segment_start - start, speech_target)
        elif total_duration is not None:
            remaining = max(0.0, total_duration - start)
            extra_allowance = min(last_segment_max_extra_sec, max(0.0, remaining - speech_target))
            capped = min(last_segment_cap_sec, speech_target + extra_allowance)
            timeline_window = max(speech_target, capped)
            if remaining > last_segment_cap_sec:
                warnings.append("last_segment_timeline_capped")
        else:
            timeline_window = max(speech_target, end - start if end > start else speech_target)
    else:
        timeline_window = legacy_budget

    if total_duration is not None and start + timeline_window > total_duration + 0.001:
        timeline_window = max(speech_target, total_duration - start)
        warnings.append("timeline_window_capped_to_video_duration")

    if next_segment_start is not None and start + timeline_window > next_segment_start + 0.001:
        timeline_window = max(speech_target, next_segment_start - start)
        warnings.append("timeline_window_capped_to_next_segment")

    inter_segment_gap = max(0.0, timeline_window - speech_target)
    trailing_allowance = min(trailing_allowance_cap, max(0.1, inter_segment_gap * 0.65))
    leading = min(leading_allowance, max(0.0, inter_segment_gap * 0.25))

    # Gap-based hard_max: the dub may occupy the real free space up to the next segment
    # (timeline_window) minus a safety buffer, instead of being pinned near the original
    # speech duration. This is the core VẤN ĐỀ 1 fix: only overflow past the next segment
    # start is "too long"; using extra free space is fine.
    hard_max = max(speech_target, timeline_window - safety_buffer)
    # soft_min: only a warning threshold to flag abnormally long silence relative to the
    # original speech, not a hard requirement to fill the window.
    soft_min = max(MIN_SPEECH_TARGET_SEC, speech_target * min_fill_ratio)

    if soft_min > speech_target:
        soft_min = speech_target * min_fill_ratio
        warnings.append("soft_min_capped_to_speech_target")
    if hard_max < speech_target:
        hard_max = speech_target
        warnings.append("hard_max_raised_to_speech_target")

    profile: dict[str, float | list[str]] = {
        "timeline_window": round(max(MIN_SPEECH_TARGET_SEC, timeline_window), 3),
        "speech_target_duration": round(speech_target, 3),
        "soft_min_duration": round(max(MIN_SPEECH_TARGET_SEC, soft_min), 3),
        "hard_max_duration": round(max(speech_target, hard_max), 3),
        "leading_silence_allowance": round(max(0.0, leading), 3),
        "trailing_silence_allowance": round(max(0.0, trailing_allowance), 3),
    }
    if warnings:
        profile["timing_profile_warnings"] = warnings
    return profile


def timing_profile_from_segment(segment: dict[str, Any]) -> dict[str, float]:
    """Read stored profile or derive from legacy fields."""
    stored = segment.get("timing_profile")
    if isinstance(stored, dict) and stored.get("speech_target_duration") is not None:
        return {
            key: round(_float(stored.get(key)), 3)
            for key in (
                "timeline_window",
                "speech_target_duration",
                "soft_min_duration",
                "hard_max_duration",
                "leading_silence_allowance",
                "trailing_silence_allowance",
            )
            if stored.get(key) is not None
        }
    budget = _float(segment.get("repair_target_duration") or segment.get("duration_budget"))
    speech = _float(segment.get("original_duration"), budget if budget > 0 else 0.5)
    if speech <= 0:
        speech = 0.5
    built = build_timing_profile(
        {
            **segment,
            "duration_budget": budget if budget > 0 else speech,
            "original_duration": speech,
        }
    )
    return {k: float(v) for k, v in built.items() if isinstance(v, (int, float))}


def attach_timing_profiles(
    segments: list[dict[str, Any]],
    *,
    total_duration: float | None = None,
    settings: dict[str, Any] | None = None,
) -> None:
    for index, segment in enumerate(segments):
        next_start = None
        if index + 1 < len(segments):
            next_start = _float(segments[index + 1].get("start"))
        profile = build_timing_profile(
            segment,
            next_segment_start=next_start,
            total_duration=total_duration,
            settings=settings,
        )
        segment["timing_profile"] = {k: v for k, v in profile.items() if k != "timing_profile_warnings"}
        warnings = profile.get("timing_profile_warnings")
        if warnings:
            segment["timing_profile_warnings"] = warnings
        if not segment.get("duration_budget"):
            segment["duration_budget"] = profile["timeline_window"]
