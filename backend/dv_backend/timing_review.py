"""Timing review helpers: flag infeasible-at-cap segments for user edits."""

from __future__ import annotations

import math
import re
from typing import Any


REVIEW_STATUS = "timing_review_required"
INFEASIBLE_REASON = "infeasible_at_cap"
OVERFLOW_THRESHOLD_SEC = 0.15

_REVIEW_FIELDS = (
    "timing_review_reason",
    "release_blocking",
    "needs_review",
    "required_speed",
    "max_allowed_speed",
    "overflow_seconds",
    "estimated_words_to_remove",
    "estimated_words_to_remove_min",
    "estimated_words_to_remove_max",
    "observed_words_per_second",
)


def estimate_word_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text or "").strip()))


def estimate_words_to_remove(
    *,
    text: str,
    fitted_duration: float,
    available_duration: float,
    overflow_seconds: float | None = None,
    speech_duration: float | None = None,
) -> dict[str, Any]:
    """Estimate a conservative word-reduction range for UI (never silent truncation).

    Prefer speech envelope for WPS so leading/trailing silence does not dilute the
    estimate. Silence pads are treated as mostly irreducible by deleting words.
    """
    duration = max(0.05, float(fitted_duration or 0.0))
    available = max(0.05, float(available_duration or 0.0))
    overflow = float(overflow_seconds) if overflow_seconds is not None else max(0.0, duration - available)
    # Never invent overflow when current audio already fits the allocated window.
    overflow = max(0.0, min(overflow, max(0.0, duration - available)))
    words = max(1, estimate_word_count(text))
    speech = float(speech_duration) if speech_duration is not None else 0.0
    if speech <= 0.05 or speech > duration + 1e-6:
        speech = duration
    silence_pad = max(0.0, duration - speech)
    # Words only shrink the speech portion; pad remains after re-TTS more often than not.
    speech_budget = max(0.05, available - silence_pad)
    reducible_need = max(overflow, max(0.0, speech - speech_budget))
    speech_wps = words / max(0.05, speech)
    raw = reducible_need * speech_wps if reducible_need > 0 else 0.0
    # 25% margin: OmniVoice length is not strictly linear with word count.
    estimated = int(math.ceil(raw * 1.25)) if raw > 0 else 0
    if overflow > OVERFLOW_THRESHOLD_SEC and estimated < 1:
        estimated = 1
    if estimated:
        # Bias upward for UI: never advertise an optimistic "min" below the estimate.
        min_words = estimated
        max_words = max(estimated, int(math.ceil(estimated * 1.5)))
    else:
        min_words = 0
        max_words = 0
    return {
        "overflow_seconds": round(overflow, 3),
        "observed_words_per_second": round(speech_wps, 3),
        "estimated_words_to_remove": estimated,
        "estimated_words_to_remove_min": min_words,
        "estimated_words_to_remove_max": max_words,
    }


def _segment_durations(segment: dict[str, Any]) -> tuple[float, float]:
    available = float(segment.get("timing_available_duration") or 0.0)
    duration = float(segment.get("repaired_duration") or segment.get("tts_duration") or 0.0)
    return duration, available


def residual_overflow_at_cap(
    segment: dict[str, Any],
    *,
    absolute_max_rate: float,
) -> tuple[float, float, float]:
    """Return (residual_overflow, duration_at_cap, available) from current audio vs window.

    Always derived from repaired/tts duration and timing_available_duration — never trust
    a stale timing_overflow_sec alone (speed apply can shrink audio without refreshing it).
    """
    duration, available = _segment_durations(segment)
    abs_max = max(1.0, float(absolute_max_rate))
    applied = max(1.0, float(segment.get("soft_speed_factor") or 1.0))
    if available <= 0.05 or duration <= 0:
        return 0.0, duration, available
    if applied >= abs_max - 1e-3:
        duration_at_cap = duration
    else:
        remaining_speed = abs_max / applied
        duration_at_cap = duration / remaining_speed
    residual = max(0.0, duration_at_cap - available)
    return residual, duration_at_cap, available


def _clear_review_marks(segment: dict[str, Any]) -> None:
    for key in _REVIEW_FIELDS:
        segment.pop(key, None)
    if segment.get("timing_status") == REVIEW_STATUS:
        segment["timing_status"] = "OK"


def mark_infeasible_at_cap(
    segment: dict[str, Any],
    *,
    absolute_max_rate: float,
    overflow_threshold: float = OVERFLOW_THRESHOLD_SEC,
) -> bool:
    """Flag a segment that still overflows even after speed ≤ absolute_max_rate.

    Returns True when the segment was marked for review.
    """
    text = str(segment.get("tts_spoken_text") or segment.get("translation") or "").strip()
    if not text or bool(segment.get("no_speech")):
        _clear_review_marks(segment)
        return False

    residual_overflow, duration_at_cap, available = residual_overflow_at_cap(
        segment, absolute_max_rate=absolute_max_rate
    )
    duration, _ = _segment_durations(segment)
    abs_max = max(1.0, float(absolute_max_rate))

    # Hard guard: current clip already fits the allocated window → leave review queue.
    if duration > 0 and available > 0.05 and duration <= available + overflow_threshold:
        _clear_review_marks(segment)
        return False

    if residual_overflow <= overflow_threshold or available <= 0.05 or duration <= 0:
        _clear_review_marks(segment)
        return False

    required = duration_at_cap / available if available > 0 else abs_max + 1.0
    speech = segment.get("tts_speech_duration") or segment.get("tts_active_speech_duration")
    estimate = estimate_words_to_remove(
        text=text,
        fitted_duration=duration,
        available_duration=available,
        overflow_seconds=residual_overflow,
        speech_duration=float(speech) if speech is not None else None,
    )
    segment["timing_status"] = REVIEW_STATUS
    segment["timing_review_reason"] = INFEASIBLE_REASON
    segment["required_speed"] = round(required, 4)
    segment["max_allowed_speed"] = round(abs_max, 4)
    segment["overflow_seconds"] = estimate["overflow_seconds"]
    segment["estimated_words_to_remove"] = estimate["estimated_words_to_remove"]
    segment["estimated_words_to_remove_min"] = estimate["estimated_words_to_remove_min"]
    segment["estimated_words_to_remove_max"] = estimate["estimated_words_to_remove_max"]
    segment["observed_words_per_second"] = estimate["observed_words_per_second"]
    segment["needs_review"] = True
    segment["release_blocking"] = True
    # Keep schedule overflow aligned with what the UI and list gate use.
    segment["timing_overflow_sec"] = round(max(0.0, duration - available), 3)
    return True


def flag_infeasible_segments(
    segments: list[dict[str, Any]],
    *,
    absolute_max_rate: float,
) -> list[dict[str, Any]]:
    flagged: list[dict[str, Any]] = []
    for segment in segments:
        if mark_infeasible_at_cap(segment, absolute_max_rate=absolute_max_rate):
            flagged.append(segment)
    return flagged


def list_timing_review_segments(
    segments: list[dict[str, Any]],
    *,
    absolute_max_rate: float = 1.2,
) -> list[dict[str, Any]]:
    """Rows that still overflow the allocated window after max allowed speed.

    Source of truth is current repaired/tts duration vs timing_available_duration (and
    remaining headroom to absolute_max_rate). Stale timing_overflow_sec alone is ignored.
    """
    abs_max = max(1.0, float(absolute_max_rate or 1.2))
    rows: list[dict[str, Any]] = []
    for segment in segments:
        if not str(segment.get("tts_spoken_text") or segment.get("translation") or "").strip():
            continue
        if bool(segment.get("no_speech")):
            continue
        duration, available = _segment_durations(segment)
        if duration <= 0 or available <= 0.05:
            continue
        # Never surface "rút gọn" when audio already fits the window.
        if duration <= available + OVERFLOW_THRESHOLD_SEC:
            continue

        residual, duration_at_cap, _ = residual_overflow_at_cap(
            segment, absolute_max_rate=abs_max
        )
        if residual <= OVERFLOW_THRESHOLD_SEC:
            continue

        estimate_min = segment.get("estimated_words_to_remove_min")
        estimate_max = segment.get("estimated_words_to_remove_max")
        estimate = segment.get("estimated_words_to_remove")
        if estimate is None or estimate_min is None:
            speech = segment.get("tts_speech_duration") or segment.get("tts_active_speech_duration")
            words = estimate_words_to_remove(
                text=str(segment.get("tts_spoken_text") or segment.get("translation") or ""),
                fitted_duration=duration,
                available_duration=available,
                overflow_seconds=residual,
                speech_duration=float(speech) if speech is not None else None,
            )
            estimate = words["estimated_words_to_remove"]
            estimate_min = words["estimated_words_to_remove_min"]
            estimate_max = words["estimated_words_to_remove_max"]

        required = segment.get("required_speed")
        if required is None and available > 0:
            required = round(duration_at_cap / available, 4)

        rows.append(
            {
                "index": int(segment.get("index", 0) or 0),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "source_text": segment.get("text"),
                "spoken_text": segment.get("tts_spoken_text") or segment.get("translation") or "",
                "plan_version": int(segment.get("plan_version") or 1),
                "timing_status": segment.get("timing_status") or REVIEW_STATUS,
                "timing_review_reason": segment.get("timing_review_reason") or INFEASIBLE_REASON,
                "required_speed": required,
                "max_allowed_speed": segment.get("max_allowed_speed") or round(abs_max, 4),
                "overflow_seconds": round(residual, 3),
                "estimated_words_to_remove": estimate,
                "estimated_words_to_remove_min": estimate_min,
                "estimated_words_to_remove_max": estimate_max,
                "timing_available_duration": available,
                "repaired_duration": duration,
                "release_blocking": bool(segment.get("release_blocking", True)),
            }
        )
    return rows
