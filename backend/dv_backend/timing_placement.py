"""Helpers for placing repaired TTS clips without hard-clipping voiced audio."""

from __future__ import annotations

import math
from typing import Any

PREFERRED_DRIFT_SEC = 0.35
SOFT_MAX_DRIFT_SEC = 0.80
HARD_MAX_DRIFT_SEC = 1.20
BOUNDARY_MARGIN_SEC = 0.025
HARD_ANCHOR_SILENCE_SEC = 1.5


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _duration_field(segment: dict[str, Any], key: str) -> float | None:
    raw = segment.get(key)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def segment_repaired_audio_duration(segment: dict[str, Any]) -> float | None:
    """Audible clip length for playback interval math (no placement minimum)."""
    if bool(segment.get("no_speech")):
        return 0.0
    repaired = _duration_field(segment, "repaired_duration")
    if repaired is not None:
        return max(0.0, repaired)
    tts = _duration_field(segment, "tts_duration")
    if tts is not None and tts > 0:
        return tts
    return None


def segment_effective_start(segment: dict[str, Any]) -> float:
    placement = _finite_float(segment.get("placement_start"))
    if placement is not None and placement >= 0:
        return placement
    start = _finite_float(segment.get("start"))
    if start is not None and start >= 0:
        return start
    return 0.0


def segment_effective_end(segment: dict[str, Any]) -> float:
    effective_start = segment_effective_start(segment)
    placement_end = _finite_float(segment.get("placement_end"))
    if placement_end is not None and placement_end >= effective_start:
        return placement_end

    repaired = segment_repaired_audio_duration(segment)
    if repaired is not None:
        return effective_start + repaired

    start = _finite_float(segment.get("start"))
    end = _finite_float(segment.get("end"))
    if start is not None and end is not None and end >= start:
        return effective_start + max(0.0, end - start)

    original = _duration_field(segment, "original_duration")
    if original is not None and original > 0:
        return effective_start + original

    budget = _finite_float(segment.get("duration_budget"))
    if budget is not None and budget > 0:
        return effective_start + budget
    return effective_start + 0.05


def segment_playback_interval(segment: dict[str, Any]) -> tuple[float, float]:
    start = segment_effective_start(segment)
    end = max(start, segment_effective_end(segment))
    return start, end


def segment_timing_diagnostics(
    segment: dict[str, Any],
    *,
    timing_stage: str | None = None,
) -> dict[str, Any]:
    source_start = _finite_float(segment.get("start"))
    source_end = _finite_float(segment.get("end"))
    placement_start = _finite_float(segment.get("placement_start"))
    placement_end = _finite_float(segment.get("placement_end"))
    effective_start, effective_end = segment_playback_interval(segment)
    payload: dict[str, Any] = {
        "source_start": source_start,
        "source_end": source_end,
        "placement_start": placement_start,
        "placement_end": placement_end,
        "effective_start": round(effective_start, 3),
        "effective_end": round(effective_end, 3),
    }
    if timing_stage:
        payload["timing_stage"] = timing_stage
    return payload


def annotate_segment_timing_diagnostics(
    segment: dict[str, Any],
    *,
    timing_stage: str | None = None,
) -> dict[str, Any]:
    segment.update(segment_timing_diagnostics(segment, timing_stage=timing_stage))
    return segment


def annotate_segments_timing_diagnostics(
    segments: list[dict[str, Any]],
    *,
    timing_stage: str | None = None,
) -> list[dict[str, Any]]:
    for segment in segments:
        annotate_segment_timing_diagnostics(segment, timing_stage=timing_stage)
    return segments


def compute_placement_starts(
    segments: list[dict[str, Any]],
    *,
    max_shift_sec: float = 0.5,
    min_silence_sec: float = 0.08,
) -> list[dict[str, Any]]:
    """Shift clip placement earlier when overflow can borrow preceding silence."""
    if not segments:
        return []

    ordered = sorted(segments, key=lambda item: int(item.get("index", 0) or 0))
    previous_end = 0.0
    for segment in ordered:
        original_start = float(segment.get("start", 0.0) or 0.0)
        placement = original_start
        repaired_duration = _clip_duration(segment)
        repair_target = float(segment.get("repair_target_duration") or 0.0)
        silence_before = max(0.0, original_start - previous_end)
        overflow = max(0.0, repaired_duration - repair_target) if repair_target > 0 else 0.0

        if overflow > 0.05 and silence_before >= min_silence_sec:
            shift = min(overflow * 0.65, silence_before * 0.85, max_shift_sec)
            placement = max(previous_end + 0.02, original_start - shift)

        segment["placement_start"] = round(placement, 3)
        segment["preferred_placement_start"] = round(placement, 3)
        segment_end = float(segment.get("end", original_start) or original_start)
        previous_end = max(previous_end, segment_end, placement + repaired_duration)
    return ordered


def _clip_duration(segment: dict[str, Any]) -> float:
    """Audible clip length. Explicit repaired_duration=0.0 must not fall through to stale fields."""
    if bool(segment.get("no_speech")):
        return 0.0
    repaired = _duration_field(segment, "repaired_duration")
    if repaired is not None:
        if repaired <= 0:
            return 0.0
        return max(0.05, repaired)
    for key in ("tts_duration", "original_duration"):
        value = _duration_field(segment, key)
        if value is not None and value > 0:
            return max(0.05, value)
    return 0.0


def schedule_soft_placements(
    segments: list[dict[str, Any]],
    *,
    preferred_drift_sec: float = PREFERRED_DRIFT_SEC,
    soft_max_drift_sec: float = SOFT_MAX_DRIFT_SEC,
    hard_max_drift_sec: float = HARD_MAX_DRIFT_SEC,
    boundary_margin_sec: float = BOUNDARY_MARGIN_SEC,
    hard_anchor_silence_sec: float = HARD_ANCHOR_SILENCE_SEC,
) -> list[dict[str, Any]]:
    """Bounded soft placement: push a little, never accumulate unbounded drift.

    Policy (ChatGPT TL):
    - prefer Chinese start
    - allow small forward push so predecessor speech finishes
    - per-segment drift caps: preferred 350ms / soft 800ms / hard 1.2s
    - beyond hard max → mark speed/compact; do NOT keep stacking timeline drift
    - allocation ceiling = next preferred start + soft drift - margin
    """
    if not segments:
        return []

    ordered = sorted(segments, key=lambda item: int(item.get("index", 0) or 0))
    prefs = [
        float(
            segment.get("preferred_placement_start")
            if segment.get("preferred_placement_start") is not None
            else (
                segment.get("placement_start")
                if segment.get("placement_start") is not None
                else (segment.get("start") if segment.get("start") is not None else 0.0)
            )
        )
        for segment in ordered
    ]
    cursor = 0.0
    previous_original_end = 0.0

    for index, segment in enumerate(ordered):
        original_start = float(segment.get("start", 0.0) or 0.0)
        preferred = prefs[index]
        duration = _clip_duration(segment)
        next_preferred = prefs[index + 1] if index + 1 < len(prefs) else preferred + duration + 5.0

        silence_before = max(0.0, original_start - previous_original_end)
        if silence_before >= hard_anchor_silence_sec:
            cursor = max(0.0, preferred - soft_max_drift_sec)

        # Zero-overlap start: never place under the previous audible cursor.
        start = max(preferred, cursor)
        drift = start - preferred
        action = "placed"
        needs_speed = False
        needs_compact = False

        if drift <= preferred_drift_sec + 1e-6:
            action = "preferred" if drift <= 0.02 else "soft_shift"
        elif drift <= soft_max_drift_sec + 1e-6:
            action = "soft_shift"
        elif drift <= hard_max_drift_sec + 1e-6:
            action = "soft_shift_max"
            needs_speed = True
        else:
            # Keep zero-overlap placement; flag compact/unresolved instead of inventing overlap.
            action = "forced_shift_unresolved"
            needs_speed = True
            needs_compact = True
            segment["timing_status"] = "UNRESOLVED_TIMING"

        # Room until next Chinese anchor, plus how far next may soft-delay.
        alloc_end = next_preferred + soft_max_drift_sec - boundary_margin_sec
        allocated = max(0.05, alloc_end - start)
        overflow = max(0.0, duration - allocated)
        if overflow > 0.15:
            needs_speed = True
            required_rate = duration / allocated
            if required_rate > 1.2:
                needs_compact = True

        segment["placement_start"] = round(start, 3)
        segment["placement_end"] = round(start + duration, 3)
        segment["placement_drift_sec"] = round(start - preferred, 3)
        segment["placement_action"] = action
        segment["timing_needs_speed"] = needs_speed
        segment["timing_needs_compact"] = needs_compact
        segment["timing_allocated_duration"] = round(allocated, 3)
        segment["timing_available_duration"] = round(allocated, 3)
        segment["timing_overflow_sec"] = round(overflow, 3)
        if segment.get("timing_status") == "UNRESOLVED_TIMING":
            pass
        elif overflow > 0.15:
            segment["timing_status"] = "OVERFLOW"
        elif abs(start - preferred) > 0.02:
            segment["timing_status"] = "SHIFTED"
        else:
            segment["timing_status"] = "OK"

        cursor = start + duration + boundary_margin_sec

        previous_original_end = max(
            previous_original_end,
            float(segment.get("end", original_start) or original_start),
        )

    return ordered


def enforce_zero_overlap_placements(
    segments: list[dict[str, Any]],
    *,
    min_gap_sec: float = 0.05,
    hard_max_drift_sec: float = HARD_MAX_DRIFT_SEC,
) -> list[dict[str, Any]]:
    """Force next_start >= previous_audible_end + gap; mark UNRESOLVED if drift > hard."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda item: int(item.get("index", 0) or 0))
    cursor = 0.0
    for segment in ordered:
        preferred = float(
            segment.get("preferred_placement_start")
            if segment.get("preferred_placement_start") is not None
            else (segment.get("start") if segment.get("start") is not None else 0.0)
        )
        start = max(preferred, cursor)
        drift = start - preferred
        if drift > hard_max_drift_sec + 1e-6:
            segment["timing_status"] = "UNRESOLVED_TIMING"
            segment["timing_needs_compact"] = True
        duration = _clip_duration(segment)
        segment["placement_start"] = round(start, 3)
        segment["placement_end"] = round(start + duration, 3)
        segment["placement_drift_sec"] = round(drift, 3)
        cursor = start + duration + min_gap_sec
    return ordered


def segments_with_voiced_overlap(
    segments: list[dict[str, Any]],
    *,
    margin_sec: float = BOUNDARY_MARGIN_SEC,
) -> list[tuple[int, int, float]]:
    """Return (index_a, index_b, overlap_sec) for adjacent placement overlaps.

    Units with repaired_duration=0 / no_speech are ignored so absorbed seam
    fragments cannot create false positives via stale tts_duration.
    """
    voiced = [item for item in segments if _clip_duration(item) > 0.02]
    ordered = sorted(voiced, key=lambda item: segment_effective_start(item))
    overlaps: list[tuple[int, int, float]] = []
    for left, right in zip(ordered, ordered[1:], strict=False):
        left_end = segment_effective_start(left) + _clip_duration(left)
        right_start = segment_effective_start(right)
        overlap = left_end + margin_sec - right_start
        if overlap > 0.02:
            overlaps.append(
                (
                    int(left.get("index", 0) or 0),
                    int(right.get("index", 0) or 0),
                    round(overlap, 3),
                )
            )
    return overlaps
