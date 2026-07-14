"""Helpers for placing repaired TTS clips without hard-clipping voiced audio."""

from __future__ import annotations

from typing import Any

PREFERRED_DRIFT_SEC = 0.35
SOFT_MAX_DRIFT_SEC = 0.80
HARD_MAX_DRIFT_SEC = 1.20
BOUNDARY_MARGIN_SEC = 0.025
HARD_ANCHOR_SILENCE_SEC = 1.5


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


def _duration_field(segment: dict[str, Any], key: str) -> float | None:
    raw = segment.get(key)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


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
            or segment.get("placement_start")
            or segment.get("start")
            or 0.0
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

        # Advance cursor only by the allocated (fit) horizon when overflowing, so drift
        # cannot accumulate forever while speed/compact still pending.
        # Zero-overlap invariant: next start is always after previous audible end.
        # Duration fitting (speed/compact/cluster) must happen before mix; do not
        # fabricate overlap by advancing only the allocation window.
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
            or segment.get("start")
            or 0.0
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
    ordered = sorted(voiced, key=lambda item: float(item.get("placement_start") or 0.0))
    overlaps: list[tuple[int, int, float]] = []
    for left, right in zip(ordered, ordered[1:], strict=False):
        left_end = float(left.get("placement_start") or 0.0) + _clip_duration(left)
        right_start = float(right.get("placement_start") or 0.0)
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
