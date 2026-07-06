"""Helpers for placing repaired TTS clips within available silence gaps."""

from __future__ import annotations

from typing import Any


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
        repaired_duration = float(
            segment.get("repaired_duration")
            or segment.get("tts_duration")
            or 0.0
        )
        repair_target = float(segment.get("repair_target_duration") or 0.0)
        silence_before = max(0.0, original_start - previous_end)
        overflow = max(0.0, repaired_duration - repair_target) if repair_target > 0 else 0.0

        if overflow > 0.05 and silence_before >= min_silence_sec:
            shift = min(overflow * 0.65, silence_before * 0.85, max_shift_sec)
            placement = max(previous_end + 0.02, original_start - shift)

        segment["placement_start"] = round(placement, 3)
        segment_end = float(segment.get("end", original_start) or original_start)
        previous_end = max(previous_end, segment_end, placement + repaired_duration)
    return ordered
