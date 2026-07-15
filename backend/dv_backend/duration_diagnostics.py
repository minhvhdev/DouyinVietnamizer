"""Duration field contract and canonical duration diagnostics.

Field semantics (must not be overwritten to change meaning between stages):

- ``duration_budget``: target spoken duration for translation/TTS planning.
  Never treat as actual audio length.
- ``predicted_duration``: estimated TTS length for the chosen translation text.
- ``tts_duration``: raw voiced WAV duration before duration-repair transforms.
- ``repaired_duration``: audible duration of the clip placed into the final mix.
  Authoritative for placement, subtitle, and QC when present (including explicit 0.0 silence).
- ``original_duration``: source-timeline speech window length, not dubbed audio.
- ``placement_start`` / ``placement_end``: playback timeline after soft placement.
- ``placement_drift_sec``: ``placement_start - preferred`` (preferred ≈ source start).

Playback / audible-duration precedence when repaired audio exists:
1. ``repaired_duration`` (explicit, including 0.0)
2. else ``tts_duration`` if > 0
3. never ``duration_budget`` as audio length
4. ``original_duration`` only as a source-window fallback when no dubbed audio exists yet
"""

from __future__ import annotations

import math
from typing import Any


_DURATION_MISS_ABS_SEC = 0.18
_PLACEMENT_SHIFT_EPS_SEC = 0.02

_REPAIR_ACTION_ALIASES = {
    "none": "none",
    "": "none",
    "time_stretch": "tempo",
    "tempo": "tempo",
    "pad": "pad",
    "silence_pad": "pad",
    "trim": "trim",
    "speech_trim": "trim",
    "rewrite": "rewrite",
    "llm_shorten": "rewrite",
    "conflict_cluster_merge": "conflict_cluster_merge",
    "interpolated": "interpolated",
}


def safe_duration(value: Any) -> float | None:
    """Return a finite duration in seconds, or None if missing/invalid."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(a - b, 3)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def normalize_repair_action(segment: dict[str, Any]) -> str:
    raw = str(segment.get("repaired_method") or "none").strip().lower()
    if raw in _REPAIR_ACTION_ALIASES:
        return _REPAIR_ACTION_ALIASES[raw]
    if "rewrite" in raw or "llm_shorten" in raw:
        return "rewrite"
    if "time_stretch" in raw or "tempo" in raw:
        return "tempo"
    if "trim" in raw:
        return "trim"
    if "pad" in raw:
        return "pad"
    if "conflict" in raw:
        return "conflict_cluster_merge"
    if raw == "none":
        return "none"
    return "other"


def classify_placement_shift_cause(
    segment: dict[str, Any],
    *,
    previous: dict[str, Any] | None = None,
) -> str | None:
    drift = safe_duration(segment.get("placement_drift_sec"))
    if drift is None:
        source = safe_duration(segment.get("start"))
        placement = safe_duration(segment.get("placement_start"))
        if source is None or placement is None:
            return None
        drift = abs(placement - source)
    if drift <= _PLACEMENT_SHIFT_EPS_SEC:
        return None

    status = str(segment.get("timing_status") or "")
    if status == "UNRESOLVED_TIMING":
        return "unresolved_overlap"
    if previous is not None:
        prev_overflow = safe_duration(previous.get("timing_overflow_sec")) or 0.0
        if prev_overflow > 0.05 or previous.get("timing_needs_compact"):
            return "previous_repaired_overflow"
    if status == "SHIFTED" or drift > _PLACEMENT_SHIFT_EPS_SEC:
        return "soft_schedule_or_source_conflict"
    return "soft_schedule_or_source_conflict"


def segment_duration_diagnostics(
    segment: dict[str, Any],
    *,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pure diagnostics payload. Does not mutate ``segment``."""
    budget = safe_duration(segment.get("duration_budget"))
    predicted = safe_duration(segment.get("predicted_duration"))
    tts = safe_duration(segment.get("tts_duration"))
    repaired = safe_duration(segment.get("repaired_duration"))
    original = safe_duration(segment.get("original_duration"))
    source_start = safe_duration(segment.get("start"))
    placement_start = safe_duration(segment.get("placement_start"))
    placement_drift = safe_duration(segment.get("placement_drift_sec"))
    if placement_drift is None and source_start is not None and placement_start is not None:
        placement_drift = round(placement_start - source_start, 3)

    audio_for_fit = repaired if repaired is not None else tts
    duration_miss = None
    if audio_for_fit is not None and budget is not None and budget > 0:
        duration_miss = abs(audio_for_fit - budget) > _DURATION_MISS_ABS_SEC

    placement_shift_sec = placement_drift
    placement_shifted = (
        placement_shift_sec is not None and abs(placement_shift_sec) > _PLACEMENT_SHIFT_EPS_SEC
    )
    placement_shift_cause = (
        classify_placement_shift_cause(segment, previous=previous) if placement_shifted else None
    )

    issues: list[str] = []
    if duration_miss:
        issues.append("duration_miss")
    if placement_shifted:
        issues.append("placement_shift")
    if str(segment.get("timing_status") or "") == "UNRESOLVED_TIMING":
        issues.append("unresolved_overlap")
    if (safe_duration(segment.get("timing_overflow_sec")) or 0.0) > 0.15:
        issues.append("allocation_overflow")

    planned = segment.get("duration_repair_decision")
    planned_action = None
    if isinstance(planned, dict):
        planned_action = planned.get("action")

    return {
        "duration_budget": budget,
        "predicted_duration": predicted,
        "tts_duration": tts,
        "repaired_duration": repaired,
        "original_duration": original,
        "tts_vs_budget_delta_sec": _delta(tts, budget),
        "tts_vs_budget_ratio": _ratio(tts, budget),
        "repaired_vs_budget_delta_sec": _delta(repaired, budget),
        "repaired_vs_budget_ratio": _ratio(repaired, budget),
        "repaired_vs_tts_delta_sec": _delta(repaired, tts),
        "predicted_vs_tts_delta_sec": _delta(predicted, tts),
        "duration_miss": duration_miss,
        "placement_shift_sec": placement_shift_sec,
        "placement_shifted": placement_shifted,
        "placement_shift_cause": placement_shift_cause,
        "repair_action": normalize_repair_action(segment),
        "planned_repair_action": planned_action,
        "timing_status": segment.get("timing_status"),
        "issues": issues,
    }


def annotate_segments_duration_diagnostics(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach ``duration_diagnostics`` on each segment in-memory. Mutates list items."""
    previous: dict[str, Any] | None = None
    ordered = sorted(segments, key=lambda item: int(item.get("index", 0) or 0))
    index_to_diag: dict[int, dict[str, Any]] = {}
    for segment in ordered:
        diag = segment_duration_diagnostics(segment, previous=previous)
        index_to_diag[int(segment.get("index", 0) or 0)] = diag
        previous = segment
    for segment in segments:
        key = int(segment.get("index", 0) or 0)
        if key in index_to_diag:
            segment["duration_diagnostics"] = index_to_diag[key]
    return segments
