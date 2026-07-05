from __future__ import annotations

from typing import Any

CHINESE_PUNCTUATION = "。！？；：，、"


def _segment_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _units_for_region(aligned_units: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for unit in aligned_units:
        unit_start = float(unit.get("start", 0.0))
        unit_end = float(unit.get("end", unit_start))
        midpoint = (unit_start + unit_end) / 2.0
        if start <= midpoint <= end:
            units.append(unit)
    return units


def split_segment_semantically(
    segment: dict[str, Any],
    speech_regions: list[dict[str, Any]],
    aligned_units: list[dict[str, Any]],
    *,
    boundary_tolerance_sec: float = 0.7,
    min_segment_sec: float = 0.2,
) -> list[dict[str, Any]]:
    text = str(segment.get("text") or "").strip()
    start = float(segment.get("start", 0.0) or 0.0)
    end = float(segment.get("end", start) or start)
    if not text or end - start <= min_segment_sec:
        return [dict(segment)]

    regions = [
        {
            "start": max(start, float(region.get("start", start) or start)),
            "end": min(end, float(region.get("end", end) or end)),
        }
        for region in speech_regions
        if float(region.get("end", 0.0) or 0.0) > start and float(region.get("start", 0.0) or 0.0) < end
    ]
    regions = [region for region in regions if region["end"] - region["start"] >= min_segment_sec]
    if len(regions) < 2 or not aligned_units:
        legacy = dict(segment)
        legacy.setdefault("split_method", "legacy")
        legacy.setdefault("original_segment_id", segment.get("index"))
        return [legacy]

    parts: list[dict[str, Any]] = []
    consumed = ""
    for index, region in enumerate(regions):
        units = _units_for_region(aligned_units, region["start"], region["end"])
        if not units:
            return [dict(segment)]
        part_text = "".join(str(unit.get("text") or "") for unit in units).strip()
        if not part_text:
            return [dict(segment)]
        part_start = max(start, min(float(unit.get("start", region["start"])) for unit in units))
        part_end = min(end, max(float(unit.get("end", region["end"])) for unit in units))
        if parts and part_start < float(parts[-1]["end"]):
            part_start = float(parts[-1]["end"])
        if part_end <= part_start:
            return [dict(segment)]
        boundary_gap = abs(part_end - region["end"]) if index < len(regions) - 1 else 0.0
        punctuation_bonus = 0.15 if part_text[-1:] in CHINESE_PUNCTUATION else 0.0
        confidence = max(0.0, min(1.0, 1.0 - boundary_gap / max(boundary_tolerance_sec, 0.001) + punctuation_bonus))
        updated = dict(segment)
        updated.update(
            {
                "start": round(part_start, 2),
                "end": round(part_end, 2),
                "text": part_text,
                "split_method": "alignment_semantic",
                "original_segment_id": segment.get("index"),
                "split_confidence": round(confidence, 3),
                "split_reason": "aligned_units_near_vad_boundary",
            }
        )
        consumed += part_text
        parts.append(updated)

    if "".join(consumed.split()) != "".join(text.split()):
        return [dict(segment)]
    return parts


def split_long_segments_with_alignment(
    raw_segments: list[dict[str, Any]],
    speech_regions: list[dict[str, Any]],
    aligned_units: list[dict[str, Any]],
    *,
    max_segment_seconds: float = 20.0,
) -> list[dict[str, Any]]:
    split: list[dict[str, Any]] = []
    for segment in raw_segments:
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", start) or start)
        if end - start <= max_segment_seconds:
            split.append(segment)
            continue
        parts = split_segment_semantically(segment, speech_regions, aligned_units)
        if len(parts) == 1 and parts[0].get("split_method") != "alignment_semantic":
            split.append(segment)
        else:
            split.extend(parts)
    return split
