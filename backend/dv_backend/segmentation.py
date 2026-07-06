from __future__ import annotations

from typing import Any

CHINESE_PUNCTUATION = "。！？；：，、"
SENTENCE_END_PUNCTUATION = set("。！？；.!?;")
# ASR groups flush at 12s; splitters should run below that so alignment/VAD boundaries apply.
MAX_SEGMENT_SPLIT_SECONDS = 9.0


def ends_with_sentence_punctuation(text: str) -> bool:
    stripped = str(text or "").strip()
    return bool(stripped) and stripped[-1] in SENTENCE_END_PUNCTUATION


def merge_incomplete_sentence_segments(
    segments: list[dict[str, Any]],
    *,
    max_gap_sec: float = 0.75,
) -> list[dict[str, Any]]:
    """Merge consecutive segments when the previous chunk does not end a full sentence."""
    if not segments:
        return []

    ordered = sorted(segments, key=lambda item: float(item.get("start", 0.0) or 0.0))
    merged: list[dict[str, Any]] = [dict(ordered[0])]
    for segment in ordered[1:]:
        previous = merged[-1]
        previous_text = str(previous.get("text") or "").strip()
        gap = float(segment.get("start", 0.0) or 0.0) - float(previous.get("end", 0.0) or 0.0)
        if (
            previous_text
            and not ends_with_sentence_punctuation(previous_text)
            and -0.05 <= gap <= max_gap_sec
        ):
            previous["text"] = previous_text + str(segment.get("text") or "")
            previous["end"] = round(float(segment.get("end", previous.get("end", 0.0)) or 0.0), 2)
            continue
        merged.append(dict(segment))
    return merged


def allocate_text_across_regions(text: str, regions: list[dict[str, float]]) -> list[str]:
    """Split text across VAD regions, preferring sentence punctuation near duration ratios."""
    cleaned = text.strip()
    if not cleaned or not regions:
        return []

    total_duration = sum(region["end"] - region["start"] for region in regions)
    if total_duration <= 0:
        return [cleaned]

    chunks: list[str] = []
    cursor = 0
    for index, region in enumerate(regions):
        if index == len(regions) - 1:
            chunk = cleaned[cursor:].strip()
            if chunk:
                chunks.append(chunk)
            break

        ratio = (region["end"] - region["start"]) / total_duration
        target_cursor = max(cursor + 1, min(len(cleaned), round(cursor + len(cleaned[cursor:]) * ratio)))
        best_cursor = target_cursor
        search_start = max(cursor + 1, target_cursor - 6)
        search_end = min(len(cleaned), target_cursor + 6)
        for position in range(search_end - 1, search_start - 1, -1):
            if cleaned[position] in SENTENCE_END_PUNCTUATION:
                best_cursor = position + 1
                break
        chunk = cleaned[cursor:best_cursor].strip()
        if chunk:
            chunks.append(chunk)
        cursor = best_cursor
    while len(chunks) < len(regions):
        chunks.append("")
    return chunks[: len(regions)]


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
    max_segment_seconds: float = MAX_SEGMENT_SPLIT_SECONDS,
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
