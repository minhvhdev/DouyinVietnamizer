from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


@dataclass(frozen=True)
class SparseDecision:
    use_sparse: bool
    reason: str
    speech_ratio: float


def _clean_regions(regions: list[dict[str, Any]], total_duration: float) -> list[dict[str, float]]:
    cleaned: list[dict[str, float]] = []
    for region in regions:
        start = max(0.0, float(region.get("start", 0.0) or 0.0))
        end = min(total_duration, float(region.get("end", start) or start))
        if end > start:
            cleaned.append({"start": start, "end": end})
    return sorted(cleaned, key=lambda item: item["start"])


def should_use_sparse_asr(
    speech_regions: list[dict[str, Any]],
    *,
    total_duration: float,
    min_silence_ratio: float,
) -> SparseDecision:
    if total_duration <= 0:
        return SparseDecision(False, "invalid_duration", 0.0)
    regions = _clean_regions(speech_regions, total_duration)
    if not regions:
        return SparseDecision(False, "no_speech_regions", 0.0)
    speech_duration = sum(region["end"] - region["start"] for region in regions)
    speech_ratio = min(1.0, speech_duration / total_duration)
    silence_ratio = 1.0 - speech_ratio
    tiny_fragments = sum(1 for region in regions if region["end"] - region["start"] < 0.25)
    if tiny_fragments > max(12, len(regions) // 2):
        return SparseDecision(False, "fragmented_vad", speech_ratio)
    if any(region["end"] - region["start"] > 60.0 for region in regions):
        return SparseDecision(False, "long_speech_region", speech_ratio)
    if silence_ratio < min_silence_ratio:
        return SparseDecision(False, "low_silence_ratio", speech_ratio)
    return SparseDecision(True, "ok", speech_ratio)


def build_sparse_chunks(
    speech_regions: list[dict[str, Any]],
    *,
    total_duration: float,
    merge_gap_sec: float,
    padding_sec: float,
    max_chunk_sec: float,
) -> list[dict[str, float]]:
    regions = _clean_regions(speech_regions, total_duration)
    merged: list[dict[str, float]] = []
    for region in regions:
        if merged and region["start"] - merged[-1]["end"] <= merge_gap_sec:
            merged[-1]["end"] = max(merged[-1]["end"], region["end"])
        else:
            merged.append(dict(region))

    chunks: list[dict[str, float]] = []
    for region in merged:
        start = max(0.0, region["start"] - padding_sec)
        end = min(total_duration, region["end"] + padding_sec)
        while end - start > max_chunk_sec:
            chunk_end = start + max_chunk_sec
            chunks.append({"source_start": round(start, 3), "source_end": round(chunk_end, 3)})
            start = chunk_end
        chunks.append({"source_start": round(start, 3), "source_end": round(end, 3)})
    return chunks


def build_stitched_timeline(chunks: list[dict[str, float]]) -> list[dict[str, float]]:
    timeline: list[dict[str, float]] = []
    cursor = 0.0
    for chunk in chunks:
        source_start = float(chunk.get("source_start", 0.0) or 0.0)
        source_end = float(chunk.get("source_end", source_start) or source_start)
        duration = source_end - source_start
        if duration <= 0:
            continue
        stitched_start = cursor
        stitched_end = cursor + duration
        timeline.append({
            "source_start": round(source_start, 3),
            "source_end": round(source_end, 3),
            "stitched_start": round(stitched_start, 3),
            "stitched_end": round(stitched_end, 3),
        })
        cursor = stitched_end
    return timeline


def stitched_timeline_duration(timeline: list[dict[str, float]]) -> float:
    if not timeline:
        return 0.0
    return round(float(timeline[-1]["stitched_end"]), 3)


def _map_stitched_time(timeline: list[dict[str, float]], value: float, *, is_end: bool) -> float:
    if not timeline:
        return round(value, 3)

    if is_end:
        for span in timeline:
            if float(span["stitched_start"]) < value <= float(span["stitched_end"]):
                return round(float(span["source_start"]) + value - float(span["stitched_start"]), 3)
        if value <= float(timeline[0]["stitched_start"]):
            return round(float(timeline[0]["source_start"]), 3)
        return round(float(timeline[-1]["source_end"]), 3)

    for span in timeline:
        if float(span["stitched_start"]) <= value < float(span["stitched_end"]):
            return round(float(span["source_start"]) + value - float(span["stitched_start"]), 3)
    if value < float(timeline[0]["stitched_start"]):
        return round(float(timeline[0]["source_start"]), 3)
    return round(float(timeline[-1]["source_end"]), 3)


def map_stitched_segments_to_source(
    timeline: list[dict[str, float]],
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for segment in segments:
        start_value = float(segment.get("start", 0.0) or 0.0)
        end_value = float(segment.get("end", start_value) or start_value)
        updated = dict(segment)
        updated["start"] = _map_stitched_time(timeline, start_value, is_end=False)
        updated["end"] = _map_stitched_time(timeline, end_value, is_end=True)
        if updated["end"] > updated["start"]:
            mapped.append(updated)
    return mapped


def rebase_sparse_segments(chunk: dict[str, float], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offset = float(chunk["source_start"])
    rebased: list[dict[str, Any]] = []
    for segment in segments:
        updated = dict(segment)
        updated["start"] = round(offset + float(segment.get("start", 0.0) or 0.0), 3)
        updated["end"] = round(offset + float(segment.get("end", segment.get("start", 0.0)) or 0.0), 3)
        if updated["end"] > updated["start"]:
            rebased.append(updated)
    return rebased


def merge_overlapping_segments(segments: list[dict[str, Any]], *, similarity_threshold: float = 0.88) -> list[dict[str, Any]]:
    ordered = sorted(segments, key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))
    merged: list[dict[str, Any]] = []
    for segment in ordered:
        if not merged:
            merged.append(dict(segment))
            continue
        previous = merged[-1]
        overlap = min(float(previous["end"]), float(segment["end"])) - max(float(previous["start"]), float(segment["start"]))
        similarity = SequenceMatcher(None, str(previous.get("text") or ""), str(segment.get("text") or "")).ratio()
        if overlap > 0 and similarity >= similarity_threshold:
            previous["end"] = max(float(previous["end"]), float(segment["end"]))
            if len(str(segment.get("text") or "")) > len(str(previous.get("text") or "")):
                previous["text"] = segment.get("text")
            continue
        merged.append(dict(segment))
    return merged
