from __future__ import annotations

from typing import Any

CHINESE_PUNCTUATION = "。！？；：，、"
SENTENCE_END_PUNCTUATION = set("。！？；.!?;")
# ASR groups flush at 6s; splitters should run below that so alignment/VAD boundaries apply.
MAX_SEGMENT_SPLIT_SECONDS = 5.0
MAX_MERGED_SEGMENT_SECONDS = 8.0
# Prefer pause-based cuts so Chinese (often unpunctuated) does not mid-cut phrases.
TARGET_SEGMENT_SECONDS = 4.5
MIN_PAUSE_SPLIT_SEC = 0.22
SOFT_PAUSE_SPLIT_SEC = 0.16
MIN_PART_SECONDS = 1.35
MIN_PART_CHARS = 8
# Common 2-char tokens that forced-aligner sometimes pauses inside.
NEVER_SPLIT_BIGRAMS = {
    "笑容",
    "百姓",
    "妖魔",
    "大人",
    "武学",
    "开门",
    "平静",
    "冷静",
    "吸收",
    "完毕",
    "狗妖",
    "开制",
    "入境",
    "出境",
    "镇魔",
    "魔司",
    "衙门",
    "差役",
    "手指",
    "手掌",
    "抹布",
    "喘气",
}
CLAUSE_START_CHARS = set("到为什他在若还不哪这可们因然其却而只先再又更还对把被让从把给")


def ends_with_sentence_punctuation(text: str) -> bool:
    stripped = str(text or "").strip()
    return bool(stripped) and stripped[-1] in SENTENCE_END_PUNCTUATION


def merge_incomplete_sentence_segments(
    segments: list[dict[str, Any]],
    *,
    max_gap_sec: float = 0.75,
    max_merged_duration_sec: float = MAX_MERGED_SEGMENT_SECONDS,
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
        merged_duration = float(segment.get("end", 0.0) or 0.0) - float(previous.get("start", 0.0) or 0.0)
        if (
            previous_text
            and not ends_with_sentence_punctuation(previous_text)
            and -0.05 <= gap <= max_gap_sec
            and merged_duration <= max_merged_duration_sec
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


def _units_overlapping_segment(
    segment: dict[str, Any],
    aligned_units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    start = float(segment.get("start", 0.0) or 0.0)
    end = float(segment.get("end", start) or start)
    return [
        unit
        for unit in aligned_units
        if float(unit.get("end", unit.get("start", 0.0)) or 0.0) > start
        and float(unit.get("start", 0.0) or 0.0) < end
    ]


def _best_pause_split_index(
    units: list[dict[str, Any]],
    *,
    segment_start: float,
    segment_end: float,
    min_pause_sec: float,
    min_part_sec: float,
    min_part_chars: int,
    soft_pause_sec: float = SOFT_PAUSE_SPLIT_SEC,
) -> int | None:
    """Return index of the unit that should start the right-hand part, or None."""
    if len(units) < 4:
        return None
    duration = max(0.0, segment_end - segment_start)
    midpoint = segment_start + duration / 2.0

    def score_candidates(threshold: float) -> int | None:
        best_index: int | None = None
        best_score = float("-inf")
        for index in range(1, len(units)):
            left = units[:index]
            right = units[index:]
            left_text = "".join(str(unit.get("text") or "") for unit in left).strip()
            right_text = "".join(str(unit.get("text") or "") for unit in right).strip()
            if len(left_text) < min_part_chars or len(right_text) < min_part_chars:
                continue
            bigram = f"{left_text[-1]}{right_text[0]}"
            if bigram in NEVER_SPLIT_BIGRAMS:
                continue
            left_end = float(left[-1].get("end", left[-1].get("start", 0.0)) or 0.0)
            right_start = float(right[0].get("start", 0.0) or 0.0)
            gap = right_start - left_end
            left_dur = left_end - segment_start
            right_dur = segment_end - right_start
            if gap < threshold or left_dur < min_part_sec or right_dur < min_part_sec:
                continue
            distance = abs(left_end - midpoint)
            punct_bonus = 0.5 if left_text[-1:] in CHINESE_PUNCTUATION else 0.0
            clause_bonus = 0.25 if right_text[0:1] in CLAUSE_START_CHARS else 0.0
            # Prefer larger real pauses; midpoint is secondary.
            score = gap * 8.0 - distance / max(duration, 0.001) + punct_bonus + clause_bonus
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    hard = score_candidates(min_pause_sec)
    if hard is not None:
        return hard
    return score_candidates(soft_pause_sec)


def _split_one_segment_at_pause(
    segment: dict[str, Any],
    aligned_units: list[dict[str, Any]],
    *,
    min_pause_sec: float,
    min_part_sec: float,
    min_part_chars: int,
) -> list[dict[str, Any]] | None:
    units = _units_overlapping_segment(segment, aligned_units)
    if len(units) < 4:
        return None
    start = float(segment.get("start", 0.0) or 0.0)
    end = float(segment.get("end", start) or start)
    split_at = _best_pause_split_index(
        units,
        segment_start=start,
        segment_end=end,
        min_pause_sec=min_pause_sec,
        min_part_sec=min_part_sec,
        min_part_chars=min_part_chars,
    )
    if split_at is None:
        return None

    left_units = units[:split_at]
    right_units = units[split_at:]
    left_text = "".join(str(unit.get("text") or "") for unit in left_units).strip()
    right_text = "".join(str(unit.get("text") or "") for unit in right_units).strip()
    if not left_text or not right_text:
        return None

    left_end = float(left_units[-1].get("end", left_units[-1].get("start", start)) or start)
    right_start = float(right_units[0].get("start", left_end) or left_end)
    boundary = round((left_end + right_start) / 2.0, 2)
    boundary = max(start + min_part_sec, min(end - min_part_sec, boundary))

    left = dict(segment)
    left.update(
        {
            "start": round(start, 2),
            "end": boundary,
            "text": left_text,
            "split_method": "alignment_pause",
            "original_segment_id": segment.get("original_segment_id", segment.get("index")),
            "split_reason": "pause_gap_near_midpoint",
        }
    )
    right = dict(segment)
    right.update(
        {
            "start": boundary,
            "end": round(end, 2),
            "text": right_text,
            "split_method": "alignment_pause",
            "original_segment_id": segment.get("original_segment_id", segment.get("index")),
            "split_reason": "pause_gap_near_midpoint",
        }
    )
    return [left, right]


def split_segments_by_alignment_pauses(
    raw_segments: list[dict[str, Any]],
    aligned_units: list[dict[str, Any]],
    *,
    target_seconds: float = TARGET_SEGMENT_SECONDS,
    min_pause_sec: float = MIN_PAUSE_SPLIT_SEC,
    min_part_sec: float = MIN_PART_SECONDS,
    min_part_chars: int = MIN_PART_CHARS,
) -> list[dict[str, Any]]:
    """Split overlong segments at the best intra-segment pause between aligned units.

    Chinese ASR often has no punctuation, so hard time/char cuts land mid-phrase.
    Using forced-aligner pauses yields boundaries closer to natural clause breaks.
    """
    if not raw_segments or not aligned_units:
        return raw_segments

    output: list[dict[str, Any]] = []
    queue = [dict(segment) for segment in raw_segments]
    while queue:
        segment = queue.pop(0)
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", start) or start)
        duration = end - start
        text = str(segment.get("text") or "").strip()
        if duration <= target_seconds or len(text) < min_part_chars * 2:
            output.append(segment)
            continue
        parts = _split_one_segment_at_pause(
            segment,
            aligned_units,
            min_pause_sec=min_pause_sec,
            min_part_sec=min_part_sec,
            min_part_chars=min_part_chars,
        )
        if not parts:
            output.append(segment)
            continue
        # Re-check both sides; may split again if still long.
        queue[0:0] = parts
    return output


def consolidate_short_segments(
    raw_segments: list[dict[str, Any]],
    *,
    min_keep_sec: float = 1.6,
    max_merged_sec: float = 5.2,
) -> list[dict[str, Any]]:
    """Merge tiny pause fragments that are too short to stand alone as dub units."""
    if not raw_segments:
        return []
    ordered = [dict(item) for item in sorted(raw_segments, key=lambda item: float(item.get("start", 0.0) or 0.0))]

    def _merge_into(target: dict[str, Any], source: dict[str, Any]) -> None:
        target["text"] = str(target.get("text") or "") + str(source.get("text") or "")
        target["end"] = round(float(source.get("end", target.get("end", 0.0)) or 0.0), 2)
        target["split_method"] = target.get("split_method") or source.get("split_method") or "consolidated"
        target["split_reason"] = "merge_tiny_pause_fragment"

    changed = True
    while changed:
        changed = False
        merged: list[dict[str, Any]] = []
        index = 0
        while index < len(ordered):
            current = ordered[index]
            cur_dur = float(current.get("end", 0.0) or 0.0) - float(current.get("start", 0.0) or 0.0)
            if cur_dur >= min_keep_sec or ends_with_sentence_punctuation(str(current.get("text") or "")):
                merged.append(current)
                index += 1
                continue

            prev = merged[-1] if merged else None
            nxt = ordered[index + 1] if index + 1 < len(ordered) else None
            merged_prev = (
                float(current.get("end", 0.0) or 0.0) - float(prev.get("start", 0.0) or 0.0)
                if prev is not None
                else None
            )
            merged_next = (
                float(nxt.get("end", 0.0) or 0.0) - float(current.get("start", 0.0) or 0.0)
                if nxt is not None
                else None
            )
            gap_prev = (
                float(current.get("start", 0.0) or 0.0) - float(prev.get("end", 0.0) or 0.0)
                if prev is not None
                else 999.0
            )
            gap_next = (
                float(nxt.get("start", 0.0) or 0.0) - float(current.get("end", 0.0) or 0.0)
                if nxt is not None
                else 999.0
            )

            prefer_prev = (
                prev is not None
                and gap_prev <= 0.35
                and merged_prev is not None
                and merged_prev <= max_merged_sec
            )
            prefer_next = (
                nxt is not None
                and gap_next <= 0.35
                and merged_next is not None
                and merged_next <= max_merged_sec
            )
            if prefer_prev and prefer_next:
                # Prefer the shorter combined result to keep clause size controlled.
                prefer_prev = merged_prev <= merged_next

            if prefer_prev and prev is not None:
                _merge_into(prev, current)
                changed = True
                index += 1
                continue
            if prefer_next and nxt is not None:
                _merge_into(current, nxt)
                merged.append(current)
                changed = True
                index += 2
                continue

            merged.append(current)
            index += 1
        ordered = merged
    return ordered
