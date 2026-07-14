"""LLM prompt and response parsing for timing-aware translation candidates."""

from __future__ import annotations

import json
import re
from typing import Any

CANDIDATE_STYLES = ("natural", "compact", "very_compact", "expanded")


def build_candidate_items(
    segments: list[dict[str, Any]],
    texts: list[str],
    *,
    timing_profiles: list[dict[str, float]],
    speaking_rate_wps: float,
    prev_context_count: int = 1,
    next_context_count: int = 1,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, (segment, text) in enumerate(zip(segments, texts, strict=True)):
        profile = timing_profiles[index] if index < len(timing_profiles) else {}
        prev_ctx = [
            str(segments[j].get("text") or "")
            for j in range(max(0, index - prev_context_count), index)
        ]
        next_ctx = [
            str(segments[j].get("text") or "")
            for j in range(index + 1, min(len(segments), index + 1 + next_context_count))
        ]
        items.append(
            {
                "segment_id": segment.get("index", index),
                "source_text": text,
                "context_before": prev_ctx,
                "context_after": next_ctx,
                "speech_target_duration_sec": profile.get("speech_target_duration"),
                "hard_max_duration_sec": profile.get("hard_max_duration"),
                "voice_speaking_rate_estimate_wps": round(float(speaking_rate_wps), 2),
                "target_vi_syllables": segment.get("target_vi_syllables"),
                "target_vi_syllable_range": segment.get("target_vi_syllable_range"),
            }
        )
    return items


def build_candidate_translation_prompt(
    items: list[dict[str, Any]],
    *,
    source: str,
    target: str,
    candidate_count: int = 3,
) -> str:
    return (
        f"Translate each item from {source} to {target} for natural dubbing.\n"
        f"For each segment_id, return up to {candidate_count} candidate translations with different concision:\n"
        "- natural: default fluent dubbing line\n"
        "- compact: shorter but complete\n"
        "- very_compact: shortest acceptable (only if source is not trivial)\n"
        "- expanded: only when speech_target_duration_sec is long and a shorter line would feel cut off\n"
        "Rules:\n"
        "- Preserve ALL numbers, percentages, dates, currency amounts, and product/model names exactly.\n"
        "- Preserve negation (không/chưa/đừng) and comparisons (hơn/kém/lớn nhất).\n"
        "- Preserve proper nouns and named entities.\n"
        "- Do not add information not present in the source.\n"
        "- Do not drop important clauses or actions.\n"
        "- Candidates must differ in concision, not just punctuation.\n"
        "- Use speech_target_duration_sec and hard_max_duration_sec as timing hints only.\n"
        "Return JSON array in the same order as input. Each element:\n"
        '{"segment_id": number, "candidates": [{"text": string, "style": string, "meaning_notes": string[]}]}\n'
        f"{json.dumps(items, ensure_ascii=False)}"
    )


def _normalize_candidate(raw: dict[str, Any]) -> dict[str, Any] | None:
    text = str(raw.get("text") or "").strip()
    if not text:
        return None
    style = str(raw.get("style") or "natural").strip().lower()
    if style not in CANDIDATE_STYLES:
        style = "natural"
    notes = raw.get("meaning_notes") or []
    if not isinstance(notes, list):
        notes = []
    return {
        "text": text,
        "style": style,
        "meaning_notes": [str(note) for note in notes if str(note).strip()],
        "candidate_source": "llm",
    }


def _extract_json_payload(text: str) -> Any:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    else:
        array_start = cleaned.find("[")
        array_end = cleaned.rfind("]")
        if array_start >= 0 and array_end > array_start:
            cleaned = cleaned[array_start : array_end + 1]
    return json.loads(cleaned)


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for candidate in candidates:
        key = re.sub(r"\s+", " ", candidate["text"].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def parse_candidate_batches(
    text: str,
    *,
    expected_count: int,
    max_candidates: int = 4,
) -> list[list[dict[str, Any]]]:
    data = _extract_json_payload(text)
    if isinstance(data, dict):
        for key in ("segments", "items", "results"):
            nested = data.get(key)
            if isinstance(nested, list):
                data = nested
                break
        else:
            data = [data]

    if not isinstance(data, list):
        raise ValueError("Candidate translation response must be a JSON array.")

    batches: list[list[dict[str, Any]]] = []
    for item in data:
        if isinstance(item, str):
            normalized = _normalize_candidate({"text": item, "style": "natural"})
            batches.append([normalized] if normalized else [])
            continue
        if not isinstance(item, dict):
            batches.append([])
            continue
        raw_candidates = item.get("candidates")
        if not isinstance(raw_candidates, list):
            text_value = item.get("text") or item.get("translation")
            normalized = _normalize_candidate({"text": text_value, "style": "natural"})
            batches.append([normalized] if normalized else [])
            continue
        normalized_list = _dedupe_candidates([
            candidate
            for candidate in (_normalize_candidate(raw) for raw in raw_candidates if isinstance(raw, dict))
            if candidate
        ])
        batches.append(normalized_list[:max_candidates])

    if len(batches) != expected_count:
        raise ValueError("Candidate translation response length mismatch.")
    return batches


def parse_candidate_batches_failsoft(
    text: str,
    *,
    expected_count: int,
    fallback_texts: list[str] | None = None,
    max_candidates: int = 4,
) -> tuple[list[list[dict[str, Any]]], str | None]:
    try:
        return parse_candidate_batches(text, expected_count=expected_count, max_candidates=max_candidates), None
    except (ValueError, json.JSONDecodeError) as error:
        warning = f"candidate_parse_fallback:{error}"
        batches: list[list[dict[str, Any]]] = []
        for index in range(expected_count):
            fallback = (fallback_texts[index] if fallback_texts and index < len(fallback_texts) else "").strip()
            if fallback:
                batches.append([{"text": fallback, "style": "natural", "meaning_notes": [], "candidate_source": "parse_fallback"}])
            else:
                batches.append([])
        return batches, warning
