"""LLM prompt and response parsing for timing-aware translation candidates."""

from __future__ import annotations

import json
import re
from typing import Any

CANDIDATE_STYLES = ("natural", "compact", "very_compact", "expanded")


HANGING_ENDINGS = (
    "có hay không",
    "bởi vì",
    "nếu",
    "nhưng",
    "và",
    "để",
    "của",
    "với",
    "một",
)


def looks_like_fragment_spill(
    prev_text: str,
    next_text: str,
    previous_source: str | None = None,
    next_source: str | None = None,
) -> bool:
    """Backward-compatible wrapper; detector lives in translation_rebalance."""
    from .translation_rebalance import looks_like_fragment_spill as _detect

    return _detect(
        prev_text,
        next_text,
        previous_source=previous_source,
        next_source=next_source,
    )

def assert_timing_immutable(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> None:
    """Ensure translate output never mutates slot count or start/end."""
    if len(before) != len(after):
        raise AssertionError(
            f"segment count changed: before={len(before)} after={len(after)}"
        )
    for index, (src, dst) in enumerate(zip(before, after, strict=True)):
        src_id = src.get("index", index)
        dst_id = dst.get("index", index)
        if src_id != dst_id:
            raise AssertionError(f"segment_id changed at {index}: {src_id} -> {dst_id}")
        for key in ("start", "end"):
            if src.get(key) is None and dst.get(key) is None:
                continue
            if float(src.get(key) or 0.0) != float(dst.get(key) or 0.0):
                raise AssertionError(
                    f"{key} changed for segment {src_id}: {src.get(key)} -> {dst.get(key)}"
                )


def build_candidate_items(
    segments: list[dict[str, Any]],
    texts: list[str],
    *,
    timing_profiles: list[dict[str, float]],
    speaking_rate_wps: float,
    prev_context_count: int = 1,
    next_context_count: int = 1,
) -> list[dict[str, Any]]:
    slot_count = len(texts)
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
                "slot_index": index,
                "slot_count": slot_count,
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
    from .translation_duration import timing_translate_prompt_rules

    transcript_lines = [
        f"[{item.get('segment_id')}] {item.get('source_text') or ''}"
        for item in items
    ]
    transcript_block = "\n".join(transcript_lines)

    return (
        f"Translate each timing slot from {source} to {target} for natural dubbing.\n"
        "\n"
        "## Video-level understanding (internal only)\n"
        "Before writing any candidate, silently infer from the FULL transcript below: "
        "short synopsis, recurring entities/names, character relationships, tone/genre, "
        "and a canonical rendering for each entity. Do NOT emit that working memory in the JSON.\n"
        "\n"
        "## Full source transcript (ordered timing slots)\n"
        f"{transcript_block}\n"
        "\n"
        "## Hard constraints\n"
        f"- Return exactly {len(items)} output items in the same order as input.\n"
        "- Each input item is a TIMING SLOT, not a sentence boundary.\n"
        "- Never change segment_id order, never invent/drop slots, never return start/end.\n"
        "- You MAY redistribute wording across adjacent slots so each spoken line is a complete thought.\n"
        "- Do not add facts absent from the source; do not drop names, numbers, negation, or core actions.\n"
        "\n"
        "## Terminology locking\n"
        "- Pick one canonical rendering per entity/proper name and reuse it across ALL slots.\n"
        "- Do not mix Hán-Việt / phonetic / translated forms of the same name in one job.\n"
        "- Pronouns may vary with speaker/relation, but not randomly.\n"
        "\n"
        "## Complete-thought rules\n"
        "- Forbidden patterns: ending a slot with hanging openers like "
        "'có hay không' / 'bởi vì' / 'nếu' / 'nhưng' / 'và' / 'để' / 'của' / 'với' / 'một' "
        "when the clause continues in the next slot; starting the next slot with leftover complements, "
        "ellipsis (...), or words that only make sense when glued to the previous slot.\n"
        "- Prefer complete speakable lines. If one semantic sentence spans multiple slots, "
        "split into complete thoughts that fit the existing slots (2 or 3 lines OK) — "
        "never increase or decrease slot count.\n"
        "\n"
        "## Few-shot (bad → good)\n"
        "BAD:\n"
        "  slot0: nhưng cuối cùng hắn đã hiểu một chuyện: có hay không...\n"
        "  slot1: ...cách giải quyết khác? Câu trả lời là không.\n"
        "GOOD (3 complete thoughts, same 2–3 slots as available — here shown as 3 lines of meaning "
        "packed into the existing slots without changing timing):\n"
        "  slot0: Nhưng cuối cùng hắn đã hiểu ra một chuyện.\n"
        "  slot1: Có hay không cách giải quyết khác? Câu trả lời là không.\n"
        "Also acceptable if syllable budgets allow packing three thoughts into the same slot count.\n"
        "\n"
        f"## Candidates (up to {candidate_count} per slot)\n"
        "- natural: fluent dubbing line that is a complete thought\n"
        "- compact: shorter but still complete\n"
        "- very_compact: shortest acceptable (only if source is not trivial)\n"
        "- expanded: only when speech_target_duration_sec is long and a shorter line would feel cut off\n"
        "- Candidates must differ in concision, not just punctuation.\n"
        "\n"
        "## Timing / length\n"
        f"- {timing_translate_prompt_rules()}\n"
        "- Preserve ALL numbers, percentages, dates, currency amounts, and product/model names exactly.\n"
        "- Preserve negation (không/chưa/đừng) and comparisons (hơn/kém/lớn nhất).\n"
        "\n"
        "Return JSON array in the same order as input. Each element:\n"
        '{"segment_id": number, "candidates": [{"text": string, "style": string, "meaning_notes": string[]}]}\n'
        "Input slots JSON:\n"
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


def extract_json_payload(text: str) -> Any:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    else:
        array_start = cleaned.find("[")
        array_end = cleaned.rfind("]")
        object_start = cleaned.find("{")
        object_end = cleaned.rfind("}")
        if array_start >= 0 and array_end > array_start and (
            object_start < 0 or array_start <= object_start
        ):
            cleaned = cleaned[array_start : array_end + 1]
        elif object_start >= 0 and object_end > object_start:
            cleaned = cleaned[object_start : object_end + 1]
    return json.loads(cleaned)


def _extract_json_payload(text: str) -> Any:
    return extract_json_payload(text)


def build_fragment_repair_prompt(
    cluster_payloads: list[dict[str, Any]],
    *,
    source: str,
    target: str,
) -> str:
    """Prompt to rebalance existing translations across immutable timing slots."""
    return (
        f"You are repairing {target} dubbing translations that were split badly across timing slots "
        f"(source language: {source}).\n"
        "\n"
        "## Task\n"
        "This is NOT a fresh independent translation. Rebalance the CURRENT Vietnamese lines so each "
        "mutable timing slot is a complete speakable thought. Keep meaning from the Chinese sources "
        "and the existing Vietnamese wording when possible.\n"
        "\n"
        "## Hard constraints\n"
        "- Return EXACTLY the mutable segments listed for each cluster — same segment_id and order.\n"
        "- Do NOT add/remove/reorder segments. Do NOT return start/end/timing fields.\n"
        "- Do NOT modify or return context_before / context_after (read-only).\n"
        "- Preserve names, numbers, negation, and core actions across the cluster as a whole.\n"
        "- Keep terminology consistent with the current translations.\n"
        "\n"
        "## Complete-thought rules\n"
        "- Forbidden: ending a slot mid-clause (e.g. '...có hay không' / hanging ':' / '...') "
        "when the next mutable slot continues the same clause.\n"
        "- Forbidden: starting a slot with leftover complements or ellipsis that only make sense "
        "glued to the previous slot.\n"
        "- You MAY redistribute words across mutable slots inside a cluster.\n"
        "- Slot count is fixed; packing 2 or 3 complete thoughts into those slots is OK.\n"
        "\n"
        "## Few-shot\n"
        "BAD mutable pair:\n"
        "  seg0: nhưng cuối cùng hắn đã hiểu một chuyện: có hay không...\n"
        "  seg1: ...cách giải quyết khác? Câu trả lời là không.\n"
        "GOOD (same 2 slots):\n"
        "  seg0: Nhưng cuối cùng hắn đã hiểu ra một chuyện.\n"
        "  seg1: Có hay không cách giải quyết khác? Câu trả lời là không.\n"
        "\n"
        "## Output JSON only\n"
        '{"clusters":[{"cluster_id":0,"segments":[{"segment_id":0,"translation":"..."},'
        '{"segment_id":1,"translation":"..."}]}]}\n'
        "\n"
        "Clusters to repair:\n"
        f"{json.dumps(cluster_payloads, ensure_ascii=False)}"
    )


def parse_fragment_repair_response(text: str) -> Any:
    return extract_json_payload(text)


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
