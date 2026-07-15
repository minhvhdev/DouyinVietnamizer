"""Tests for translation candidate LLM parser and Phase-1 prompt contract."""

from __future__ import annotations

import json

import pytest

from dv_backend.translation_candidate_llm import (
    assert_timing_immutable,
    build_candidate_items,
    build_candidate_translation_prompt,
    looks_like_fragment_spill,
    parse_candidate_batches,
    parse_candidate_batches_failsoft,
)
from dv_backend.translation_duration import timing_translate_prompt_rules


def test_parse_markdown_json_fence() -> None:
    raw = '```json\n[{"segment_id": 0, "candidates": [{"text": "Xin chào", "style": "natural"}]}]\n```'
    batches = parse_candidate_batches(raw, expected_count=1)
    assert batches[0][0]["text"] == "Xin chào"


def test_parse_duplicate_candidates_deduped() -> None:
    raw = json_payload([
        {"segment_id": 0, "candidates": [
            {"text": "A", "style": "natural"},
            {"text": "A", "style": "compact"},
        ]},
    ])
    batches = parse_candidate_batches(raw, expected_count=1)
    assert len(batches[0]) == 1


def test_failsoft_on_malformed_json() -> None:
    batches, warning = parse_candidate_batches_failsoft(
        "not json",
        expected_count=2,
        fallback_texts=["fallback one", "fallback two"],
    )
    assert warning is not None
    assert batches[0][0]["text"] == "fallback one"
    assert batches[1][0]["text"] == "fallback two"


def test_failsoft_on_length_mismatch() -> None:
    raw = json_payload([{"segment_id": 0, "candidates": [{"text": "Only one", "style": "natural"}]}])
    batches, warning = parse_candidate_batches_failsoft(
        raw,
        expected_count=2,
        fallback_texts=["a", "b"],
    )
    assert warning is not None
    assert len(batches) == 2


def test_candidate_prompt_requires_video_understanding_and_complete_thoughts() -> None:
    segments = [
        {"index": 0, "text": "但最终他明白了一件事：有没有", "target_vi_syllables": 10, "target_vi_syllable_range": [8, 12]},
        {"index": 1, "text": "别的解决办法？答案是没有。", "target_vi_syllables": 12, "target_vi_syllable_range": [10, 14]},
    ]
    texts = [s["text"] for s in segments]
    items = build_candidate_items(
        segments,
        texts,
        timing_profiles=[{"speech_target_duration": 3.0}, {"speech_target_duration": 3.5}],
        speaking_rate_wps=3.2,
    )
    prompt = build_candidate_translation_prompt(items, source="zh-CN", target="vi", candidate_count=3)

    assert "TIMING SLOT" in prompt or "timing slot" in prompt.lower()
    assert "Full source transcript" in prompt or "FULL transcript" in prompt or "Full source transcript" in prompt
    assert "canonical" in prompt.lower() or "Terminology locking" in prompt
    assert "có hay không" in prompt
    assert "redistribute" in prompt.lower() or "tái phân phối" in prompt.lower() or "adjacent" in prompt.lower()
    assert "Priority order" in timing_translate_prompt_rules()
    assert "no fewer than" in timing_translate_prompt_rules()
    assert "slot_count" in json.dumps(items)
    assert items[0]["slot_count"] == 2


def test_looks_like_fragment_spill_detects_user_example() -> None:
    assert looks_like_fragment_spill(
        "nhưng cuối cùng hắn đã hiểu một chuyện: có hay không...",
        "...cách giải quyết khác? Câu trả lời là không.",
    )
    assert not looks_like_fragment_spill(
        "Nhưng cuối cùng hắn đã hiểu ra một chuyện.",
        "Có hay không cách giải quyết khác? Câu trả lời là không.",
    )


def test_assert_timing_immutable_passes_and_fails() -> None:
    before = [{"index": 0, "start": 1.0, "end": 3.0}, {"index": 1, "start": 3.0, "end": 5.0}]
    after = [{"index": 0, "start": 1.0, "end": 3.0, "translation": "A"}, {"index": 1, "start": 3.0, "end": 5.0, "translation": "B"}]
    assert_timing_immutable(before, after)

    with pytest.raises(AssertionError, match="end changed"):
        assert_timing_immutable(before, [{"index": 0, "start": 1.0, "end": 3.1}, {"index": 1, "start": 3.0, "end": 5.0}])


def json_payload(items: list[dict]) -> str:
    return json.dumps(items, ensure_ascii=False)
