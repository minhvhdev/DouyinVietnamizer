"""Tests for translation candidate LLM parser."""

from __future__ import annotations

from dv_backend.translation_candidate_llm import parse_candidate_batches, parse_candidate_batches_failsoft


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


def json_payload(items: list[dict]) -> str:
    import json

    return json.dumps(items, ensure_ascii=False)
