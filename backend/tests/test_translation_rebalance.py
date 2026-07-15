"""Phase 2: translation-time fragment rebalance."""

from __future__ import annotations

from typing import Any

import pytest

from dv_backend.translation_candidate_llm import (
    assert_timing_immutable,
    build_fragment_repair_prompt,
)
from dv_backend.translation_candidates import translate_segments_with_candidates
from dv_backend.translation_rebalance import (
    build_fragment_clusters,
    find_fragment_spill_pairs,
    parse_repair_clusters,
    rebalance_fragment_spills,
    validate_cluster_repair,
)


def _seg(
    index: int,
    text: str,
    translation: str,
    *,
    start: float,
    end: float,
    syllable_range: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "index": index,
        "text": text,
        "translation": translation,
        "start": start,
        "end": end,
        "target_vi_syllable_range": syllable_range or [4, 20],
        "timing_profile": {"speech_target_duration": end - start},
        "translation_candidates": [
            {"text": translation, "style": "natural", "meaning_notes": [], "candidate_source": "llm"}
        ],
        "selected_candidate_index": 0,
    }


def test_pair_becomes_cluster() -> None:
    translations = [
        "nhưng cuối cùng hắn đã hiểu một chuyện: có hay không...",
        "...cách giải quyết khác? Câu trả lời là không.",
        "Sau đó hắn tiếp tục hành trình.",
    ]
    pairs = find_fragment_spill_pairs(translations)
    assert pairs == [(0, 1)]
    clusters = build_fragment_clusters(translations)
    assert len(clusters) == 1
    assert clusters[0].mutable_indices == (0, 1)
    assert clusters[0].context_before_index is None
    assert clusters[0].context_after_index == 2


def test_no_repair_when_output_clean() -> None:
    segments = [
        _seg(0, "你好", "Xin chào mọi người.", start=0.0, end=2.0),
        _seg(1, "再见", "Tạm biệt nhé.", start=2.0, end=4.0),
    ]
    calls: list[Any] = []

    def repair_fn(payloads, *, source, target):
        calls.append(payloads)
        return {"clusters": []}

    diagnostics = rebalance_fragment_spills(
        segments,
        repair_fn=repair_fn,
        source_lang="zh-CN",
        target_lang="vi",
    )
    assert diagnostics["skipped"] is True
    assert diagnostics["repair_calls"] == 0
    assert calls == []


def test_one_request_for_multiple_clusters() -> None:
    segments = [
        _seg(0, "A1", "có hay không...", start=0.0, end=1.0),
        _seg(1, "A2", "...cách khác?", start=1.0, end=2.0),
        _seg(2, "B mid", "Giữa đoạn bình thường.", start=2.0, end=3.0),
        _seg(3, "C1", "bởi vì...", start=3.0, end=4.0),
        _seg(4, "C2", "...hắn muốn đi.", start=4.0, end=5.0),
    ]
    calls: list[Any] = []

    def repair_fn(payloads, *, source, target):
        calls.append(payloads)
        return {
            "clusters": [
                {
                    "cluster_id": 0,
                    "segments": [
                        {"segment_id": 0, "translation": "Có cách khác không?"},
                        {"segment_id": 1, "translation": "Câu hỏi đã rõ."},
                    ],
                },
                {
                    "cluster_id": 1,
                    "segments": [
                        {"segment_id": 3, "translation": "Bởi vì hắn muốn đi."},
                        {"segment_id": 4, "translation": "Đó là lý do duy nhất."},
                    ],
                },
            ]
        }

    diagnostics = rebalance_fragment_spills(
        segments,
        repair_fn=repair_fn,
        source_lang="zh-CN",
        target_lang="vi",
    )
    assert diagnostics["repair_calls"] == 1
    assert len(calls) == 1
    assert len(calls[0]) == 2


def test_user_example_repaired_at_runtime() -> None:
    segments = [
        _seg(
            0,
            "但最终他明白了一件事：有没有",
            "nhưng cuối cùng hắn đã hiểu một chuyện: có hay không...",
            start=1.0,
            end=3.0,
        ),
        _seg(
            1,
            "别的解决办法？答案是没有。",
            "...cách giải quyết khác? Câu trả lời là không.",
            start=3.0,
            end=5.5,
        ),
    ]
    snapshot = [{"index": s["index"], "start": s["start"], "end": s["end"]} for s in segments]

    def repair_fn(payloads, *, source, target):
        assert len(payloads) == 1
        return {
            "clusters": [
                {
                    "cluster_id": 0,
                    "segments": [
                        {"segment_id": 0, "translation": "Nhưng cuối cùng hắn đã hiểu ra một chuyện."},
                        {
                            "segment_id": 1,
                            "translation": "Có hay không cách giải quyết khác? Câu trả lời là không.",
                        },
                    ],
                }
            ]
        }

    diagnostics = rebalance_fragment_spills(
        segments,
        repair_fn=repair_fn,
        source_lang="zh-CN",
        target_lang="vi",
    )
    assert 0 in diagnostics["accepted_clusters"]
    assert segments[0]["translation"] == "Nhưng cuối cùng hắn đã hiểu ra một chuyện."
    assert "Có hay không cách giải quyết khác" in segments[1]["translation"]
    assert_timing_immutable(snapshot, segments)
    assert len(segments) == 2


def test_chain_three_segments_same_slot_count() -> None:
    segments = [
        _seg(0, "第一句", "nhưng cuối cùng hắn đã hiểu một chuyện: có hay không...", start=0.0, end=1.0),
        _seg(1, "第二句", "...cách giải quyết khác: câu trả lời", start=1.0, end=2.0),
        _seg(2, "第三句", "...là không.", start=2.0, end=3.0),
    ]

    def repair_fn(payloads, *, source, target):
        assert len(payloads[0]["mutable_segments"]) == 3
        return {
            "clusters": [
                {
                    "cluster_id": 0,
                    "segments": [
                        {"segment_id": 0, "translation": "Nhưng cuối cùng hắn đã hiểu ra một chuyện."},
                        {"segment_id": 1, "translation": "Có hay không cách giải quyết khác?"},
                        {"segment_id": 2, "translation": "Câu trả lời là không."},
                    ],
                }
            ]
        }

    diagnostics = rebalance_fragment_spills(
        segments,
        repair_fn=repair_fn,
        source_lang="zh-CN",
        target_lang="vi",
    )
    assert diagnostics["accepted_clusters"] == [0]
    assert len(segments) == 3
    assert segments[0]["start"] == 0.0 and segments[2]["end"] == 3.0


def test_reject_id_mismatch() -> None:
    segments = [
        _seg(0, "A", "có hay không...", start=0.0, end=1.0),
        _seg(1, "B", "...tiếp theo.", start=1.0, end=2.0),
    ]
    clusters = build_fragment_clusters([s["translation"] for s in segments])
    with pytest.raises(ValueError, match="segment_id mismatch"):
        parse_repair_clusters(
            {
                "clusters": [
                    {
                        "cluster_id": 0,
                        "segments": [
                            {"segment_id": 99, "translation": "Ok một."},
                            {"segment_id": 1, "translation": "Ok hai."},
                        ],
                    }
                ]
            },
            expected_clusters=clusters,
            segments=segments,
        )


def test_reject_semantic_regression() -> None:
    segments = [
        _seg(0, "价格是100元", "giá là 100...", start=0.0, end=1.0),
        _seg(1, "不是200", "...không phải 200.", start=1.0, end=2.0),
    ]
    clusters = build_fragment_clusters([s["translation"] for s in segments])
    repaired = {0: "Giá rất cao.", 1: "Không phải mức kia."}
    accepted, reason, _ = validate_cluster_repair(segments, clusters[0], repaired)
    assert accepted is False
    assert reason and "semantic" in reason
    assert accepted is False


def test_reject_boundary_spill() -> None:
    segments = [
        _seg(0, "CTX", "Phần trước hoàn chỉnh.", start=0.0, end=1.0),
        _seg(1, "A", "có hay không...", start=1.0, end=2.0),
        _seg(2, "B", "...cách khác?", start=2.0, end=3.0),
        _seg(3, "CTX2", "Phần sau hoàn chỉnh.", start=3.0, end=4.0),
    ]
    clusters = build_fragment_clusters([s["translation"] for s in segments])
    repaired = {
        1: "Anh ấy hỏi có hay không",
        2: "cách khác:",
    }
    accepted, reason, _ = validate_cluster_repair(segments, clusters[0], repaired)
    assert accepted is False
    assert reason in {"boundary_regression", "boundary_spill", "internal_spill_remaining", "score_not_improved"}


def test_timing_identity_immutable_after_repair() -> None:
    segments = [
        _seg(0, "A", "có hay không...", start=10.0, end=12.5),
        _seg(1, "B", "...được không?", start=12.5, end=14.0),
    ]
    snapshot = [{"index": s["index"], "start": s["start"], "end": s["end"]} for s in segments]

    def repair_fn(payloads, *, source, target):
        return {
            "clusters": [
                {
                    "cluster_id": 0,
                    "segments": [
                        {"segment_id": 0, "translation": "Anh ấy hỏi có được không?"},
                        {"segment_id": 1, "translation": "Câu trả lời vẫn chưa rõ."},
                    ],
                }
            ]
        }

    rebalance_fragment_spills(segments, repair_fn=repair_fn, source_lang="zh", target_lang="vi")
    assert_timing_immutable(snapshot, segments)
    assert [s["index"] for s in segments] == [0, 1]


def test_graceful_fallback_on_repair_exception() -> None:
    segments = [
        _seg(0, "A", "có hay không...", start=0.0, end=1.0),
        _seg(1, "B", "...tiếp.", start=1.0, end=2.0),
    ]
    original = [s["translation"] for s in segments]

    def repair_fn(payloads, *, source, target):
        raise RuntimeError("boom")

    diagnostics = rebalance_fragment_spills(
        segments,
        repair_fn=repair_fn,
        source_lang="zh",
        target_lang="vi",
    )
    assert "repair_call_failed" in diagnostics.get("error", "") or diagnostics.get("error") == "api_error"
    assert [s["translation"] for s in segments] == original


def test_orchestration_wires_single_repair_call() -> None:
    segments = [
        _seg(0, "A", "", start=0.0, end=1.0),
        _seg(1, "B", "", start=1.0, end=2.0),
    ]
    repair_calls: list[Any] = []

    def translate_fn(*args, **kwargs):
        raise AssertionError("single translate should not run")

    def translate_candidates_fn(*args, **kwargs):
        return [
            [{"text": "có hay không...", "style": "natural", "meaning_notes": [], "candidate_source": "llm"}],
            [{"text": "...cách khác?", "style": "natural", "meaning_notes": [], "candidate_source": "llm"}],
        ]

    def repair_fn(payloads, *, source, target):
        repair_calls.append(payloads)
        return {
            "clusters": [
                {
                    "cluster_id": 0,
                    "segments": [
                        {"segment_id": 0, "translation": "Có cách khác không?"},
                        {"segment_id": 1, "translation": "Đó là câu hỏi chính."},
                    ],
                }
            ]
        }

    translate_segments_with_candidates(
        {"timing_candidate_translation_enabled": True, "translation_backend": "gemini"},
        database=None,
        segments=segments,
        source_lang="zh-CN",
        target_lang="vi",
        translate_fn=translate_fn,
        translate_candidates_fn=translate_candidates_fn,
        repair_fragment_fn=repair_fn,
    )
    assert len(repair_calls) == 1
    assert segments[0]["translation"] == "Có cách khác không?"
    assert segments[0]["start"] == 0.0 and segments[1]["end"] == 2.0


def test_repair_prompt_is_rebalance_not_fresh_translate() -> None:
    prompt = build_fragment_repair_prompt(
        [{"cluster_id": 0, "mutable_segments": [{"segment_id": 0, "current_translation": "x"}]}],
        source="zh-CN",
        target="vi",
    )
    assert "NOT a fresh independent translation" in prompt or "rebalance" in prompt.lower()
    assert "có hay không" in prompt
    assert "cluster_id" in prompt


def test_detector_without_ellipsis_or_hanging_token() -> None:
    from dv_backend.translation_rebalance import looks_like_fragment_spill

    # Colon mid-thought without hanging token list / ellipsis on next
    assert looks_like_fragment_spill(
        "Cuối cùng hắn hiểu một chuyện:",
        "cách giải quyết khác hoàn toàn",
        previous_source="最后他明白一件事：",
        next_source="别的解决办法",
    )
    assert not looks_like_fragment_spill(
        "Hắn đã hiểu rõ mọi chuyện.",
        "Sau đó hắn tiếp tục đi.",
    )


def test_hanging_start_and_short_head_signals() -> None:
    from dv_backend.translation_rebalance import looks_like_fragment_spill

    assert looks_like_fragment_spill(
        "Hắn hỏi rằng",
        "cách giải quyết khác là gì",
    )
    assert looks_like_fragment_spill(
        "Vấn đề còn lại:",
        "việc đó",
    )
    assert not looks_like_fragment_spill(
        "Hắn hỏi rõ ràng.",
        "Câu trả lời rất ngắn gọn và đầy đủ ý.",
    )


def test_unbalanced_punctuation_positive_and_negative() -> None:
    from dv_backend.translation_rebalance import looks_like_fragment_spill

    assert looks_like_fragment_spill(
        'Hắn nói: "có hay không',
        'cách khác?"',
    )
    assert not looks_like_fragment_spill(
        "Danh sách gồm: táo, cam.",
        "Và thêm chuối vào cuối.",
    )


def test_digit_identifier_not_semantic_number() -> None:
    from dv_backend.semantic_safeguards import evaluate_semantic_safeguards, extract_semantic_numbers

    assert extract_semantic_numbers("S0 seg_02 cluster1") == []
    assert "100" in extract_semantic_numbers("có 100 người và 10%")
    result = evaluate_semantic_safeguards(
        "Xin chào mọi người.",
        source_text="S0 hello",
        reference_text="S0 xin chào",
    )
    assert result["critical_violation"] is False
    assert "rejected_missing_number" not in result["rejection_reasons"]


def test_should_accept_repair_priority() -> None:
    from dv_backend.translation_rebalance import RepairMetrics, should_accept_repair

    old = RepairMetrics(internal_spills=1, boundary_spills=0, syllable_penalty=0)
    assert should_accept_repair(old, RepairMetrics(0, 0, 2))
    assert not should_accept_repair(old, RepairMetrics(1, 0, 0))
    assert not should_accept_repair(old, RepairMetrics(0, 1, 0))
    assert not should_accept_repair(old, RepairMetrics(0, 0, 0, semantic_critical=True))
    assert not should_accept_repair(old, RepairMetrics(0, 0, 9))


def test_single_path_skips_fragment_rebalance() -> None:
    segments = [
        _seg(0, "A", "", start=0.0, end=1.0),
        _seg(1, "B", "", start=1.0, end=2.0),
    ]
    repair_calls: list[Any] = []

    def translate_fn(*args, **kwargs):
        return ["có hay không...", "...cách khác?"]

    def repair_fn(payloads, *, source, target):
        repair_calls.append(payloads)
        return {"clusters": []}

    translate_segments_with_candidates(
        {"timing_candidate_translation_enabled": False, "translation_backend": "gemini"},
        database=None,
        segments=segments,
        source_lang="zh-CN",
        target_lang="vi",
        translate_fn=translate_fn,
        translate_candidates_fn=None,
        repair_fragment_fn=repair_fn,
    )
    assert repair_calls == []
    assert segments[0]["translation"] == "có hay không..."


def test_adapter_style_gemini_and_openai_orchestration_contract() -> None:
    """Simulate Gemini/OpenAI adapter contracts: first-pass + one repair."""
    for backend in ("gemini", "openai"):
        segments = [
            _seg(0, "有没有", "", start=0.0, end=1.0),
            _seg(1, "别的办法", "", start=1.0, end=2.0),
        ]
        calls = {"candidates": 0, "repair": 0}

        def translate_fn(*args, **kwargs):
            raise AssertionError("fallback single should not run")

        def translate_candidates_fn(*args, **kwargs):
            calls["candidates"] += 1
            return [
                [{"text": "có hay không...", "style": "natural", "meaning_notes": [], "candidate_source": backend}],
                [{"text": "...cách khác?", "style": "natural", "meaning_notes": [], "candidate_source": backend}],
            ]

        def repair_fn(payloads, *, source, target):
            calls["repair"] += 1
            return {
                "clusters": [
                    {
                        "cluster_id": 0,
                        "segments": [
                            {"segment_id": 0, "translation": "Có cách khác không?"},
                            {"segment_id": 1, "translation": "Đó là vấn đề chính."},
                        ],
                    }
                ]
            }

        translate_segments_with_candidates(
            {"timing_candidate_translation_enabled": True, "translation_backend": backend},
            database=None,
            segments=segments,
            source_lang="zh-CN",
            target_lang="vi",
            translate_fn=translate_fn,
            translate_candidates_fn=translate_candidates_fn,
            repair_fragment_fn=repair_fn,
        )
        assert calls == {"candidates": 1, "repair": 1}
        assert segments[0]["start"] == 0.0 and segments[1]["end"] == 2.0
        assert "fragment_rebalance" in segments[0]
