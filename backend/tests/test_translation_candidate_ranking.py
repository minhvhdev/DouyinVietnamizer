"""Unit tests for translation candidate ranking."""

from __future__ import annotations

from dv_backend.translation_candidate_ranking import rank_translation_candidates


PROFILE = {
    "speech_target_duration": 3.9,
    "hard_max_duration": 4.45,
    "soft_min_duration": 3.3,
}


def test_closest_duration_candidate_selected() -> None:
    candidates = [
        {"text": "Hôm nay chúng ta sẽ thử món này và xem nó có ngon không nhé.", "style": "natural"},
        {"text": "Hôm nay ta thử món này nhé.", "style": "compact"},
        {"text": "Giờ thử món.", "style": "very_compact"},
    ]
    result = rank_translation_candidates(
        candidates,
        timing_profile=PROFILE,
        source_text="今天我们试试这个。",
        language="vi",
    )
    assert result["selected_candidate_index"] in {0, 1, 2}


def test_missing_number_penalized() -> None:
    candidates = [
        {"text": "Giá tăng mạnh trong ngày.", "style": "compact"},
        {"text": "Giá tăng 25% trong ngày.", "style": "natural"},
    ]
    result = rank_translation_candidates(
        candidates,
        timing_profile=PROFILE,
        source_text="今天价格上涨25%。",
        language="vi",
        reference_text="Giá tăng 25% trong ngày.",
    )
    assert result["selected_candidate_index"] == 1


def test_missing_negation_falls_back_to_natural() -> None:
    candidates = [
        {"text": "Không được để trẻ em sử dụng.", "style": "natural"},
        {"text": "Cho trẻ em sử dụng.", "style": "compact"},
    ]
    result = rank_translation_candidates(
        candidates,
        timing_profile=PROFILE,
        source_text="Không được cho trẻ em sử dụng.",
        language="vi",
        reference_text="Không được để trẻ em sử dụng.",
    )
    assert result["selected_candidate_index"] == 0


def test_entity_preservation_affects_score() -> None:
    candidates = [
        {"text": "Công ty ra mắt sản phẩm mới.", "style": "compact"},
        {"text": "Apple ra mắt sản phẩm mới.", "style": "natural"},
    ]
    result = rank_translation_candidates(
        candidates,
        timing_profile=PROFILE,
        source_text="Apple发布新产品。",
        language="vi",
    )
    semantic_scores = {
        item["index"]: item.get("semantic_score", 0.0)
        for item in result["rankings"]
        if not item.get("skipped")
    }
    assert semantic_scores[1] >= semantic_scores[0]


def test_repetition_lowers_naturalness_score() -> None:
    candidates = [
        {"text": "Rất rất rất rất hay.", "style": "compact"},
        {"text": "Rất hay.", "style": "natural"},
    ]
    result = rank_translation_candidates(
        candidates,
        timing_profile=PROFILE,
        source_text="非常好。",
        language="vi",
    )
    naturalness = {
        item["index"]: item.get("naturalness_score", 0.0)
        for item in result["rankings"]
        if not item.get("skipped")
    }
    assert naturalness[1] >= naturalness[0]


def test_too_long_candidate_penalized() -> None:
    long_text = " ".join(["rất dài"] * 40)
    candidates = [
        {"text": long_text, "style": "expanded"},
        {"text": "Ngắn thôi.", "style": "compact"},
    ]
    result = rank_translation_candidates(
        candidates,
        timing_profile=PROFILE,
        source_text="短句。",
        language="vi",
    )
    assert result["selected_candidate_index"] == 1


def test_single_candidate_fallback() -> None:
    candidates = [{"text": "Một câu duy nhất.", "style": "natural"}]
    result = rank_translation_candidates(
        candidates,
        timing_profile=PROFILE,
        source_text="一句。",
        language="vi",
    )
    assert result["selected_candidate_index"] == 0


def test_empty_candidate_list() -> None:
    result = rank_translation_candidates(
        [],
        timing_profile=PROFILE,
        source_text="x",
        language="vi",
    )
    assert result["selected_candidate_index"] == -1


def test_all_invalid_still_picks_deterministic_fallback() -> None:
    candidates = [{"text": " ", "style": "natural"}, {"text": "", "style": "compact"}]
    result = rank_translation_candidates(
        candidates,
        timing_profile=PROFILE,
        source_text="x",
        language="vi",
    )
    assert result["selected_candidate_index"] == -1
