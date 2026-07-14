"""Tests for semantic safeguards."""

from __future__ import annotations

from dv_backend.semantic_safeguards import candidate_passes_semantic_guards, evaluate_semantic_safeguards


def test_missing_negation_is_critical() -> None:
    result = evaluate_semantic_safeguards(
        "Cho trẻ em sử dụng.",
        source_text="Không được cho trẻ em sử dụng.",
        reference_text="Không được để trẻ em sử dụng.",
    )
    assert result["critical_violation"] is True
    assert "rejected_missing_negation" in result["penalties"]


def test_missing_percentage_is_critical() -> None:
    result = evaluate_semantic_safeguards(
        "Giá còn 700 nghìn đồng.",
        source_text="Giá giảm 30%, còn 700 nghìn đồng.",
        reference_text="Giá giảm 30%, còn 700 nghìn đồng.",
    )
    assert result["critical_violation"] is True
    assert "rejected_missing_percentage" in result["penalties"]


def test_missing_model_identifier_is_critical() -> None:
    result = evaluate_semantic_safeguards(
        "Mẫu RTX nhanh hơn.",
        source_text="Mẫu RTX 5090 nhanh hơn RTX 4090.",
        reference_text="Mẫu RTX 5090 nhanh hơn RTX 4090.",
    )
    assert result["critical_violation"] is True
    assert "rejected_missing_entity" in result["penalties"]


def test_valid_compact_passes() -> None:
    assert candidate_passes_semantic_guards(
        "Không được để trẻ em sử dụng.",
        source_text="Không được cho trẻ em sử dụng.",
    )
