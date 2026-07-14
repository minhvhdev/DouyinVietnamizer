"""Tests for OmniVoice semantic chunking."""
from __future__ import annotations

import pytest

from dv_backend.omnivoice_chunking import (
    chunking_required,
    normalize_text_for_compare,
    omnivoice_chunk_settings,
    pause_ms_for_chunk,
    segment_text_diagnostics,
    smaller_retry_max_chars,
    split_omnivoice_text_semantic,
    validate_chunk_reconstruction,
)
from dv_backend.adapters.tts import split_omnivoice_tts_text


def test_short_text_not_chunked() -> None:
    text = "Khách đến sao tôi cản?"
    specs = split_omnivoice_text_semantic(text, max_chars=220)
    assert len(specs) == 1
    # Default external chunking is ON, but short text still stays single-shot.
    assert not chunking_required(text, {})


def test_defaults_enable_external_chunking_and_retry() -> None:
    cfg = omnivoice_chunk_settings({})
    assert cfg["external_chunking_enabled"] is True
    assert cfg["retry_on_fidelity_failure"] is True
    assert cfg["retry_max_chars"][1] < cfg["max_chars"]
    assert cfg["retry_max_chars"][2] < cfg["retry_max_chars"][1]
    assert smaller_retry_max_chars(cfg["max_chars"], cfg) == cfg["retry_max_chars"][1]


def test_long_text_requires_chunking_by_default() -> None:
    text = " ".join(["Đây là một câu tiếng Việt dài."] * 20)
    assert len(text) > 240
    assert chunking_required(text, {})
    assert not chunking_required(text, {"omnivoice_external_chunking_enabled": False})


def test_split_on_period() -> None:
    text = "Câu một. Câu hai. Câu ba."
    specs = split_omnivoice_text_semantic(text, max_chars=12)
    assert len(specs) >= 2
    validate_chunk_reconstruction(text, [item["text"] for item in specs])


def test_split_on_comma() -> None:
    text = "Xin chào, bạn khỏe không, tôi rất vui."
    specs = split_omnivoice_text_semantic(text, max_chars=18)
    assert len(specs) >= 2
    validate_chunk_reconstruction(text, [item["text"] for item in specs])


def test_hard_split_without_punctuation() -> None:
    text = " ".join(["từ"] * 80)
    specs = split_omnivoice_text_semantic(text, max_chars=40)
    assert len(specs) >= 2
    validate_chunk_reconstruction(text, [item["text"] for item in specs])
    assert all(len(item["text"]) <= 40 for item in specs)


def test_vietnamese_boundary_safety_accents_punctuation_numbers() -> None:
    text = (
        "Giá RTX 5090 là 3.14 triệu đồng, còn RX 7900 khoảng 2.71. "
        "Ông ấy nói: \"Đừng cắt giữa chữ!\" Rồi cười..."
    )
    specs = split_omnivoice_text_semantic(text, max_chars=42)
    chunks = [item["text"] for item in specs]
    validate_chunk_reconstruction(text, chunks)
    joined = "".join(chunks)
    assert "3.14" in joined
    assert "2.71" in joined
    assert "RTX 5090" in joined
    assert all(len(chunk) <= 42 for chunk in chunks)
    # No chunk should start mid-token relative to a prior alphanumeric without whitespace gap.
    cleaned = " ".join(text.split())
    cursor = 0
    for chunk in chunks:
        start = cleaned.find(chunk.strip(), cursor)
        assert start >= 0
        if start > 0 and cleaned[start - 1].isalnum() and chunk.strip()[0].isalnum():
            token = cleaned[start:]
            # Only allowed when forced by overlong unbroken token.
            left = start
            while left > 0 and not cleaned[left - 1].isspace():
                left -= 1
            right = start
            while right < len(cleaned) and not cleaned[right].isspace():
                right += 1
            assert right - left > 42
        cursor = start + len(chunk.strip())


def test_no_text_loss_on_long_vietnamese() -> None:
    text = (
        "Vào đi, dọn cho sạch. Tại sao? Sao bỗng ra tay? "
        "Vừa rồi còn hàn huyên, chớp mắt đã như hành quyết. "
        * 5
    )
    specs = split_omnivoice_text_semantic(text, max_chars=120)
    validate_chunk_reconstruction(text, [item["text"] for item in specs])


def test_no_duplicate_or_empty_chunks() -> None:
    text = " ".join([f"doan{index}" for index in range(120)])
    chunks = split_omnivoice_tts_text(text, max_chars=100)
    assert chunks
    assert all(chunk.strip() for chunk in chunks)
    validate_chunk_reconstruction(text, chunks)


def test_decimal_not_split() -> None:
    text = "Giá là 3.14 đơn vị và 2.71 nữa."
    specs = split_omnivoice_text_semantic(text, max_chars=200)
    joined = "".join(item["text"] for item in specs)
    assert "3.14" in joined
    assert "2.71" in joined


def test_rtx_model_token_not_split() -> None:
    text = "Card RTX 5090 chạy rất nhanh trong máy."
    specs = split_omnivoice_text_semantic(text, max_chars=200)
    joined = "".join(item["text"] for item in specs)
    assert "RTX 5090" in joined


def test_normalize_reconstruct_equivalent() -> None:
    original = "Xin chào! Bạn khỏe không?"
    specs = split_omnivoice_text_semantic(original, max_chars=15)
    joined = "".join(item["text"] for item in specs)
    assert normalize_text_for_compare(original) == normalize_text_for_compare(joined)


def test_pause_policy() -> None:
    cfg = omnivoice_chunk_settings({})
    assert pause_ms_for_chunk("Câu.", "sentence", cfg) == cfg["pause_sentence_ms"]
    assert pause_ms_for_chunk("A,", "comma", cfg) == cfg["pause_comma_ms"]
    assert pause_ms_for_chunk("word", "hard", cfg) == cfg["pause_hard_ms"]


def test_segment_diagnostics_flags() -> None:
    diag = segment_text_diagnostics("x" * 600, {})
    assert "very_long_text_segment" in diag["segment_diagnostics"]
    # Default chunking is enabled, so long text flags chunking_required.
    assert "tts_chunking_required" in diag["segment_diagnostics"]

    disabled_diag = segment_text_diagnostics(
        "x" * 600, {"omnivoice_external_chunking_enabled": False}
    )
    assert "tts_chunking_required" not in disabled_diag["segment_diagnostics"]


def test_validate_reconstruction_raises_on_mismatch() -> None:
    with pytest.raises(ValueError):
        validate_chunk_reconstruction("abc def", ["abc", "xyz"])


def _collect_leaf_texts(text: str, *, max_chars: int, ladder: list[int]) -> list[str]:
    """Simulate adaptive fallback splits (nested 140/90 ladder) into leaf texts."""
    specs = split_omnivoice_text_semantic(text, max_chars=max_chars)
    leaves: list[str] = []
    for item in specs:
        piece = item["text"]
        next_values = [value for value in ladder if value < max_chars]
        if next_values and len(piece) > next_values[0]:
            next_max = next_values[0]
            try:
                sub = split_omnivoice_text_semantic(piece, max_chars=next_max)
            except ValueError:
                leaves.append(piece)
                continue
            if len(sub) > 1:
                leaves.extend(
                    _collect_leaf_texts(piece, max_chars=next_max, ladder=next_values[1:])
                )
                continue
        leaves.append(piece)
    return leaves


def test_nested_fallback_leaf_tree_reconstructs_without_gaps() -> None:
    """Full leaf trees including nested 140/90 fallbacks must reconstruct prepared text."""
    text = (
        "Vào đi, dọn cho sạch. Tại sao? Sao bỗng ra tay? "
        "Vừa rồi còn hàn huyên, chớp mắt đã như hành quyết. "
        "Giá RTX 5090 là 3.14 triệu đồng, còn RX 7900 khoảng 2.71. "
        * 4
    )
    leaves = _collect_leaf_texts(text, max_chars=220, ladder=[140, 90])
    assert leaves
    assert all(leaf.strip() for leaf in leaves)
    validate_chunk_reconstruction(text, leaves)
    # Soft whitespace canonicalize OK; Vietnamese tones must remain.
    joined = "".join(leaves)
    assert "hành quyết" in joined
    assert "hàn huyên" in joined
    assert "3.14" in joined
    assert "RTX 5090" in joined
    # Nested rungs themselves also reconstruct.
    for max_chars in (140, 90):
        nested = split_omnivoice_text_semantic(text, max_chars=max_chars)
        validate_chunk_reconstruction(text, [item["text"] for item in nested])
        assert all(len(item["text"]) <= max_chars for item in nested)
