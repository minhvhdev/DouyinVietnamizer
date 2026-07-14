"""Tests for OmniVoice semantic chunking."""
from __future__ import annotations

import pytest

from dv_backend.omnivoice_chunking import (
    chunking_required,
    normalize_text_for_compare,
    omnivoice_chunk_settings,
    pause_ms_for_chunk,
    segment_text_diagnostics,
    split_omnivoice_text_semantic,
    validate_chunk_reconstruction,
)
from dv_backend.adapters.tts import split_omnivoice_tts_text


def test_short_text_not_chunked() -> None:
    text = "Khách đến sao tôi cản?"
    specs = split_omnivoice_text_semantic(text, max_chars=220)
    assert len(specs) == 1
    assert not chunking_required(text, {})


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
    assert "tts_chunking_required" not in diag["segment_diagnostics"]

    enabled_diag = segment_text_diagnostics("x" * 600, {"omnivoice_external_chunking_enabled": True})
    assert "tts_chunking_required" in enabled_diag["segment_diagnostics"]


def test_validate_reconstruction_raises_on_mismatch() -> None:
    with pytest.raises(ValueError):
        validate_chunk_reconstruction("abc def", ["abc", "xyz"])
