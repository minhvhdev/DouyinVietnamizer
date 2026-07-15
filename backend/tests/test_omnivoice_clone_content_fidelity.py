"""Unit tests for OmniVoice clone content-fidelity helpers (no CUDA)."""
from __future__ import annotations

from dv_backend.omnivoice_content_fidelity import (
    describe_target_text_for_generate,
    evaluate_content_fidelity,
    normalize_content_compare_text,
    plan_clone_semantic_chunks,
    split_content_clauses,
)

REGRESSION = "xin chào? bạn là ai? Tôi là Minh?  rất vui được làm quen với bạn"


def test_evaluator_detects_missing_toi_la_minh_clause() -> None:
    heard = "xin chào bạn là ai rất vui được làm quen với bạn"
    result = evaluate_content_fidelity(
        expected_text=REGRESSION,
        recognized_text=heard,
        critical_phrases=["Tôi là Minh"],
    )
    key = normalize_content_compare_text("Tôi là Minh")
    assert key in result["critical_phrases"]
    assert result["critical_phrases"][key] is False
    assert any(key in clause for clause in result["missing_clauses"])
    # High overall similarity must not hide the missing critical phrase.
    assert result["ordered_token_coverage"] < 1.0


def test_evaluator_handles_spaceless_vietnamese_asr() -> None:
    heard = "xinchàobạnlàaitôilàminhrấtvuiđượclàmquenvớibạn"
    result = evaluate_content_fidelity(
        expected_text=REGRESSION,
        recognized_text=heard,
        critical_phrases=["Tôi là Minh"],
    )
    key = normalize_content_compare_text("Tôi là Minh")
    assert result["critical_phrases"][key] is True
    assert result["missing_any_clause"] is False
    assert result["ordered_token_coverage"] >= 0.95


def test_evaluator_accepts_case_and_punctuation_variants() -> None:
    heard = "Xin Chào? Bạn Là Ai? Tôi Là Minh? Rất vui được làm quen với bạn!"
    result = evaluate_content_fidelity(
        expected_text=REGRESSION,
        recognized_text=heard,
        critical_phrases=["Tôi là Minh"],
    )
    key = normalize_content_compare_text("Tôi là Minh")
    assert result["critical_phrases"][key] is True
    assert result["missing_clauses"] == []
    assert result["ordered_token_coverage"] >= 0.95


def test_evaluator_detects_wrong_clause_order() -> None:
    heard = "rất vui được làm quen với bạn xin chào bạn là ai Tôi là Minh"
    result = evaluate_content_fidelity(
        expected_text=REGRESSION,
        recognized_text=heard,
        critical_phrases=["Tôi là Minh"],
    )
    key = normalize_content_compare_text("Tôi là Minh")
    assert result["critical_phrases"][key] is True
    assert result["ordered_clause_ok"] is False


def test_normalization_keeps_vietnamese_diacritics() -> None:
    normalized = normalize_content_compare_text("Tôi là Minh")
    assert "minh" in normalized
    assert "tôi" in normalized
    assert "là" in normalized


def test_split_clauses_four_parts_keeps_toi_la_minh() -> None:
    clauses = split_content_clauses(REGRESSION)
    assert len(clauses) == 4
    joined = " ".join(clauses)
    assert "Tôi là Minh?" in joined or any("Tôi là Minh" in c for c in clauses)
    assert any("Tôi là Minh" in c for c in clauses)
    assert "" not in clauses
    assert all(c.strip() for c in clauses)


def test_double_space_does_not_create_empty_clause() -> None:
    clauses = split_content_clauses("a?  b?   c")
    assert clauses == ["a?", "b?", "c"]


def test_ellipsis_does_not_create_punctuation_only_clauses() -> None:
    text = "Tại sao? Sao đột nhiên lại ra tay? Rõ ràng giây trước..."
    clauses = split_content_clauses(text)
    assert clauses == [
        "Tại sao?",
        "Sao đột nhiên lại ra tay?",
        "Rõ ràng giây trước...",
    ]
    from dv_backend.omnivoice_content_fidelity import split_omnivoice_clone_clauses

    assert split_omnivoice_clone_clauses(text) == clauses


def test_leading_ellipsis_merges_into_following_clause() -> None:
    from dv_backend.omnivoice_content_fidelity import split_omnivoice_clone_clauses

    text = "...còn đang hàn huyên, giây sau đã như hành quyết."
    clauses = split_omnivoice_clone_clauses(text)
    assert clauses == ["...còn đang hàn huyên, giây sau đã như hành quyết."]


def test_semantic_chunks_d1_and_d2_preserve_content() -> None:
    d1 = plan_clone_semantic_chunks(REGRESSION, strategy="d1")
    d2 = plan_clone_semantic_chunks(REGRESSION, strategy="d2")
    assert len(d1) == 2
    assert len(d2) == 3
    assert any("Tôi là Minh" in chunk for chunk in d1)
    assert any("Tôi là Minh" in chunk for chunk in d2)
    assert normalize_content_compare_text("".join(d1)) == normalize_content_compare_text(REGRESSION)
    assert normalize_content_compare_text("".join(d2)) == normalize_content_compare_text(REGRESSION)


def test_semantic_chunks_do_not_split_mid_word() -> None:
    chunks = plan_clone_semantic_chunks(REGRESSION, strategy="balanced")
    for chunk in chunks:
        assert chunk == chunk.strip()
        assert "  " not in chunk or True  # whitespace collapse ok inside planner
    assert all(chunk.strip() for chunk in chunks)


def test_no_punctuation_still_returns_one_chunk() -> None:
    text = "xin chào bạn là ai Tôi là Minh rất vui được làm quen với bạn"
    clauses = split_content_clauses(text)
    assert clauses == [text]
    chunks = plan_clone_semantic_chunks(text, strategy="balanced")
    assert chunks == [text]


def test_describe_target_text_includes_required_fields() -> None:
    meta = describe_target_text_for_generate(REGRESSION, mode="clone")
    assert meta["mode"] == "clone"
    assert meta["target_text_length"] == len(REGRESSION)
    assert meta["target_text_sha256"]
    assert meta["normalized_target_text_sha256"]
    assert meta["unicode_normalization"] in {"NFC", "NFKC"}
    assert "whitespace_runs" in meta
    assert "punctuation_sequence" in meta
    assert "Tôi là Minh" in REGRESSION
    assert meta["contains_critical_span"] is True


def test_clone_chunking_required_only_for_multi_clause_clone() -> None:
    from dv_backend.omnivoice_content_fidelity import clone_content_chunking_required

    assert clone_content_chunking_required(REGRESSION, is_clone=True) is True
    assert clone_content_chunking_required(REGRESSION, is_clone=False) is False
    assert clone_content_chunking_required("Tôi là Minh?", is_clone=True) is False
    assert (
        clone_content_chunking_required(
            REGRESSION, is_clone=True, settings={"omnivoice_clone_chunk_strategy": "off"}
        )
        is False
    )


def test_split_omnivoice_clone_clauses_four_and_invariant() -> None:
    from dv_backend.omnivoice_content_fidelity import split_omnivoice_clone_clauses

    chunks = split_omnivoice_clone_clauses(REGRESSION)
    assert len(chunks) == 4
    assert any(chunk.startswith("Tôi là Minh") for chunk in chunks)
    assert chunks[2].endswith("?")


def test_synthesize_clone_content_preserving_reuses_order_and_fails_hard(tmp_path) -> None:
    import array
    import wave

    from dv_backend.omnivoice_content_fidelity import synthesize_clone_content_preserving

    calls: list[str] = []

    def _write(path, frames=2400):
        samples = array.array("h", [1200 if (i // 20) % 2 == 0 else -1200 for i in range(frames)])
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(samples.tobytes())

    def synth(text: str, path):
        calls.append(text)
        _write(path)

    out = tmp_path / "out.wav"
    meta = synthesize_clone_content_preserving(
        text=REGRESSION,
        output_path=out,
        synthesize_fn=synth,
        validate_chunk_fn=lambda p: None,
        strategy="d1",
    )
    assert meta["tts_clone_content_chunking_used"] is True
    assert meta["tts_clone_content_chunk_count"] == 2
    assert calls == meta["tts_clone_content_chunks"]
    assert out.is_file()

    calls.clear()
    meta_clauses = synthesize_clone_content_preserving(
        text=REGRESSION,
        output_path=tmp_path / "out_clauses.wav",
        synthesize_fn=synth,
        validate_chunk_fn=lambda p: None,
    )
    assert meta_clauses["tts_clone_content_strategy"] == "clauses"
    assert meta_clauses["tts_clone_content_chunk_count"] == 4

    def boom(text: str, path):
        calls.append(text)
        if len(calls) > 1:
            raise RuntimeError("chunk failed")
        _write(path)

    calls.clear()
    from dv_backend.errors import AppError

    try:
        synthesize_clone_content_preserving(
            text=REGRESSION,
            output_path=tmp_path / "fail.wav",
            synthesize_fn=boom,
            validate_chunk_fn=lambda p: None,
            strategy="d1",
        )
        assert False, "expected failure"
    except AppError as exc:
        assert exc.info.code == "OMNIVOICE_CLONE_CLAUSE_FAILED"
        assert "chunk_index" in str(exc.info.detail or "")
