"""Unit tests for omnivoice steps evaluation helpers."""
from __future__ import annotations

from dv_backend.eval.omnivoice_steps_eval import extract_segment_row


def test_extract_segment_row_skips_missing_number_when_not_checked(tmp_path) -> None:
    wav = tmp_path / "ok.wav"
    wav.write_bytes(b"RIFF")
    row = extract_segment_row(
        num_steps=32,
        run_index=0,
        segment_index=0,
        group="long",
        target_text="Tỷ lệ 95,7% năm 2026.",
        output_path=wav,
        segment={"tts_fidelity_status": "not_checked"},
    )
    assert row["missing_number"] is False
    assert row["truncated_ending"] is False


def test_extract_segment_row_marks_missing_output(tmp_path) -> None:
    row = extract_segment_row(
        num_steps=32,
        run_index=3,
        segment_index=0,
        group="short",
        target_text="Xin chào.",
        output_path=tmp_path / "missing.wav",
        segment={"tts_fidelity_status": "not_checked"},
        audio_duration_sec=None,
    )
    assert row["missing_output"] is True
    assert row["failure"] is False
