from __future__ import annotations

import json
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dv_backend.adapters.subtitles import format_ass_time
from dv_backend.final_dub_alignment import (
    AlignmentPair,
    AsrToken,
    TargetToken,
    align_job_segments_final_dub,
    align_segment_final_dub,
    align_target_tokens_to_asr_tokens,
    apply_placement_to_dub_words,
    assign_timestamps_to_target_tokens,
    build_alignment_cache_identity,
    classify_qwen_asr_backend,
    compute_audio_content_hash,
    filter_valid_dub_words,
    interpolate_token_timestamps,
    normalize_alignment_token,
    refresh_segment_dub_word_timestamps,
    resolve_final_alignment_method,
    segment_has_usable_dub_words,
    strip_absolute_from_dub_words,
    text_similarity,
    tokenize_asr_units,
    tokenize_target_text,
    validate_dub_words_timeline,
    validate_word_timeline,
    wav_has_detectable_speech,
)
from dv_backend.subtitle_timing import (
    build_cues_from_dub_words,
    build_subtitle_cues,
    resolve_ass_quantized_cues,
)


def _asr_tokens(pairs: list[tuple[str, float, float]]) -> list[AsrToken]:
    return [
        AsrToken(text=text, norm=normalize_alignment_token(text), start=start, end=end)
        for text, start, end in pairs
    ]


def _write_wav(path: Path, duration: float = 0.5, *, amplitude: int = 5000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        frame_count = int(16000 * duration)
        sample = amplitude.to_bytes(2, "little", signed=True)
        handle.writeframes(sample * frame_count)


def test_normalize_alignment_token_case_and_punctuation() -> None:
    assert normalize_alignment_token("Món,") == normalize_alignment_token("món")
    assert normalize_alignment_token("NÀY!") == normalize_alignment_token("này")


def test_tokenize_target_text_keeps_punctuation_model_b() -> None:
    tokens = tokenize_target_text("Hôm nay chúng ta thử món này.")
    assert [token.text for token in tokens] == ["Hôm", "nay", "chúng", "ta", "thử", "món", "này."]


def test_required_asr_mismatch_keeps_target_text() -> None:
    target = tokenize_target_text("Hôm nay chúng ta thử món này.")
    asr = _asr_tokens(
        [
            ("hôm", 0.0, 0.2),
            ("nay", 0.2, 0.4),
            ("chúng", 0.4, 0.7),
            ("ta", 0.7, 0.9),
            ("thử", 0.9, 1.1),
            ("món", 1.1, 1.3),
            ("nầy", 1.3, 1.5),
        ]
    )
    pairs = align_target_tokens_to_asr_tokens(target, asr)
    words = assign_timestamps_to_target_tokens(target, asr, pairs, duration=1.6)
    words = validate_word_timeline(words, max_duration=1.6)
    assert words[-1]["text"] == "này."
    assert words[-1]["alignment"] == "replace"


def test_vietnamese_dash_and_quotes_tokenization() -> None:
    tokens = tokenize_target_text('Đà Nẵng — hôm nay trời đẹp. “Thật sao?”')
    assert any("Đà" in token.text for token in tokens)
    assert any("đẹp." in token.text for token in tokens)


def test_number_mismatch_does_not_crash_alignment() -> None:
    target = tokenize_target_text("Giá là 25.000 đồng.")
    asr = _asr_tokens([("giá", 0.0, 0.2), ("là", 0.2, 0.4), ("đồng", 0.8, 1.0)])
    pairs = align_target_tokens_to_asr_tokens(target, asr)
    words = assign_timestamps_to_target_tokens(target, asr, pairs, duration=1.2)
    words = validate_word_timeline(words, max_duration=1.2)
    assert len(words) == len(target)
    assert words[0]["text"] == "Giá"


def test_cache_stores_relative_timestamps_only(tmp_path: Path) -> None:
    words = apply_placement_to_dub_words(
        [{"text": "A", "start": 0.1, "end": 0.3, "alignment": "exact", "confidence": 0.9}],
        placement_start=10.0,
    )
    relative = strip_absolute_from_dub_words(words)
    assert "absolute_start" not in relative[0]
    assert relative[0]["start"] == 0.1


def test_placement_change_recomputes_absolute_without_model(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    wav = job_dir / "artifacts" / "tts" / "tts_repaired_0.wav"
    _write_wav(wav)
    cache_dir = job_dir / "artifacts" / "subtitle_asr"
    segment = {
        "index": 0,
        "translation": "Hôm nay thử món này.",
        "placement_start": 5.0,
        "repaired_duration": 0.5,
        "tts_path": str(wav),
    }

    def fake_details(*_args, **_kwargs):
        return {
            "aligned_units": [
                {"text": "hôm", "start": 0.0, "end": 0.1},
                {"text": "nay", "start": 0.1, "end": 0.2},
                {"text": "thử", "start": 0.2, "end": 0.3},
                {"text": "món", "start": 0.3, "end": 0.4},
                {"text": "nầy", "start": 0.4, "end": 0.5},
            ],
            "segments": [],
            "from_forced_aligner": True,
        }

    with patch("dv_backend.subtitle_timing.transcribe_tts_clip_details_for_subtitles", side_effect=fake_details):
        first = align_segment_final_dub(
            segment,
            job_dir=job_dir,
            cache_dir=cache_dir,
            transcribe_fn=MagicMock(),
            vendor_dir=tmp_path / "vendor",
            settings={"qwen3_asr_model": "asr", "qwen3_aligner_model": "align"},
            ffmpeg_path=Path("ffmpeg"),
            language="Vietnamese",
        )
    assert first["model_called"] is True
    relative_start = segment["dub_words"][0]["start"]
    absolute_at_five = segment["dub_words"][0]["absolute_start"]

    segment.pop("dub_words", None)
    segment["placement_start"] = 12.0
    second = align_segment_final_dub(
        segment,
        job_dir=job_dir,
        cache_dir=cache_dir,
        transcribe_fn=MagicMock(),
        vendor_dir=tmp_path / "vendor",
        settings={"qwen3_asr_model": "asr", "qwen3_aligner_model": "align"},
        ffmpeg_path=Path("ffmpeg"),
        language="Vietnamese",
    )
    assert second["cache_hit"] is True
    assert second["model_called"] is False
    assert segment["dub_words"][0]["start"] == relative_start
    assert segment["dub_words"][0]["absolute_start"] == pytest.approx(12.0 + relative_start, abs=0.01)
    assert segment["dub_words"][0]["absolute_start"] != absolute_at_five


def test_cache_invalidates_on_audio_content_change(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    _write_wav(wav, amplitude=5000)
    identity_a = build_alignment_cache_identity(
        audio_path=wav,
        target_text="hello",
        target_language="Vietnamese",
        asr_model="asr",
        aligner_model="align",
    )
    _write_wav(wav, amplitude=9000)
    identity_b = build_alignment_cache_identity(
        audio_path=wav,
        target_text="hello",
        target_language="Vietnamese",
        asr_model="asr",
        aligner_model="align",
    )
    assert identity_a != identity_b
    assert compute_audio_content_hash(wav) == compute_audio_content_hash(wav)


def test_cache_invalidates_on_target_text_change(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    _write_wav(wav)
    id1 = build_alignment_cache_identity(
        audio_path=wav,
        target_text="A",
        target_language="Vietnamese",
        asr_model="asr",
        aligner_model="align",
    )
    id2 = build_alignment_cache_identity(
        audio_path=wav,
        target_text="B",
        target_language="Vietnamese",
        asr_model="asr",
        aligner_model="align",
    )
    assert id1 != id2


def test_asr_empty_with_detected_audio_uses_fallback_reason(tmp_path: Path) -> None:
    wav = tmp_path / "speech.wav"
    _write_wav(wav, amplitude=8000)
    assert wav_has_detectable_speech(wav)
    backend = classify_qwen_asr_backend(aligned_units=[], asr_segments=[], language="Vietnamese")
    status, method, _confidence = resolve_final_alignment_method(
        backend,
        interpolated_count=3,
        total_tokens=3,
        fallback_reason="asr_empty_with_detected_audio",
    )
    assert status == "fallback_interpolated"
    assert "asr_empty_with_detected_audio" in method


def test_resolve_ass_quantized_cues_prevents_centisecond_overlap() -> None:
    cues = resolve_ass_quantized_cues(
        [
            {"start": 1.004, "end": 1.506, "text": "A"},
            {"start": 1.505, "end": 2.004, "text": "B"},
        ]
    )
    assert cues[1]["start"] >= cues[0]["end"]
    assert format_ass_time(cues[0]["end"]) <= format_ass_time(cues[1]["start"])


def test_ass_quantization_with_tight_placement_offset() -> None:
    segment = {
        "index": 0,
        "placement_start": 31.479,
        "repaired_duration": 2.0,
        "translation": "Xin chào.",
        "dub_words": [
            {
                "text": "Xin",
                "start": 0.006,
                "end": 0.25,
                "absolute_start": 31.485,
                "absolute_end": 31.729,
            },
            {
                "text": "chào.",
                "start": 0.25,
                "end": 0.55,
                "absolute_start": 31.729,
                "absolute_end": 32.029,
            },
        ],
        "dub_alignment_status": "aligned",
    }
    cues = build_subtitle_cues([segment])
    assert cues
    assert format_ass_time(cues[0]["start"])


def test_validate_dub_words_timeline_catches_absolute_mismatch() -> None:
    words = apply_placement_to_dub_words(
        [{"text": "A", "start": 0.0, "end": 0.2, "alignment": "exact", "confidence": 1.0}],
        placement_start=5.0,
    )
    words[0]["absolute_start"] = 99.0
    result = validate_dub_words_timeline(words, placement_start=5.0, max_duration=1.0)
    assert result["absolute_timeline_valid"] is False


def test_align_job_skips_model_when_all_cache_hits(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    wav0 = job_dir / "artifacts" / "tts" / "tts_repaired_0.wav"
    wav1 = job_dir / "artifacts" / "tts" / "tts_repaired_1.wav"
    _write_wav(wav0)
    _write_wav(wav1)
    cache_dir = job_dir / "artifacts" / "subtitle_asr"
    segments = [
        {"index": 0, "translation": "A", "placement_start": 0.0, "repaired_duration": 0.5, "tts_path": str(wav0)},
        {"index": 1, "translation": "B", "placement_start": 1.0, "repaired_duration": 0.5, "tts_path": str(wav1)},
    ]

    transcribe = MagicMock()
    with patch("dv_backend.subtitle_timing.transcribe_tts_clip_details_for_subtitles") as mock_details:
        mock_details.return_value = {
            "aligned_units": [{"text": "a", "start": 0.0, "end": 0.2}],
            "segments": [],
            "from_forced_aligner": True,
        }
        align_job_segments_final_dub(
            segments,
            job_dir=job_dir,
            cache_dir=cache_dir,
            transcribe_fn=transcribe,
            vendor_dir=tmp_path / "vendor",
            settings={"qwen3_asr_model": "asr", "qwen3_aligner_model": "align"},
            ffmpeg_path=Path("ffmpeg"),
            language="Vietnamese",
        )
        first_calls = mock_details.call_count
        result = align_job_segments_final_dub(
            segments,
            job_dir=job_dir,
            cache_dir=cache_dir,
            transcribe_fn=transcribe,
            vendor_dir=tmp_path / "vendor",
            settings={"qwen3_asr_model": "asr", "qwen3_aligner_model": "align"},
            ffmpeg_path=Path("ffmpeg"),
            language="Vietnamese",
        )
    assert first_calls >= 1
    assert result["cache_hits"] == 2
    assert result["model_calls"] == 0


def test_refresh_segment_dub_word_timestamps() -> None:
    segment = {
        "placement_start": 3.5,
        "dub_words": [
            {"text": "Hi", "start": 0.1, "end": 0.3, "absolute_start": 1.1, "absolute_end": 1.3},
        ],
    }
    refresh_segment_dub_word_timestamps(segment)
    assert segment["dub_words"][0]["absolute_start"] == pytest.approx(3.6, abs=0.001)


def test_punctuation_cues_preserve_display_text() -> None:
    segment = {
        "dub_words": [
            {"text": "Xin", "absolute_start": 0.0, "absolute_end": 0.2},
            {"text": "chào!", "absolute_start": 0.2, "absolute_end": 0.5},
        ]
    }
    cues = build_cues_from_dub_words(segment)
    assert cues[0]["text"] == "Xin chào!"


def test_build_subtitle_cues_fallback_without_dub_words() -> None:
    cues = build_subtitle_cues(
        [{"start": 2.0, "placement_start": 1.5, "repaired_duration": 2.0, "translation": "Xin chào."}]
    )
    assert cues[0]["start"] == 1.5


def test_filter_valid_dub_words_rejects_invalid_and_out_of_bounds() -> None:
    segment = {
        "placement_start": 2.0,
        "repaired_duration": 2.0,
        "dub_words": [
            {"text": "ok", "start": 0.0, "end": 0.5, "absolute_start": 2.0, "absolute_end": 2.5},
            {"text": "", "start": 0.5, "end": 0.8, "absolute_start": 2.5, "absolute_end": 2.8},
            {"text": "bad", "start": float("nan"), "end": 1.0},
            {"text": "far", "start": 5.0, "end": 5.5, "absolute_start": 20.0, "absolute_end": 20.5},
        ],
    }
    valid = filter_valid_dub_words(segment["dub_words"], segment)
    assert len(valid) == 1
    assert valid[0]["text"] == "ok"
    assert segment_has_usable_dub_words(segment) is True


def test_segment_with_only_invalid_dub_words_is_not_usable() -> None:
    segment = {
        "placement_start": 0.0,
        "repaired_duration": 2.0,
        "dub_words": [{"text": "x", "start": float("nan"), "end": 1.0}],
    }
    assert segment_has_usable_dub_words(segment) is False
