"""Unit tests for TTS candidate retry policy."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from dv_backend.duration_fit_policy import acceptable_duration_fit, classify_duration_fit
from dv_backend.tts_candidate_retry import synthesize_with_candidate_retry, timing_attempt_limits


PROFILE = {
    "timeline_window": 4.8,
    "speech_target_duration": 3.9,
    "soft_min_duration": 3.3,
    "hard_max_duration": 4.45,
    "leading_silence_allowance": 0.2,
    "trailing_silence_allowance": 0.45,
}


def _make_wav(path: Path, duration_sec: float = 1.0) -> None:
    import array
    import wave

    rate = 16000
    samples = [0.2] * int(rate * duration_sec)
    pcm = array.array("h", [int(sample * 32767) for sample in samples])
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def test_first_candidate_accepted_no_second_synth(tmp_path: Path) -> None:
    wav = tmp_path / "out.wav"
    calls: list[str] = []

    def synthesize_one(text: str, output_path: Path) -> None:
        calls.append(text)
        _make_wav(output_path, 3.9)

    segment = {
        "translation_candidates": [
            {"text": "Hôm nay ta thử món này nhé.", "style": "compact"},
            {"text": "Giờ thử món.", "style": "very_compact"},
        ],
        "selected_candidate_index": 0,
        "timing_profile": PROFILE,
        "translation": "Hôm nay ta thử món này nhé.",
    }
    settings = {
        "timing_max_tts_attempts": 3,
        "timing_max_candidate_tts_attempts": 2,
        "timing_max_llm_rewrite_attempts": 0,
        "voice_duration_profile_enabled": False,
        "tts_backend": "omnivoice",
    }
    result = synthesize_with_candidate_retry(
        segment,
        settings=settings,
        data_dir=tmp_path,
        language="vi",
        session=Mock(),
        synthesize_one=synthesize_one,
        wav_path=wav,
    )
    assert len(calls) == 1
    assert result["accepted"] is True
    assert len(segment["tts_attempts"]) == 1


def test_second_candidate_used_when_first_too_long(tmp_path: Path) -> None:
    wav = tmp_path / "out.wav"
    durations = iter([6.0, 3.8])

    def synthesize_one(text: str, output_path: Path) -> None:
        _make_wav(output_path, next(durations))

    segment = {
        "translation_candidates": [
            {"text": "Câu dài " + "rất " * 30, "style": "natural"},
            {"text": "Câu ngắn.", "style": "compact"},
        ],
        "selected_candidate_index": 0,
        "timing_profile": PROFILE,
        "translation": "Câu dài",
    }
    settings = {
        "timing_max_tts_attempts": 3,
        "timing_max_candidate_tts_attempts": 2,
        "timing_max_llm_rewrite_attempts": 0,
        "voice_duration_profile_enabled": False,
        "tts_backend": "omnivoice",
    }
    result = synthesize_with_candidate_retry(
        segment,
        settings=settings,
        data_dir=tmp_path,
        language="vi",
        session=Mock(),
        synthesize_one=synthesize_one,
        wav_path=wav,
    )
    assert len(segment["tts_attempts"]) == 2
    assert segment["selected_candidate_index"] == 1
    assert result["accepted"] is True


def test_max_attempts_respected(tmp_path: Path) -> None:
    wav = tmp_path / "out.wav"

    def synthesize_one(text: str, output_path: Path) -> None:
        _make_wav(output_path, 8.0)

    segment = {
        "translation_candidates": [
            {"text": "Một", "style": "natural"},
            {"text": "Hai", "style": "compact"},
            {"text": "Ba", "style": "very_compact"},
        ],
        "selected_candidate_index": 0,
        "timing_profile": PROFILE,
        "translation": "Một",
    }
    settings = {
        "timing_max_tts_attempts": 2,
        "timing_max_candidate_tts_attempts": 2,
        "timing_max_llm_rewrite_attempts": 0,
        "voice_duration_profile_enabled": False,
        "tts_backend": "omnivoice",
    }
    synthesize_with_candidate_retry(
        segment,
        settings=settings,
        data_dir=tmp_path,
        language="vi",
        session=Mock(),
        synthesize_one=synthesize_one,
        wav_path=wav,
    )
    assert len(segment["tts_attempts"]) <= 2


def test_all_semantically_rejected_candidates_do_not_rewrite_or_divide_by_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dv_backend.tts_candidate_retry as retry_module

    monkeypatch.setattr(retry_module, "candidate_passes_semantic_guards", lambda *args, **kwargs: False)
    rewrite = Mock(side_effect=AssertionError("rewrite must not run without synthesized audio"))
    monkeypatch.setattr(retry_module, "lengthen_translation_for_timing", rewrite)

    segment = {
        "translation_candidates": [
            {"text": "Bản tự nhiên", "style": "natural"},
            {"text": "Bản gọn", "style": "compact"},
        ],
        "selected_candidate_index": 0,
        "timing_profile": PROFILE,
        "translation": "Bản gọn",
        "text": "原文",
    }
    settings = {
        "timing_max_tts_attempts": 3,
        "timing_max_candidate_tts_attempts": 2,
        "timing_max_llm_rewrite_attempts": 1,
        "voice_duration_profile_enabled": False,
        "tts_backend": "omnivoice",
    }

    result = synthesize_with_candidate_retry(
        segment,
        settings=settings,
        data_dir=tmp_path,
        language="vi",
        session=Mock(),
        synthesize_one=Mock(side_effect=AssertionError("rejected candidates must not synthesize")),
        wav_path=tmp_path / "out.wav",
        database=Mock(),
        estimate_word_count=lambda text: len(text.split()),
    )

    assert result["accepted"] is False
    assert segment["tts_attempt_count"] == 0
    assert all(attempt["reason"] == "rejected_semantic" for attempt in segment["tts_attempts"])
    rewrite.assert_not_called()


def test_rewrite_uses_last_measured_attempt_when_later_candidate_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dv_backend.tts_candidate_retry as retry_module

    guard_results = iter([True, False])
    monkeypatch.setattr(
        retry_module,
        "candidate_passes_semantic_guards",
        lambda *args, **kwargs: next(guard_results),
    )
    rewrite = Mock(return_value=(None, 2))
    monkeypatch.setattr(retry_module, "lengthen_translation_for_timing", rewrite)

    segment = {
        "translation_candidates": [
            {"text": "Bản ngắn", "style": "natural"},
            {"text": "Bản bị chặn", "style": "compact"},
        ],
        "selected_candidate_index": 0,
        "timing_profile": PROFILE,
        "translation": "Bản ngắn",
        "text": "原文",
    }
    settings = {
        "timing_max_tts_attempts": 3,
        "timing_max_candidate_tts_attempts": 2,
        "timing_max_llm_rewrite_attempts": 1,
        "voice_duration_profile_enabled": False,
        "tts_backend": "omnivoice",
        "short_tts_lengthen_min_gap_sec": 0.2,
    }

    synthesize_with_candidate_retry(
        segment,
        settings=settings,
        data_dir=tmp_path,
        language="vi",
        session=Mock(),
        synthesize_one=lambda text, output_path: _make_wav(output_path, 1.0),
        wav_path=tmp_path / "out.wav",
        database=Mock(),
        estimate_word_count=lambda text: len(text.split()),
    )

    assert segment["tts_attempt_count"] == 1
    assert segment["tts_attempts"][-1]["reason"] == "rejected_semantic"
    assert rewrite.call_args.kwargs["current_duration"] > 0


def test_timing_attempt_limits_defaults() -> None:
    limits = timing_attempt_limits({})
    assert limits["max_total"] == 3
    assert limits["max_candidate"] == 2


def test_fingerprint_cache_invalidation_logic() -> None:
    from dv_backend.pipeline import _tts_text_fingerprint

    assert _tts_text_fingerprint("a") != _tts_text_fingerprint("b")


@pytest.mark.parametrize(
    ("speech", "expected"),
    [
        (3.9, "good"),
        (4.2, "slightly_long"),
        (2.5, "too_short"),
    ],
)
def test_duration_fit_bands(speech: float, expected: str) -> None:
    fit = classify_duration_fit(speech, PROFILE)
    if expected == "good":
        assert acceptable_duration_fit(fit)
