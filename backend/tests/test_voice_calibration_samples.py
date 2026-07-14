"""Tests for calibration sample eligibility and aggregation."""

from __future__ import annotations

import array
import wave
from pathlib import Path

from dv_backend.voice_calibration_dataset import CalibrationSample
from dv_backend.voice_calibration_samples import (
    aggregate_calibration_profile,
    compute_validation_metrics_for_samples,
    deterministic_train_validation_split,
    evaluate_calibration_sample,
)
from dv_backend.voice_profile_policy import classify_profile_quality


def _write_tone(path: Path, *, duration_sec: float = 1.0, amplitude: float = 0.2) -> None:
    rate = 16000
    frames = int(rate * duration_sec)
    samples = array.array("h", [int(amplitude * 32767 * (1 if (index // 80) % 2 == 0 else 0.6)) for index in range(frames)])
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


def test_clean_raw_accepted(tmp_path: Path) -> None:
    wav = tmp_path / "ok.wav"
    _write_tone(wav, duration_sec=1.2)
    sample = CalibrationSample(id="s1", text="Xin chào các bạn nhé", category="normal_sentence")
    result = evaluate_calibration_sample(sample, wav_path=wav, speed=1.0)
    assert result.accepted is True


def test_silent_audio_rejected(tmp_path: Path) -> None:
    wav = tmp_path / "silent.wav"
    with wave.open(str(wav), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    sample = CalibrationSample(id="s2", text="Xin chào các bạn", category="normal_sentence")
    result = evaluate_calibration_sample(sample, wav_path=wav)
    assert result.accepted is False
    assert result.rejection_reason == "silent_audio"


def test_speed_not_unity_rejected(tmp_path: Path) -> None:
    wav = tmp_path / "ok.wav"
    _write_tone(wav)
    sample = CalibrationSample(id="s3", text="Xin chào các bạn", category="normal_sentence")
    result = evaluate_calibration_sample(sample, wav_path=wav, speed=1.2)
    assert result.rejection_reason == "user_speed_not_unity"


def test_aggregate_uses_total_syllables_over_total_duration() -> None:
    from dv_backend.voice_calibration_samples import SampleEvaluation
    from dv_backend.tts_speech_analysis import SpeechEnvelope

    evaluations = [
        SampleEvaluation("a", True, None, 10, 5.0, SpeechEnvelope(1.2, 0.1, 0.1, 2.0, 0.1, 2.1, 0.0, 2.0, 0.9), 0.0, {}),
        SampleEvaluation("b", True, None, 20, 4.0, SpeechEnvelope(2.4, 0.1, 0.1, 5.0, 0.1, 5.1, 0.0, 5.0, 0.9), 0.0, {}),
    ]
    aggregate = aggregate_calibration_profile(evaluations)
    assert aggregate["syllables_per_second"] == round(30 / 7.0, 3)


def test_outlier_detection() -> None:
    from dv_backend.voice_calibration_samples import SampleEvaluation
    from dv_backend.tts_speech_analysis import SpeechEnvelope

    evaluations = []
    for index, sps in enumerate([4.0, 4.1, 4.0, 4.2, 9.5, 4.1]):
        syllables = 8
        duration = syllables / sps
        evaluations.append(
            SampleEvaluation(
                f"s{index}",
                True,
                None,
                syllables,
                sps,
                SpeechEnvelope(duration, 0.0, 0.0, duration, 0.0, duration, 0.0, duration, 0.9),
                0.0,
                {},
            )
        )
    aggregate = aggregate_calibration_profile(evaluations)
    assert aggregate["sample_count_outliers"] >= 1


def test_quality_policy() -> None:
    assert classify_profile_quality(accepted_count=5, validation_mae_ms=500, mode="quick") == "insufficient"
    assert classify_profile_quality(accepted_count=12, validation_mae_ms=500, mode="quick") == "partial"
    assert classify_profile_quality(accepted_count=30, validation_mae_ms=500, mode="standard") == "good"
    assert classify_profile_quality(accepted_count=30, validation_mae_ms=1200, mode="standard") == "poor"


def test_train_validation_split_deterministic() -> None:
    ids = [f"sample_{index:03d}" for index in range(20)]
    train_a, val_a = deterministic_train_validation_split(ids)
    train_b, val_b = deterministic_train_validation_split(ids)
    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) == 16
    assert len(val_a) == 4


def test_validation_metrics_use_holdout() -> None:
    from dv_backend.voice_calibration_samples import SampleEvaluation
    from dv_backend.tts_speech_analysis import SpeechEnvelope

    samples = [
        CalibrationSample(id=f"s{i}", text=f"Câu mẫu số {i} cho kiểm tra.", category="normal_sentence")
        for i in range(10)
    ]
    evaluations = []
    for sample in samples:
        duration = 1.0
        evaluations.append(
            SampleEvaluation(
                sample.id,
                True,
                None,
                6,
                6.0,
                SpeechEnvelope(duration, 0.0, 0.0, duration, 0.0, duration, 0.0, duration, 0.9),
                0.0,
                {},
            )
        )
    metrics = compute_validation_metrics_for_samples(samples, evaluations, profile_sps=6.0)
    assert metrics["validation_sample_count"] >= 1
