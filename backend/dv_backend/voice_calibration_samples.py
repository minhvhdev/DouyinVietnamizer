"""Sample eligibility, outlier handling, and profile aggregation for calibration."""

from __future__ import annotations

import hashlib
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .duration_predictor import count_vietnamese_syllables, default_voice_profile, predict_spoken_duration
from .tts_speech_analysis import SpeechEnvelope, measure_speech_envelope
from .voice_calibration_dataset import CalibrationSample

MIN_SPEECH_SEC = 0.25
MIN_SYLLABLES = 2
MIN_SPS = 2.0
MAX_SPS = 8.0
MIN_ANALYSIS_CONFIDENCE = 0.35
MAX_CLIPPING_RATIO = 0.02
ANALYSIS_SCHEMA_VERSION = 1
TRAIN_VALIDATION_SEED = "voice_calibration_v1"


@dataclass
class SampleEvaluation:
    sample_id: str
    accepted: bool
    rejection_reason: str | None
    syllables: int
    observed_sps: float | None
    envelope: SpeechEnvelope | None
    clipping_ratio: float
    analysis: dict[str, Any]


def _clipping_ratio(wav_path: Path) -> float:
    try:
        with wave.open(str(wav_path), "rb") as handle:
            sample_width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
    except (OSError, wave.Error):
        return 1.0
    if not frames:
        return 1.0
    if sample_width == 2:
        threshold = 32700
        clipped = sum(1 for index in range(0, len(frames), 2) if abs(int.from_bytes(frames[index : index + 2], "little", signed=True)) >= threshold)
        total = max(1, len(frames) // 2)
    else:
        clipped = 0
        total = 1
    return clipped / total


def evaluate_calibration_sample(
    sample: CalibrationSample,
    *,
    wav_path: Path | None,
    speed: float = 1.0,
    time_stretched: bool = False,
    from_repaired_audio: bool = False,
    cancelled: bool = False,
    identity_mismatch: bool = False,
    duplicate: bool = False,
    tts_failed: bool = False,
) -> SampleEvaluation:
    if cancelled:
        return SampleEvaluation(sample.id, False, "cancelled", 0, None, None, 0.0, {})
    if identity_mismatch:
        return SampleEvaluation(sample.id, False, "identity_mismatch", 0, None, None, 0.0, {})
    if duplicate:
        return SampleEvaluation(sample.id, False, "duplicate_sample", 0, None, None, 0.0, {})
    if tts_failed:
        return SampleEvaluation(sample.id, False, "tts_failed", 0, None, None, 0.0, {})
    text = (sample.text or "").strip()
    if not text:
        return SampleEvaluation(sample.id, False, "empty_text", 0, None, None, 0.0, {})
    if wav_path is None or not wav_path.is_file():
        return SampleEvaluation(sample.id, False, "missing_audio", 0, None, None, 0.0, {})
    if time_stretched or from_repaired_audio:
        return SampleEvaluation(sample.id, False, "from_repaired_audio" if from_repaired_audio else "time_stretched", 0, None, None, 0.0, {})
    if abs(float(speed) - 1.0) > 0.01:
        return SampleEvaluation(sample.id, False, "user_speed_not_unity", 0, None, None, 0.0, {})

    try:
        envelope = measure_speech_envelope(wav_path)
    except Exception:
        return SampleEvaluation(sample.id, False, "invalid_wav", 0, None, None, 1.0, {})

    clipping = _clipping_ratio(wav_path)
    syllables = count_vietnamese_syllables(text)
    analysis = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "raw_wav_duration": envelope.raw_wav_duration,
        "leading_silence": envelope.leading_silence,
        "trailing_silence": envelope.trailing_silence,
        "speech_envelope_duration": envelope.speech_duration,
        "active_speech_duration": envelope.active_speech_duration,
        "internal_pause_duration": envelope.internal_pause_duration,
        "measurement_confidence": envelope.measurement_confidence,
        "clipping_ratio": round(clipping, 4),
        "syllables": syllables,
    }

    if envelope.speech_duration <= 0.0 or envelope.measurement_confidence < 0.2:
        return SampleEvaluation(sample.id, False, "silent_audio", syllables, None, envelope, clipping, analysis)
    if envelope.speech_duration < MIN_SPEECH_SEC:
        return SampleEvaluation(sample.id, False, "duration_too_short", syllables, None, envelope, clipping, analysis)
    if envelope.measurement_confidence < MIN_ANALYSIS_CONFIDENCE:
        return SampleEvaluation(sample.id, False, "low_analysis_confidence", syllables, None, envelope, clipping, analysis)
    if clipping > MAX_CLIPPING_RATIO:
        return SampleEvaluation(sample.id, False, "audio_clipping", syllables, None, envelope, clipping, analysis)
    if syllables < MIN_SYLLABLES:
        return SampleEvaluation(sample.id, False, "duration_too_short", syllables, None, envelope, clipping, analysis)

    observed_sps = syllables / max(envelope.speech_duration, 0.01)
    analysis["observed_sps"] = round(observed_sps, 4)
    if observed_sps < MIN_SPS or observed_sps > MAX_SPS:
        return SampleEvaluation(sample.id, False, "implausible_speaking_rate", syllables, observed_sps, envelope, clipping, analysis)

    return SampleEvaluation(sample.id, True, None, syllables, observed_sps, envelope, clipping, analysis)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return ordered[index]


def detect_outliers(evaluations: list[SampleEvaluation]) -> tuple[list[SampleEvaluation], list[str]]:
    accepted = [item for item in evaluations if item.accepted and item.observed_sps is not None]
    if len(accepted) < 4:
        return accepted, []
    sps_values = [float(item.observed_sps) for item in accepted]
    q1 = _percentile(sps_values, 0.25)
    q3 = _percentile(sps_values, 0.75)
    iqr = max(0.05, q3 - q1)
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    kept: list[SampleEvaluation] = []
    outlier_ids: list[str] = []
    for item in accepted:
        sps = float(item.observed_sps or 0)
        if sps < lower or sps > upper:
            outlier_ids.append(item.sample_id)
        else:
            kept.append(item)
    if not kept:
        return accepted, []
    return kept, outlier_ids


def aggregate_calibration_profile(
    evaluations: list[SampleEvaluation],
    *,
    language: str = "vi",
) -> dict[str, Any]:
    accepted_before, outlier_ids = detect_outliers(evaluations)
    total_syllables = sum(item.syllables for item in accepted_before)
    total_speech_sec = sum(float(item.envelope.speech_duration) for item in accepted_before if item.envelope)
    aggregate_sps = total_syllables / total_speech_sec if total_speech_sec > 0 else 0.0
    sps_values = [float(item.observed_sps or 0) for item in accepted_before if item.observed_sps]
    return {
        "syllables_per_second": round(aggregate_sps, 3),
        "median_syllables_per_second": round(_percentile(sps_values, 0.5), 3) if sps_values else None,
        "p10_syllables_per_second": round(_percentile(sps_values, 0.10), 3) if sps_values else None,
        "p25_syllables_per_second": round(_percentile(sps_values, 0.25), 3) if sps_values else None,
        "p75_syllables_per_second": round(_percentile(sps_values, 0.75), 3) if sps_values else None,
        "p90_syllables_per_second": round(_percentile(sps_values, 0.90), 3) if sps_values else None,
        "accepted_before_outlier_filter": len([item for item in evaluations if item.accepted]),
        "accepted_after_outlier_filter": len(accepted_before),
        "outlier_sample_ids": outlier_ids,
        "sample_count_outliers": len(outlier_ids),
        "pause_source": "default_vi_v1",
        "comma_pause_ms": default_voice_profile(language).get("comma_pause_ms"),
        "sentence_pause_ms": default_voice_profile(language).get("sentence_pause_ms"),
        "ellipsis_pause_ms": default_voice_profile(language).get("ellipsis_pause_ms"),
    }


def deterministic_train_validation_split(sample_ids: list[str]) -> tuple[list[str], list[str]]:
    scored = sorted(sample_ids, key=lambda sample_id: hashlib.sha256(f"{TRAIN_VALIDATION_SEED}:{sample_id}".encode()).hexdigest())
    split_index = max(1, int(math.floor(len(scored) * 0.8))) if len(scored) >= 5 else len(scored)
    if len(scored) <= 4:
        return scored, []
    return scored[:split_index], scored[split_index:]


def compute_validation_metrics(
    evaluations: list[SampleEvaluation],
    *,
    profile_sps: float,
    language: str = "vi",
) -> dict[str, Any]:
    accepted_map = {item.sample_id: item for item in evaluations if item.accepted}
    train_ids, validation_ids = deterministic_train_validation_split(list(accepted_map.keys()))
    del train_ids
    errors: list[float] = []
    for sample_id in validation_ids:
        item = accepted_map[sample_id]
        if not item.envelope:
            continue
        text = ""
        predicted = predict_spoken_duration(
            text,
            language,
            voice_profile={"syllables_per_second": profile_sps},
        )
        del predicted
        predicted_sec = item.syllables / max(profile_sps, 0.01)
        observed = float(item.envelope.speech_duration)
        errors.append(abs(observed - predicted_sec) * 1000.0)
    if not errors:
        return {
            "prediction_mae_ms": None,
            "prediction_median_error_ms": None,
            "prediction_p90_error_ms": None,
            "validation_sample_count": 0,
        }
    return {
        "prediction_mae_ms": round(sum(errors) / len(errors), 1),
        "prediction_median_error_ms": round(_percentile(errors, 0.5), 1),
        "prediction_p90_error_ms": round(_percentile(errors, 0.9), 1),
        "validation_sample_count": len(errors),
    }


def compute_validation_metrics_for_samples(
    samples: list[CalibrationSample],
    evaluations: list[SampleEvaluation],
    *,
    profile_sps: float,
    language: str = "vi",
) -> dict[str, Any]:
    sample_text = {sample.id: sample.text for sample in samples}
    accepted_map = {item.sample_id: item for item in evaluations if item.accepted}
    _, validation_ids = deterministic_train_validation_split(list(accepted_map.keys()))
    errors: list[float] = []
    for sample_id in validation_ids:
        item = accepted_map[sample_id]
        if not item.envelope:
            continue
        text = sample_text.get(sample_id, "")
        prediction = predict_spoken_duration(text, language, voice_profile={"syllables_per_second": profile_sps})
        observed = float(item.envelope.speech_duration)
        errors.append(abs(observed - float(prediction["predicted_seconds"])) * 1000.0)
    if not errors:
        return {
            "prediction_mae_ms": None,
            "prediction_median_error_ms": None,
            "prediction_p90_error_ms": None,
            "validation_sample_count": 0,
        }
    return {
        "prediction_mae_ms": round(sum(errors) / len(errors), 1),
        "prediction_median_error_ms": round(_percentile(errors, 0.5), 1),
        "prediction_p90_error_ms": round(_percentile(errors, 0.9), 1),
        "validation_sample_count": len(errors),
    }


def cache_key_for_sample(
    *,
    identity_key: str,
    dataset_version: str,
    sample: CalibrationSample,
    generation_config_hash: str,
    cache_schema_version: int = 1,
) -> str:
    payload = {
        "cache_schema_version": cache_schema_version,
        "identity_key": identity_key,
        "dataset_version": dataset_version,
        "sample_id": sample.id,
        "text": sample.text,
        "generation_config_hash": generation_config_hash,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
