"""Voice duration profile storage and online calibration from TTS samples."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .duration_predictor import count_vietnamese_syllables, default_voice_profile
from .voice_identity import identity_from_settings, identity_matches_stored, identity_profile_key

PROFILE_FILENAME = "voice_duration_profiles.json"
PROFILE_SCHEMA_VERSION = 2
EMA_ALPHA = 0.15
MIN_SPEECH_SEC = 0.25
MIN_SYLLABLES = 2
MIN_SPS = 2.0
MAX_SPS = 8.0
MAX_SINGLE_SAMPLE_DELTA = 0.35
ONLINE_LEARNING_RATE = 0.08
BOOTSTRAP_WEIGHT = 0.85


def _profiles_path(data_dir: Path | None) -> Path | None:
    if data_dir is None:
        return None
    return Path(data_dir) / "artifacts" / PROFILE_FILENAME


def load_profiles(data_dir: Path | None) -> dict[str, Any]:
    path = _profiles_path(data_dir)
    if path is None or not path.is_file():
        return {"schema_version": PROFILE_SCHEMA_VERSION, "profiles": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": PROFILE_SCHEMA_VERSION, "profiles": {}, "recovered_from_corruption": True}
    if not isinstance(payload, dict):
        return {"schema_version": PROFILE_SCHEMA_VERSION, "profiles": {}, "recovered_from_corruption": True}
    payload.setdefault("schema_version", PROFILE_SCHEMA_VERSION)
    payload.setdefault("profiles", {})
    return payload


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def save_profiles(data_dir: Path | None, payload: dict[str, Any]) -> None:
    path = _profiles_path(data_dir)
    if path is None:
        return
    payload["schema_version"] = PROFILE_SCHEMA_VERSION
    _atomic_write(path, payload)


def _file_fingerprint(path: str) -> str | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(65536):
            digest.update(chunk)
    return digest.hexdigest()[:32]


def resolve_voice_id(settings: dict[str, Any]) -> str:
    backend = str(settings.get("tts_backend") or "omnivoice")
    ref = str(settings.get("omnivoice_ref_audio") or "").strip()
    if ref:
        fp = _file_fingerprint(ref)
        return f"ref:{fp or ref}"
    instruct = str(settings.get("omnivoice_instruct") or "").strip()
    if instruct:
        return f"instruct:{hashlib.sha256(instruct.encode()).hexdigest()[:16]}"
    voice = str(settings.get("edge_tts_voice") or settings.get("google_tts_voice") or "default")
    return f"{backend}:{voice}"


def profile_key(*, tts_backend: str, voice_id: str, language: str, model: str) -> str:
    return f"{tts_backend}|{voice_id}|{language}|{model}"


def resolve_profile_key(settings: dict[str, Any], *, language: str = "vi") -> str:
    identity = identity_from_settings(settings, language=language)
    return identity_profile_key(identity)


def resolve_voice_profile(
    settings: dict[str, Any],
    *,
    language: str = "vi",
    data_dir: Path | None = None,
) -> dict[str, Any]:
    if not bool(settings.get("voice_duration_profile_enabled", True)):
        return default_voice_profile(language)

    identity = identity_from_settings(settings, language=language)
    key = identity_profile_key(identity)
    store = load_profiles(data_dir)
    stored = store.get("profiles", {}).get(key)
    default = default_voice_profile(language)
    if isinstance(stored, dict) and stored.get("syllables_per_second"):
        if stored.get("voice_identity") and not identity_matches_stored(stored.get("voice_identity"), identity):
            return {
                **default,
                "voice_id": identity.get("voice_id"),
                "profile_key": key,
                "status": "stale",
                "quality": "stale",
            }
        return {**default, **stored, "voice_id": identity.get("voice_id"), "profile_key": key}
    legacy_backend = str(settings.get("tts_backend") or "omnivoice")
    legacy_voice_id = resolve_voice_id(settings)
    legacy_model = str(settings.get("omnivoice_model") or settings.get("edge_tts_voice") or settings.get("google_tts_voice") or "")
    legacy_key = profile_key(tts_backend=legacy_backend, voice_id=legacy_voice_id, language=language, model=legacy_model)
    if legacy_key != key:
        stored_legacy = store.get("profiles", {}).get(legacy_key)
        if isinstance(stored_legacy, dict) and stored_legacy.get("syllables_per_second"):
            return {**default, **stored_legacy, "voice_id": legacy_voice_id, "profile_key": legacy_key}
    return {
        **default,
        "voice_id": identity.get("voice_id"),
        "tts_backend": legacy_backend,
        "profile_key": key,
        "status": "not_started",
        "quality": "insufficient",
    }


def _robust_ema(current: float, measured: float, *, alpha: float = EMA_ALPHA) -> float:
    delta = abs(measured - current)
    if delta > MAX_SINGLE_SAMPLE_DELTA:
        alpha *= 0.25
    damped_alpha = alpha if delta < 0.5 else alpha * 0.5
    return (1.0 - damped_alpha) * current + damped_alpha * measured


def _record_rejection(profiles: dict[str, Any], key: str, reason: str) -> None:
    entry = profiles.setdefault(key, {})
    counts = entry.setdefault("rejected_sample_reasons", {})
    counts[reason] = int(counts.get(reason) or 0) + 1
    entry["rejected_sample_count"] = int(entry.get("rejected_sample_count") or 0) + 1


def update_voice_profile_from_sample(
    settings: dict[str, Any],
    *,
    text: str,
    speech_duration_sec: float,
    data_dir: Path | None = None,
    language: str = "vi",
    time_stretched: bool = False,
    speech_trimmed: bool = False,
    fallback_error: bool = False,
    from_repaired_audio: bool = False,
    candidate_rejected: bool = False,
    measurement_confidence: float = 1.0,
    automatic_tempo_applied: bool = False,
    user_speed_not_unity: bool = False,
    text_similarity: float | None = None,
    semantic_critical: bool = False,
    clipping_detected: bool = False,
    retry_rejected: bool = False,
) -> dict[str, Any] | None:
    """Update profile only from clean raw TTS samples."""
    backend = str(settings.get("tts_backend") or "omnivoice")
    voice_id = resolve_voice_id(settings)
    model = str(settings.get("omnivoice_model") or settings.get("edge_tts_voice") or settings.get("google_tts_voice") or "")
    key = resolve_profile_key(settings, language=language)
    store = load_profiles(data_dir)
    profiles = store.setdefault("profiles", {})

    def reject(reason: str) -> None:
        _record_rejection(profiles, key, reason)
        save_profiles(data_dir, store)

    if not bool(settings.get("voice_duration_profile_enabled", True)):
        return None

    guards = (
        (time_stretched, "time_stretched"),
        (speech_trimmed, "speech_trimmed"),
        (fallback_error, "fallback_error"),
        (from_repaired_audio, "from_repaired_audio"),
        (candidate_rejected, "candidate_rejected"),
        (automatic_tempo_applied, "automatic_tempo_applied"),
        (user_speed_not_unity, "user_speed_not_unity"),
        (semantic_critical, "semantic_critical"),
        (clipping_detected, "clipping_detected"),
        (retry_rejected, "retry_rejected"),
    )
    for flag, reason in guards:
        if flag:
            reject(reason)
            return None

    if speech_duration_sec < MIN_SPEECH_SEC:
        reject("clip_too_short")
        return None
    if measurement_confidence < 0.35:
        reject("low_measurement_confidence")
        return None
    if text_similarity is not None and text_similarity < 0.55:
        reject("low_text_similarity")
        return None

    syllables = count_vietnamese_syllables(text)
    if syllables < MIN_SYLLABLES:
        return None

    measured_sps = syllables / speech_duration_sec
    if measured_sps < MIN_SPS or measured_sps > MAX_SPS:
        reject("implausible_speaking_rate")
        return None

    current = profiles.get(key) or default_voice_profile(language)
    prior_sps = float(current.get("syllables_per_second") or 4.0)
    bootstrap_count = int(current.get("bootstrap_sample_count") or 0)
    production_count = int(current.get("production_sample_count") or 0)
    learning_rate = ONLINE_LEARNING_RATE
    if bootstrap_count >= 20:
        learning_rate = min(learning_rate, 0.05)
    if abs(measured_sps - prior_sps) > MAX_SINGLE_SAMPLE_DELTA and int(current.get("samples") or 0) >= 3:
        current["outlier_count"] = int(current.get("outlier_count") or 0) + 1
        profiles[key] = current
        save_profiles(data_dir, store)
        reject("outlier_sample")
        return None
    new_sps = max(MIN_SPS, min(MAX_SPS, _robust_ema(prior_sps, measured_sps, alpha=learning_rate)))

    samples = int(current.get("samples") or 0) + 1
    production_count += 1
    predicted = syllables / prior_sps
    error_ms = abs(speech_duration_sec - predicted) * 1000.0
    prior_mae = current.get("prediction_error_mae_ms")
    new_mae = error_ms if prior_mae is None else _robust_ema(float(prior_mae), error_ms)

    updated = {
        **current,
        "voice_id": voice_id,
        "tts_backend": backend,
        "language": language,
        "model": model,
        "samples": samples,
        "production_sample_count": production_count,
        "syllables_per_second": round(new_sps, 3),
        "prediction_error_mae_ms": round(new_mae, 1),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profile_key": key,
        "source": "bootstrap_plus_online" if bootstrap_count else current.get("source", "online"),
    }
    profiles[key] = updated
    save_profiles(data_dir, store)
    return updated


def merge_bootstrap_profile(
    data_dir: Path | None,
    *,
    identity: dict[str, Any],
    aggregate: dict[str, Any],
    validation: dict[str, Any],
    manifest: dict[str, Any],
    quality: str,
) -> dict[str, Any]:
    language = str(identity.get("target_language") or "vi")
    key = identity_profile_key(identity)
    store = load_profiles(data_dir)
    profiles = store.setdefault("profiles", {})
    existing = profiles.get(key) if isinstance(profiles.get(key), dict) else {}
    bootstrap_count = int(manifest.get("sample_accepted") or aggregate.get("accepted_after_outlier_filter") or 0)
    production_count = int(existing.get("production_sample_count") or 0)
    prior_sps = float(existing.get("syllables_per_second") or 0)
    new_sps = float(aggregate.get("syllables_per_second") or prior_sps or default_voice_profile(language)["syllables_per_second"])
    if prior_sps > 0 and production_count > 0:
        new_sps = BOOTSTRAP_WEIGHT * new_sps + (1.0 - BOOTSTRAP_WEIGHT) * prior_sps
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    source = "bootstrap_calibration"
    if production_count > 0:
        source = "bootstrap_plus_online"
    payload = {
        **default_voice_profile(language),
        **existing,
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "profile_key": key,
        "voice_identity": identity,
        "status": "ready" if quality in {"good", "partial", "poor"} else "partial",
        "quality": quality,
        "source": source,
        "dataset_version": manifest.get("dataset_version"),
        "calibration_mode": manifest.get("mode"),
        "sample_count_total": manifest.get("sample_total"),
        "sample_count_synthesized": manifest.get("sample_synthesized"),
        "sample_count_accepted": bootstrap_count,
        "sample_count_rejected": manifest.get("sample_rejected"),
        "sample_count_outliers": aggregate.get("sample_count_outliers"),
        "bootstrap_sample_count": bootstrap_count,
        "production_sample_count": production_count,
        "samples": bootstrap_count + production_count,
        "syllables_per_second": round(new_sps, 3),
        "median_syllables_per_second": aggregate.get("median_syllables_per_second"),
        "p10_syllables_per_second": aggregate.get("p10_syllables_per_second"),
        "p90_syllables_per_second": aggregate.get("p90_syllables_per_second"),
        "prediction_mae_ms": validation.get("prediction_mae_ms"),
        "prediction_median_error_ms": validation.get("prediction_median_error_ms"),
        "prediction_p90_error_ms": validation.get("prediction_p90_error_ms"),
        "pause_source": aggregate.get("pause_source", "default_vi_v1"),
        "comma_pause_ms": aggregate.get("comma_pause_ms"),
        "sentence_pause_ms": aggregate.get("sentence_pause_ms"),
        "ellipsis_pause_ms": aggregate.get("ellipsis_pause_ms"),
        "calibration_job_id": manifest.get("job_id"),
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
    }
    profiles[key] = payload
    try:
        save_profiles(data_dir, store)
    except OSError:
        pass
    return payload


def reset_voice_profile(data_dir: Path | None, profile_key_value: str) -> None:
    store = load_profiles(data_dir)
    profiles = store.setdefault("profiles", {})
    profiles.pop(profile_key_value, None)
    save_profiles(data_dir, store)


def save_manual_wps_profile(
    data_dir: Path | None,
    *,
    identity: dict[str, Any],
    words_per_second: float,
    source: str = "manual",
    measure_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    language = str(identity.get("target_language") or "vi")
    rate = max(2.0, min(5.0, float(words_per_second)))
    syllables_per_second = max(2.5, min(6.0, round(rate * 1.15, 3)))
    key = identity_profile_key(identity)
    store = load_profiles(data_dir)
    profiles = store.setdefault("profiles", {})
    existing = profiles.get(key) if isinstance(profiles.get(key), dict) else {}
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {
        **default_voice_profile(language),
        **existing,
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "profile_key": key,
        "voice_identity": identity,
        "voice_id": identity.get("voice_id"),
        "tts_backend": identity.get("tts_backend"),
        "language": language,
        "model": identity.get("model") or "",
        "words_per_second": round(rate, 2),
        "syllables_per_second": syllables_per_second,
        "status": "ready",
        "quality": "manual" if source == "manual" else "measured",
        "source": source,
        "samples": int(existing.get("samples") or 0),
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
    }
    if measure_meta:
        payload["last_measure"] = measure_meta
    profiles[key] = payload
    save_profiles(data_dir, store)
    return payload
