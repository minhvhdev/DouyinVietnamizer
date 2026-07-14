"""Voice duration profile convergence and blending policy."""

from __future__ import annotations

from typing import Any

from .duration_predictor import default_voice_profile
from .voice_duration_profile import load_profiles, profile_key, resolve_voice_id

MIN_SAMPLES_DEFAULT = 5
MIN_SAMPLES_LEARNED = 20
HIGH_MAE_MS = 900.0
BOOTSTRAP_MIN_SAMPLES = 10
PARTIAL_MIN_SAMPLES = 10
GOOD_MIN_SAMPLES = 25
POOR_MAE_MS = 900.0
GOOD_MAE_MS = 650.0


def classify_profile_quality(
    *,
    accepted_count: int,
    validation_mae_ms: float | None,
    mode: str = "standard",
    status: str = "ready",
) -> str:
    if status in {"insufficient", "failed", "cancelled"}:
        return status if status in {"failed", "cancelled"} else "insufficient"
    if accepted_count < BOOTSTRAP_MIN_SAMPLES:
        return "insufficient"
    if validation_mae_ms is not None and float(validation_mae_ms) > POOR_MAE_MS:
        return "poor"
    if mode == "quick" and accepted_count >= PARTIAL_MIN_SAMPLES:
        return "partial"
    if accepted_count >= GOOD_MIN_SAMPLES and (
        validation_mae_ms is None or float(validation_mae_ms) <= GOOD_MAE_MS
    ):
        return "good"
    if accepted_count >= PARTIAL_MIN_SAMPLES:
        return "partial"
    return "insufficient"


def blend_profiles(default: dict[str, Any], learned: dict[str, Any], *, weight: float) -> dict[str, Any]:
    weight = max(0.0, min(1.0, weight))
    sps = (1.0 - weight) * float(default.get("syllables_per_second") or 4.0) + weight * float(
        learned.get("syllables_per_second") or 4.0
    )
    return {
        **default,
        **learned,
        "syllables_per_second": round(sps, 3),
        "profile_blend_weight": round(weight, 3),
        "profile_source": "blended" if 0 < weight < 1 else ("learned" if weight >= 1 else "default"),
    }


def effective_voice_profile(
    settings: dict[str, Any],
    *,
    language: str = "vi",
    data_dir=None,
) -> dict[str, Any]:
    default = default_voice_profile(language)
    if not bool(settings.get("voice_duration_profile_enabled", True)):
        return {**default, "profile_source": "default_disabled", "prediction_method": "default_disabled"}

    from .voice_duration_profile import resolve_voice_profile

    learned = resolve_voice_profile(settings, language=language, data_dir=data_dir)
    quality = str(learned.get("quality") or "")
    status = str(learned.get("status") or "")
    samples = int(learned.get("sample_count_accepted") or learned.get("samples") or 0)
    mae = learned.get("prediction_mae_ms") or learned.get("prediction_error_mae_ms")
    source = str(learned.get("source") or "")

    if status == "stale" or quality == "stale":
        return {**default, "profile_source": "default_stale_profile", "samples": samples, "prediction_method": "default_stale_vi_v1"}
    if quality == "insufficient" or samples < MIN_SAMPLES_DEFAULT:
        return {
            **default,
            "profile_source": "default_insufficient_samples",
            "samples": samples,
            "prediction_method": "default_vi_v1",
        }
    if quality == "poor" or (mae is not None and float(mae) > HIGH_MAE_MS):
        blended = blend_profiles(default, learned, weight=0.35)
        blended["prediction_method"] = "voice_calibrated_blend_vi_v1"
        return blended
    if quality == "partial" or samples < MIN_SAMPLES_LEARNED:
        weight = min(0.65, max(0.35, samples / max(MIN_SAMPLES_LEARNED, 1)))
        blended = blend_profiles(default, learned, weight=weight)
        blended["prediction_method"] = "voice_calibrated_partial_vi_v1"
        return blended
    method = "voice_calibrated_vi_v1"
    if source.startswith("bootstrap"):
        method = "voice_calibrated_bootstrap_vi_v1"
    elif source.startswith("bootstrap_plus"):
        method = "voice_calibrated_merged_vi_v1"
    return {**learned, "profile_source": "learned", "prediction_method": method}


def profile_convergence_report(
    data_dir,
    settings: dict[str, Any],
    *,
    language: str = "vi",
) -> dict[str, Any]:
    backend = str(settings.get("tts_backend") or "omnivoice")
    voice_id = resolve_voice_id(settings)
    model = str(settings.get("omnivoice_model") or settings.get("edge_tts_voice") or settings.get("google_tts_voice") or "")
    key = profile_key(tts_backend=backend, voice_id=voice_id, language=language, model=model)
    store = load_profiles(data_dir)
    profile = store.get("profiles", {}).get(key) or {}
    samples = int(profile.get("samples") or 0)
    mae = profile.get("prediction_error_mae_ms")
    issues: list[str] = []
    if samples < MIN_SAMPLES_DEFAULT:
        issues.append("insufficient_samples")
    if mae is not None and float(mae) > HIGH_MAE_MS:
        issues.append("high_mae")
    sps = float(profile.get("syllables_per_second") or 0)
    if sps and (sps < 2.5 or sps > 7.0):
        issues.append("abnormal_speaking_rate")
    return {
        "profile_key": key,
        "voice_id": voice_id,
        "samples": samples,
        "syllables_per_second": profile.get("syllables_per_second"),
        "prediction_error_mae_ms": mae,
        "rejected_sample_count": profile.get("rejected_sample_count", 0),
        "outlier_count": profile.get("outlier_count", 0),
        "updated_at": profile.get("updated_at"),
        "issues": issues,
        "effective_profile": effective_voice_profile(settings, language=language, data_dir=data_dir),
    }
