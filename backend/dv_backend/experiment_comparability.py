"""Validate A/B experiment job comparability."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .checkpoints import load_checkpoint

FIXED_SETTING_KEYS = (
    "translation_backend",
    "gemini_translation_model",
    "openai_translation_model",
    "tts_backend",
    "omnivoice_model",
    "edge_tts_voice",
    "google_tts_voice",
    "omnivoice_ref_audio",
    "omnivoice_ref_text",
    "tts_global_speed",
    "qwen3_asr_model",
    "qwen3_aligner_model",
    "mix_mode",
    "subtitles_enabled",
    "subtitle_burn_in",
)

ALLOWED_DIFF_KEYS = (
    "timing_candidate_translation_enabled",
    "timing_translation_candidate_count",
    "timing_max_candidate_tts_attempts",
    "timing_max_tts_attempts",
    "timing_max_llm_rewrite_attempts",
    "voice_duration_profile_enabled",
    "timing_good_ratio",
    "timing_good_abs_sec",
    "timing_preferred_tempo_min",
    "timing_preferred_tempo_max",
    "timing_warning_tempo_min",
    "timing_warning_tempo_max",
    "last_segment_max_extra_seconds",
    "last_segment_max_window_seconds",
)


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def source_artifact_fingerprint(data_dir: Path, job_id: str) -> dict[str, Any]:
    artifacts = data_dir / "jobs" / job_id / "artifacts"
    original = artifacts / "original.mp4"
    norm = load_checkpoint(data_dir, job_id, "normalize_segments") or {}
    segments = norm.get("segments") or []
    segment_sig = hashlib.sha256(
        json.dumps(
            [{"index": s.get("index"), "text": s.get("text"), "start": s.get("start"), "end": s.get("end")} for s in segments],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8"),
    ).hexdigest()[:24]
    return {
        "original_video_sha256": file_sha256(original),
        "normalized_segment_signature": segment_sig,
        "segment_count": len(segments),
    }


def voice_identity(settings: dict[str, Any]) -> str:
    from .voice_duration_profile import resolve_voice_id

    backend = str(settings.get("tts_backend") or "")
    model = str(settings.get("omnivoice_model") or settings.get("edge_tts_voice") or settings.get("google_tts_voice") or "")
    return f"{backend}|{resolve_voice_id(settings)}|{model}"


def validate_experiment_comparability(
    data_dir: Path,
    baseline_job_id: str,
    experiment_job_id: str,
    *,
    baseline_settings: dict[str, Any],
    experiment_settings: dict[str, Any],
) -> dict[str, Any]:
    differences: list[str] = []
    warnings: list[str] = []

    base_fp = source_artifact_fingerprint(data_dir, baseline_job_id)
    exp_fp = source_artifact_fingerprint(data_dir, experiment_job_id)

    if base_fp.get("original_video_sha256") != exp_fp.get("original_video_sha256"):
        differences.append("source_video_hash")
    if base_fp.get("normalized_segment_signature") != exp_fp.get("normalized_segment_signature"):
        differences.append("normalized_segments")
    if base_fp.get("segment_count") != exp_fp.get("segment_count"):
        differences.append("segment_count")

    for key in FIXED_SETTING_KEYS:
        base_val = baseline_settings.get(key)
        exp_val = experiment_settings.get(key)
        if json.dumps(base_val, sort_keys=True) != json.dumps(exp_val, sort_keys=True):
            differences.append(f"fixed_setting:{key}")

    if voice_identity(baseline_settings) != voice_identity(experiment_settings):
        differences.append("voice_identity")

    unexpected = []
    all_keys = set(baseline_settings) | set(experiment_settings)
    for key in sorted(all_keys):
        if key in FIXED_SETTING_KEYS or key in ALLOWED_DIFF_KEYS:
            continue
        if key.endswith("_cursor") or key.startswith("gemini_key"):
            continue
        base_val = baseline_settings.get(key)
        exp_val = experiment_settings.get(key)
        if json.dumps(base_val, sort_keys=True) != json.dumps(exp_val, sort_keys=True):
            unexpected.append(key)

    if unexpected:
        warnings.append(f"unexpected_setting_differences:{','.join(unexpected[:10])}")

    return {
        "comparison_valid": not differences,
        "differences": differences,
        "warnings": warnings,
        "baseline_fingerprint": base_fp,
        "experiment_fingerprint": exp_fp,
    }
