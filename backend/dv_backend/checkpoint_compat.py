"""ASR/diarization checkpoint schema validation and fingerprinting."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from .errors import AppError
from .models import ErrorInfo

logger = logging.getLogger(__name__)

ASR_ALIGNMENT_SCHEMA_VERSION = 2
DIARIZE_CHECKPOINT_SCHEMA_VERSION = 3

DIARIZATION_FINGERPRINT_KEYS = (
    "speaker_diarization",
    "diarization_backend",
    "diarization_fallback_backend",
    "diarization_min_speakers",
    "diarization_max_speakers",
    "diarization_demucs_mode",
    "diarization_ensemble_enabled",
    "diarization_second_pass_on_low_confidence",
    "speaker_assignment_min_coverage",
    "speaker_assignment_min_margin",
    "speaker_overlap_flag_threshold",
    "speaker_review_confidence_threshold",
    "speaker_profile_min_seconds",
    "speaker_merge_gap_sec",
)


def fingerprint_subset(settings: dict[str, Any], keys: tuple[str, ...]) -> str:
    payload = {key: settings.get(key) for key in keys}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return digest[:16]


def asr_checkpoint_fingerprint(asr_cp: dict[str, Any]) -> str:
    payload = {
        "schema_version": asr_cp.get("schema_version"),
        "segment_count": len(asr_cp.get("segments") or []),
        "aligned_unit_count": len(asr_cp.get("aligned_units") or []),
        "segments_digest": hashlib.sha256(
            json.dumps(asr_cp.get("segments") or [], sort_keys=True, default=str).encode()
        ).hexdigest()[:12],
        "alignment_digest": hashlib.sha256(
            json.dumps(asr_cp.get("aligned_units") or [], sort_keys=True, default=str).encode()
        ).hexdigest()[:12],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def diarization_settings_fingerprint(settings: dict[str, Any]) -> str:
    return fingerprint_subset(settings, DIARIZATION_FINGERPRINT_KEYS)


def validate_aligned_units(units: Any) -> list[dict[str, Any]]:
    if not isinstance(units, list) or not units:
        raise AppError(
            422,
            ErrorInfo(
                code="MISSING_ALIGNED_UNITS",
                message="ASR checkpoint does not contain aligned units required for diarization.",
                action="Rerun the ASR step with speaker diarization enabled.",
            ),
        )
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(units):
        if not isinstance(raw, dict):
            raise AppError(
                422,
                ErrorInfo(
                    code="CORRUPT_ASR_ALIGNMENT",
                    message=f"Aligned unit at index {index} is not a valid object.",
                    action="Rerun the ASR step to rebuild alignment output.",
                ),
            )
        text = str(raw.get("text") or "")
        if not text.strip():
            continue
        try:
            start = float(raw.get("start", 0.0))
            end = float(raw.get("end", start))
        except (TypeError, ValueError) as error:
            raise AppError(
                422,
                ErrorInfo(
                    code="CORRUPT_ASR_ALIGNMENT",
                    message=f"Aligned unit at index {index} has invalid timestamps.",
                    action="Rerun the ASR step to rebuild alignment output.",
                    detail=str(error),
                ),
            ) from error
        if end <= start:
            end = start + 0.01
        normalized.append({"text": text, "start": round(start, 3), "end": round(end, 3)})
    if not normalized:
        raise AppError(
            422,
            ErrorInfo(
                code="EMPTY_ASR_ALIGNMENT",
                message="ASR alignment contains no usable aligned units.",
                action="Verify source audio and rerun the ASR step.",
            ),
        )
    return normalized


def validate_asr_for_diarization(
    asr_cp: dict[str, Any],
    *,
    speaker_diarization_enabled: bool,
) -> list[dict[str, Any]]:
    if not speaker_diarization_enabled:
        return []

    schema_version = int(asr_cp.get("schema_version") or 1)
    aligned_units = asr_cp.get("aligned_units")

    if schema_version < ASR_ALIGNMENT_SCHEMA_VERSION or not aligned_units:
        legacy_speaker_segments = any(
            segment.get("speaker_id") is not None for segment in (asr_cp.get("segments") or [])
        )
        if legacy_speaker_segments and not aligned_units:
            logger.info(
                "Using legacy ASR speaker labels without aligned units; diarization will passthrough legacy data."
            )
            return []
        if not aligned_units:
            raise AppError(
                422,
                ErrorInfo(
                    code="INCOMPATIBLE_ASR_ALIGNMENT",
                    message=(
                        "This ASR checkpoint was created without character-level alignment required for diarization."
                    ),
                    action="Rerun the ASR step after enabling speaker diarization.",
                    detail=f"schema_version={schema_version}",
                ),
            )
        raise AppError(
            422,
            ErrorInfo(
                code="INCOMPATIBLE_ASR_ALIGNMENT",
                message="ASR alignment schema is too old for the current diarization attribution layer.",
                action="Rerun the ASR step to rebuild aligned units.",
                detail=f"schema_version={schema_version}, required={ASR_ALIGNMENT_SCHEMA_VERSION}",
            ),
        )

    return validate_aligned_units(aligned_units)


def diarize_checkpoint_is_stale(
    diarize_cp: dict[str, Any] | None,
    asr_cp: dict[str, Any],
    settings: dict[str, Any],
) -> bool:
    if not diarize_cp or diarize_cp.get("skipped"):
        return True
    if diarize_cp.get("asr_fingerprint") != asr_checkpoint_fingerprint(asr_cp):
        return True
    if diarize_cp.get("settings_fingerprint") != diarization_settings_fingerprint(settings):
        return True
    return False


def downstream_invalidation_for_voice_mapping_only() -> dict[str, Any]:
    return {
        "invalidated_steps": ["tts", "duration_repair", "mix", "render", "qc"],
        "resume_from": "tts",
        "reason": "voice_mapping_only",
    }


def downstream_invalidation_for_speaker_merge() -> dict[str, Any]:
    return {
        "invalidated_steps": [
            "normalize_segments",
            "translate",
            "tts",
            "duration_repair",
            "mix",
            "render",
            "qc",
        ],
        "resume_from": "normalize_segments",
        "reason": "speaker_merge",
    }


def downstream_invalidation_for_segment_override() -> dict[str, Any]:
    return {
        "invalidated_steps": ["translate", "tts", "duration_repair", "mix", "render", "qc"],
        "resume_from": "translate",
        "reason": "segment_speaker_override",
    }
