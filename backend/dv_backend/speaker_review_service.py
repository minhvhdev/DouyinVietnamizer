"""Speaker review API helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoints import PIPELINE_STEPS, checkpoint_path, load_checkpoint, save_checkpoint
from .checkpoint_compat import (
    downstream_invalidation_for_segment_override,
    downstream_invalidation_for_speaker_merge,
    downstream_invalidation_for_voice_mapping_only,
)
from .errors import AppError
from .models import ErrorInfo


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _speaker_ids(diarize_cp: dict[str, Any]) -> set[str]:
    ids = {str(profile.get("speaker_id")) for profile in diarize_cp.get("speaker_profiles") or []}
    ids.discard("None")
    ids.discard("")
    return ids


def require_diarize_checkpoint(data_dir: Path, job_id: str) -> dict[str, Any]:
    diarize_cp = load_checkpoint(data_dir, job_id, "diarize")
    if not diarize_cp or diarize_cp.get("skipped"):
        raise AppError(
            404,
            ErrorInfo(
                code="DIARIZATION_NOT_AVAILABLE",
                message="Diarization data is not available for this job.",
                action="Enable speaker diarization and rerun from ASR.",
            ),
        )
    return diarize_cp


def validate_speaker_id(diarize_cp: dict[str, Any], speaker_id: str) -> None:
    if speaker_id not in _speaker_ids(diarize_cp):
        raise AppError(
            404,
            ErrorInfo(
                code="SPEAKER_NOT_FOUND",
                message="The requested speaker profile was not found.",
                action="Refresh speaker data and try again.",
            ),
        )


def validate_merge_speakers(diarize_cp: dict[str, Any], source: str, target: str) -> None:
    if source == target:
        raise AppError(
            422,
            ErrorInfo(
                code="INVALID_SPEAKER_MERGE",
                message="Source and target speaker must be different.",
                action="Choose two distinct speaker IDs.",
            ),
        )
    validate_speaker_id(diarize_cp, source)
    validate_speaker_id(diarize_cp, target)


def preserve_manual_overrides(existing: dict[str, Any], incoming: dict[str, str]) -> dict[str, str]:
    merged = dict(existing.get("speaker_manual_overrides") or {})
    merged.update(incoming)
    return merged


def reset_downstream_steps(data_dir: Path, job_id: str, step_names: list[str]) -> None:
    for step_name in step_names:
        if step_name not in PIPELINE_STEPS:
            continue
        cp = checkpoint_path(data_dir, job_id, step_name)
        if cp.is_file():
            cp.unlink(missing_ok=True)


def complete_speaker_review(
    *,
    data_dir: Path,
    job_id: str,
    job_status: str,
    database,
    runner,
) -> dict[str, Any]:
    diarize_cp = require_diarize_checkpoint(data_dir, job_id)
    artifacts_ok = (data_dir / "jobs" / job_id / "artifacts" / "diarization" / "diagnostics.json").is_file()
    if not artifacts_ok:
        raise AppError(
            409,
            ErrorInfo(
                code="DIARIZATION_ARTIFACTS_INCOMPLETE",
                message="Diarization artifacts are not fully persisted yet.",
                action="Wait for the diarize step to finish, then retry review completion.",
            ),
        )

    if diarize_cp.get("manual_review_completed") and job_status in {"queued", "running", "completed"}:
        return {
            "status": "already_completed",
            "job_id": job_id,
            "resume_from": "normalize_segments",
        }

    if job_status != "waiting_for_speaker_review":
        raise AppError(
            409,
            ErrorInfo(
                code="JOB_NOT_AWAITING_REVIEW",
                message="This job is not waiting for speaker review.",
                action="Continue the pipeline or rerun diarization if needed.",
            ),
        )

    diarize_cp["manual_review_completed"] = True
    diarize_cp["review_required"] = False
    if diarize_cp.get("diagnostics"):
        diarize_cp["diagnostics"]["manual_review_completed"] = True
    save_checkpoint(data_dir, job_id, "diarize", diarize_cp)

    now = utc_now()
    with database.connection:
        database.connection.execute(
            "UPDATE jobs SET status = 'queued', updated_at = ? WHERE id = ?",
            (now, job_id),
        )
        for step_name in PIPELINE_STEPS[PIPELINE_STEPS.index("normalize_segments") :]:
            database.connection.execute(
                """
                UPDATE job_steps
                SET status = 'pending', started_at = NULL, completed_at = NULL,
                    error_code = NULL, error_message = NULL
                WHERE job_id = ? AND name = ?
                """,
                (job_id, step_name),
            )
    reset_downstream_steps(
        data_dir,
        job_id,
        list(PIPELINE_STEPS[PIPELINE_STEPS.index("normalize_segments") :]),
    )
    runner.start_job(job_id)
    return {
        "status": "resumed",
        "job_id": job_id,
        "resume_from": "normalize_segments",
        "invalidated_steps": list(PIPELINE_STEPS[PIPELINE_STEPS.index("normalize_segments") :]),
    }


def update_voice_mapping(
    diarize_cp: dict[str, Any],
    speaker_id: str,
    tts_voice: str,
) -> dict[str, Any]:
    validate_speaker_id(diarize_cp, speaker_id)
    for profile in diarize_cp.get("speaker_profiles") or []:
        if str(profile.get("speaker_id")) == speaker_id:
            profile["tts_voice"] = tts_voice
            profile["tts_voice_source"] = "manual"
            profile["manual_override"] = True
    diarize_cp["speaker_manual_overrides"] = preserve_manual_overrides(
        diarize_cp,
        {speaker_id: tts_voice},
    )
    invalidation = downstream_invalidation_for_voice_mapping_only()
    return {"diarize_cp": diarize_cp, **invalidation}


def merge_speakers(diarize_cp: dict[str, Any], source: str, target: str) -> dict[str, Any]:
    validate_merge_speakers(diarize_cp, source, target)
    for segment in diarize_cp.get("segments") or []:
        if str(segment.get("diarization_speaker_id")) == source:
            segment["diarization_speaker_id"] = target
            if segment.get("speaker_id") == source:
                segment["speaker_id"] = target
    diarize_cp["speaker_profiles"] = [
        profile
        for profile in (diarize_cp.get("speaker_profiles") or [])
        if str(profile.get("speaker_id")) != source
    ]
    overrides = dict(diarize_cp.get("speaker_manual_overrides") or {})
    if source in overrides:
        overrides[target] = overrides.pop(source)
    diarize_cp["speaker_manual_overrides"] = overrides
    invalidation = downstream_invalidation_for_speaker_merge()
    return {"diarize_cp": diarize_cp, **invalidation}


def segment_override_invalidation() -> dict[str, Any]:
    return downstream_invalidation_for_segment_override()
