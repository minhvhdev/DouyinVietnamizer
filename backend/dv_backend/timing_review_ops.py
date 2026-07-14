"""Targeted timing-review edit → re-TTS → recheck for needs_review jobs."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from .adapters.tts import TtsSession, create_tts_adapter
from .checkpoints import load_checkpoint, save_checkpoint
from .config import AppConfig
from .database import Database
from .errors import AppError
from .models import ErrorInfo
from .pipeline import (
    _convert_tts_to_final_wav,
    _propose_then_apply_uniform_speed,
    get_wav_duration,
    resolve_tool_path,
)
from .timing_placement import (
    compute_placement_starts,
    enforce_zero_overlap_placements,
    schedule_soft_placements,
    segments_with_voiced_overlap,
)
from .timing_review import flag_infeasible_segments, list_timing_review_segments

logger = logging.getLogger(__name__)


def _load_settings(database: Database) -> dict[str, Any]:
    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    return {r["key"]: json.loads(r["value"]) for r in rows}


def _checkpoint_for_review(config: AppConfig, job_id: str) -> tuple[str, dict[str, Any]]:
    for step in ("duration_repair", "align_final_dub", "tts"):
        cp = load_checkpoint(config.data_dir, job_id, step)
        if cp and cp.get("segments"):
            return step, cp
    raise AppError(
        404,
        ErrorInfo(
            code="TIMING_REVIEW_CHECKPOINT_MISSING",
            message="No segment checkpoint is available for timing review.",
            action="Wait until duration_repair finishes or re-run from TTS.",
        ),
    )


def get_timing_review_payload(
    config: AppConfig,
    job_id: str,
    *,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    step, cp = _checkpoint_for_review(config, job_id)
    segments = list(cp.get("segments") or [])
    cfg = settings if settings is not None else {}
    absolute_max = float(cfg.get("edge_tts_overflow_speed_hard_max", 1.2) or 1.2)
    absolute_max = max(1.0, min(1.2, absolute_max))
    compute_placement_starts(segments)
    schedule_soft_placements(segments)
    flag_infeasible_segments(segments, absolute_max_rate=absolute_max)
    rows = list_timing_review_segments(segments, absolute_max_rate=absolute_max)
    return {
        "job_id": job_id,
        "source_step": step,
        "segments": rows,
        "remaining_count": len(rows),
        "release_eligible": len(rows) == 0 and len(segments_with_voiced_overlap(segments)) == 0,
        "max_speed": absolute_max,
        "pace_policy": cfg.get("pace_policy") or "narration_uniform",
    }


def submit_timing_review_edits(
    *,
    config: AppConfig,
    database: Database,
    runner,
    job_id: str,
    edits: list[dict[str, Any]],
    resume_pipeline: bool = True,
) -> dict[str, Any]:
    if not edits:
        raise AppError(
            422,
            ErrorInfo(
                code="TIMING_REVIEW_EMPTY",
                message="No segment edits were submitted.",
                action="Edit at least one flagged segment before submitting.",
            ),
        )

    settings = _load_settings(database)
    absolute_max = float(settings.get("edge_tts_overflow_speed_hard_max", 1.2) or 1.2)
    absolute_max = max(1.0, min(1.2, absolute_max))
    step, cp = _checkpoint_for_review(config, job_id)
    segments = list(cp.get("segments") or [])
    by_index = {int(s.get("index", -1)): s for s in segments}
    tts_dir = config.data_dir / "jobs" / job_id / "artifacts" / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = resolve_tool_path(config, "ffmpeg")

    # Optimistic lock: reject stale edits before any synthesis.
    for edit in edits:
        idx = int(edit.get("index"))
        if idx not in by_index:
            raise AppError(
                404,
                ErrorInfo(
                    code="SEGMENT_NOT_FOUND",
                    message=f"Segment {idx} was not found.",
                    action="Refresh timing review and try again.",
                ),
            )
        expected = edit.get("expected_plan_version")
        if expected is not None:
            current = int(by_index[idx].get("plan_version") or 1)
            if int(expected) != current:
                raise AppError(
                    409,
                    ErrorInfo(
                        code="PLAN_VERSION_CONFLICT",
                        message=(
                            f"Segment {idx} was updated elsewhere "
                            f"(expected plan_version={expected}, current={current})."
                        ),
                        action="Reload timing review and re-apply your edits.",
                        detail=f"index={idx},expected={expected},current={current}",
                    ),
                )

    edited_indices: list[int] = []
    pending_promotes: list[dict[str, Any]] = []
    temp_paths: list[Path] = []

    def _cleanup_temps() -> None:
        for path in temp_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Temp cleanup failed path=%s", path, exc_info=True)

    try:
        with TtsSession(
            settings,
            data_dir=config.data_dir,
            runner=runner,
            adapter_factory=create_tts_adapter,
        ) as session:
            for edit in edits:
                idx = int(edit.get("index"))
                spoken = str(edit.get("spoken_text") or "").strip()
                if not spoken:
                    raise AppError(
                        422,
                        ErrorInfo(
                            code="EMPTY_SPOKEN_TEXT",
                            message=f"Segment {idx} spoken text cannot be empty.",
                            action="Enter shortened Vietnamese text for that segment.",
                        ),
                    )
                seg = by_index[idx]
                if spoken == str(seg.get("tts_spoken_text") or seg.get("translation") or "").strip():
                    continue
                raw_tmp = tts_dir / f"tts_review_{idx}.tmp.wav"
                final_tmp = tts_dir / f"tts_review_final_{idx}.tmp.wav"
                raw_path = tts_dir / f"tts_review_{idx}.wav"
                repaired = tts_dir / f"tts_repaired_{idx}.wav"
                raw_tmp.unlink(missing_ok=True)
                final_tmp.unlink(missing_ok=True)
                temp_paths.extend([raw_tmp, final_tmp])
                session.synthesize(spoken, raw_tmp, segment=seg)
                from .wav_canonical_validate import validate_canonical_wav_candidate

                raw_check = validate_canonical_wav_candidate(raw_tmp)
                if not raw_check.ok:
                    raise AppError(
                        500,
                        ErrorInfo(
                            code="TTS_REVIEW_INVALID_AUDIO",
                            message=(
                                f"TTS candidate invalid for segment {idx} "
                                f"({raw_check.reason}). Previous WAV preserved."
                            ),
                            action="Retry submit, or shorten the text further.",
                            detail=f"index={idx},reason={raw_check.reason}",
                        ),
                    )
                try:
                    _convert_tts_to_final_wav(ffmpeg_path, raw_tmp, final_tmp, job_id, runner)
                except Exception:
                    logger.exception("Review WAV convert failed index=%s", idx)
                    raise AppError(
                        500,
                        ErrorInfo(
                            code="TTS_REVIEW_FAILED",
                            message=f"Could not convert TTS audio for segment {idx}. Previous WAV preserved.",
                            action="Retry submit.",
                        ),
                    )
                candidate = final_tmp if final_tmp.is_file() else raw_tmp
                candidate_check = validate_canonical_wav_candidate(candidate)
                if not candidate_check.ok and candidate is final_tmp:
                    candidate = raw_tmp
                    candidate_check = raw_check
                if not candidate_check.ok:
                    raise AppError(
                        500,
                        ErrorInfo(
                            code="TTS_REVIEW_INVALID_AUDIO",
                            message=(
                                f"TTS candidate failed canonical validation for segment {idx} "
                                f"({candidate_check.reason}). Previous WAV preserved."
                            ),
                            action="Retry submit, or shorten the text further.",
                            detail=f"index={idx},reason={candidate_check.reason}",
                        ),
                    )
                pending_promotes.append(
                    {
                        "index": idx,
                        "spoken": spoken,
                        "candidate": candidate,
                        "raw_tmp": raw_tmp,
                        "final_tmp": final_tmp,
                        "raw_path": raw_path,
                        "repaired": repaired,
                        "duration": candidate_check.duration,
                    }
                )

        # Promote only after every edited segment validated — never partial canonical writes.
        for item in pending_promotes:
            idx = int(item["index"])
            seg = by_index[idx]
            candidate: Path = item["candidate"]
            repaired: Path = item["repaired"]
            raw_tmp: Path = item["raw_tmp"]
            raw_path: Path = item["raw_path"]
            os.replace(str(candidate), str(repaired))
            if candidate is raw_tmp:
                shutil.copyfile(repaired, raw_path)
            elif raw_tmp.is_file():
                os.replace(str(raw_tmp), str(raw_path))
            seg["translation"] = item["spoken"]
            seg["tts_spoken_text"] = item["spoken"]
            seg["tts_path"] = str(repaired)
            seg["tts_raw_path"] = str(raw_path)
            seg["tts_duration"] = round(float(item["duration"] or get_wav_duration(repaired)), 2)
            seg["repaired_duration"] = seg["tts_duration"]
            seg.pop("tts_speed_base_path", None)
            seg["proposed_speed_factor"] = 1.0
            seg["soft_speed_factor"] = 1.0
            seg["plan_version"] = int(seg.get("plan_version") or 1) + 1
            seg["spoken_text_source"] = "user_timing_review"
            seg["tts_status"] = "ok"
            try:
                from .tts_speech_analysis import attach_speech_metrics, measure_speech_envelope

                attach_speech_metrics(seg, measure_speech_envelope(repaired))
            except Exception:
                logger.debug("Speech metrics attach failed index=%s", idx, exc_info=True)
            method = str(seg.get("repaired_method") or "none")
            seg["repaired_method"] = (
                f"{method}+user_review_tts" if method != "none" else "user_review_tts"
            )
            edited_indices.append(idx)
    except Exception:
        _cleanup_temps()
        raise
    else:
        _cleanup_temps()

    # Narration_uniform: one-shot re-apply across all voiced from fresh bases where needed.
    for seg in segments:
        if int(seg.get("index", -1)) in edited_indices:
            seg.pop("tts_speed_base_path", None)
    _propose_then_apply_uniform_speed(
        segments=segments,
        absolute_max_rate=absolute_max,
        ffmpeg_path=ffmpeg_path,
        tts_dir=tts_dir,
        job_id=job_id,
        runner=runner,
    )
    enforce_zero_overlap_placements(segments)
    # enforce updates starts only — recompute overflow/available before flagging.
    compute_placement_starts(segments)
    schedule_soft_placements(segments)
    flag_infeasible_segments(segments, absolute_max_rate=absolute_max)
    overlaps = segments_with_voiced_overlap(segments)

    cp["segments"] = segments
    remaining = list_timing_review_segments(segments, absolute_max_rate=absolute_max)
    release_eligible = len(remaining) == 0 and len(overlaps) == 0
    cp["timing_review_segments"] = remaining
    cp["release_eligible"] = release_eligible
    cp["timing_review_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_checkpoint(config.data_dir, job_id, "duration_repair", cp)
    # Keep align checkpoint in sync if it was the loaded source or already existed.
    align_cp = load_checkpoint(config.data_dir, job_id, "align_final_dub")
    if align_cp:
        align_cp["segments"] = segments
        align_cp["release_eligible"] = release_eligible
        save_checkpoint(config.data_dir, job_id, "align_final_dub", align_cp)

    now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    if release_eligible and resume_pipeline:
        with database.connection:
            database.connection.execute(
                """
                UPDATE jobs
                SET status = 'queued', current_step = 'align_final_dub',
                    last_error_code = NULL, last_error_message = NULL, updated_at = ?
                WHERE id = ?
                """,
                (now, job_id),
            )
            database.connection.execute(
                """
                UPDATE job_steps
                SET status = 'completed', completed_at = COALESCE(completed_at, ?),
                    error_code = NULL, error_message = NULL
                WHERE job_id = ? AND name = 'duration_repair'
                """,
                (now, job_id),
            )
            database.connection.execute(
                """
                UPDATE job_steps
                SET status = 'pending', started_at = NULL, completed_at = NULL,
                    duration_ms = NULL, error_code = NULL, error_message = NULL
                WHERE job_id = ? AND name IN ('align_final_dub', 'mix', 'render', 'qc')
                """,
                (job_id,),
            )
        runner.start_job(job_id)
        status = "resumed"
    else:
        detail = ",".join(str(r["index"]) for r in remaining[:40])
        with database.connection:
            database.connection.execute(
                """
                UPDATE jobs
                SET status = 'needs_review', current_step = 'duration_repair',
                    last_error_code = 'TIMING_REVIEW_REQUIRED',
                    last_error_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    f"{len(remaining)} segment(s) still need shortening."
                    if remaining
                    else f"{len(overlaps)} voiced overlap(s) remain.",
                    now,
                    job_id,
                ),
            )
        status = "needs_review"
        if not remaining and overlaps:
            detail = f"overlaps={len(overlaps)}"

    return {
        "status": status,
        "edited_indices": edited_indices,
        "remaining_count": len(remaining),
        "overlap_count": len(overlaps),
        "release_eligible": release_eligible,
        "segments": remaining,
        "detail": detail if not release_eligible else None,
        "source_step": step,
    }
