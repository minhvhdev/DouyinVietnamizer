from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .checkpoints import load_checkpoint, save_checkpoint
from .segment_edit_ops import (
    CorruptSegmentEditPlanError,
    InvalidJobStateError,
    SEGMENT_EDIT_CHECKPOINT,
    _job_lock,
)
from .segment_edit_plan import (
    EditableSegment,
    PlanVersionConflictError,
    SegmentEditPlan,
    diff_applied_to_draft,
)


def materialize_target_segments(
    plan: SegmentEditPlan,
    *,
    source_segments: Sequence[Mapping[str, Any]],
    synthesized_by_id: Mapping[str, Mapping[str, Any]],
    repaired_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build a complete downstream segment list using stable segment identity."""

    source_by_index = {
        int(segment.get("index", position)): segment
        for position, segment in enumerate(source_segments)
    }
    target: list[dict[str, Any]] = []
    for index, editable in enumerate(plan.draft_segments):
        source = (
            source_by_index.get(editable.source_segment_index)
            if editable.source_segment_index is not None
            else None
        )
        synthesized = synthesized_by_id.get(editable.segment_id)
        repaired = repaired_by_id.get(editable.segment_id)
        if source is None and synthesized is None:
            raise ValueError(
                f"Segment {editable.segment_id} has no source or synthesized artifact."
            )

        segment = deepcopy(dict(source or synthesized or {}))
        if synthesized is not None:
            segment.update(deepcopy(dict(synthesized)))
        if repaired is not None:
            segment.update(deepcopy(dict(repaired)))
        segment.update(
            {
                "segment_id": editable.segment_id,
                "index": index,
                "start": editable.start_ms / 1000.0,
                "end": editable.end_ms / 1000.0,
                "original_duration": (
                    editable.end_ms - editable.start_ms
                )
                / 1000.0,
                "duration_budget": (
                    editable.end_ms - editable.start_ms
                )
                / 1000.0,
                "translation": editable.spoken_text,
                "tts_spoken_text": editable.spoken_text,
            }
        )
        target.append(segment)
    return target


SEGMENT_EXPORT_STATE_CHECKPOINT = "segment_export_state"
SEGMENT_AUDIO_MANIFEST_CHECKPOINT = "segment_audio_manifest"
ACTIVE_EXPORT_STATUSES = {"queued", "preparing", "prepared", "running"}
DOWNSTREAM_STEPS = ("align_final_dub", "mix", "render", "qc")
BACKED_UP_STEPS = ("tts", "duration_repair", *DOWNSTREAM_STEPS)
RECOVERABLE_JOB_STATUSES = {
    "interrupted",
    "failed",
    "completed",
    "needs_review",
    "queued",
}


class SegmentExportInProgressError(RuntimeError):
    pass


class SegmentArtifactUnavailableError(RuntimeError):
    pass


def _audio_artifact_usable(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _paths_equivalent(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return Path(left) == Path(right)


class SegmentExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_plan_version: int = Field(ge=1)


class SegmentExportService:
    def __init__(self, data_dir, database, jobs, runner, segment_edits) -> None:
        self.data_dir = data_dir
        self.database = database
        self.jobs = jobs
        self.runner = runner
        self.segment_edits = segment_edits

    def request_export(self, job_id: str, *, expected_plan_version: int) -> dict[str, Any]:
        with _job_lock(self.data_dir, job_id):
            job = self.jobs.get(job_id)
            current_state = load_checkpoint(
                self.data_dir, job_id, SEGMENT_EXPORT_STATE_CHECKPOINT
            )
            if str((current_state or {}).get("status")) in ACTIVE_EXPORT_STATUSES:
                raise SegmentExportInProgressError(
                    f"A segment export is already active for job {job_id}."
                )
            if str(job.status) != "completed":
                raise InvalidJobStateError(job_id, str(job.status))
            plan = self.segment_edits._load_canonical(job_id)
            if plan is None:
                plan = self.segment_edits._initialize_from_legacy(job_id)
            if expected_plan_version != plan.plan_version:
                raise PlanVersionConflictError(
                    expected=expected_plan_version,
                    current=plan.plan_version,
                )

            diff = diff_applied_to_draft(
                plan.applied_segments,
                plan.draft_segments,
            )
            if not diff.has_changes:
                return {
                    "status": "unchanged",
                    "plan_version": plan.plan_version,
                    "applied_plan_version": plan.applied_plan_version,
                }

            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            output_path = (
                Path(self.data_dir) / "jobs" / job_id / "output" / "dubbed.mp4"
            )
            previous_output_path = None
            if output_path.is_file():
                backup_output = (
                    Path(self.data_dir)
                    / "jobs"
                    / job_id
                    / "output"
                    / f"dubbed.pre_export_v{plan.plan_version}.mp4"
                )
                shutil.copyfile(output_path, backup_output)
                previous_output_path = str(backup_output)
            backup_checkpoints = {
                step: load_checkpoint(self.data_dir, job_id, step)
                for step in BACKED_UP_STEPS
            }
            state = {
                "schema_version": 1,
                "status": "queued",
                "job_id": job_id,
                "captured_plan_version": plan.plan_version,
                "captured_applied_plan_version": plan.applied_plan_version,
                "captured_applied_segments": [
                    segment.model_dump(mode="json")
                    for segment in plan.applied_segments
                ],
                "captured_segments": [
                    segment.model_dump(mode="json")
                    for segment in plan.draft_segments
                ],
                "requires_tts_segment_ids": sorted(
                    diff.requires_tts_segment_ids
                ),
                "requires_duration_check_segment_ids": sorted(
                    diff.requires_duration_check_segment_ids
                ),
                "backup_checkpoints": backup_checkpoints,
                "previous_output_path": previous_output_path,
                "created_at": now,
                "updated_at": now,
                "error": None,
            }
            save_checkpoint(
                self.data_dir,
                job_id,
                SEGMENT_EXPORT_STATE_CHECKPOINT,
                state,
            )
            with self.database.connection:
                self.database.connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'queued', current_step = 'align_final_dub',
                        last_error_code = NULL, last_error_message = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, job_id),
                )
                self.database.connection.execute(
                    """
                    UPDATE job_steps
                    SET status = 'completed', error_code = NULL, error_message = NULL
                    WHERE job_id = ? AND name IN ('tts', 'duration_repair')
                    """,
                    (job_id,),
                )
                self.database.connection.execute(
                    """
                    UPDATE job_steps
                    SET status = 'pending', started_at = NULL, completed_at = NULL,
                        duration_ms = NULL, error_code = NULL, error_message = NULL
                    WHERE job_id = ? AND name IN ('align_final_dub', 'mix', 'render', 'qc')
                    """,
                    (job_id,),
                )
            self.runner.start_job(job_id)
            return {
                "status": "queued",
                "plan_version": plan.plan_version,
                "requires_tts_segment_ids": state["requires_tts_segment_ids"],
                "requires_duration_check_segment_ids": state[
                    "requires_duration_check_segment_ids"
                ],
            }


    def prepare_pending(self, job_id: str) -> dict[str, Any]:
        with _job_lock(self.data_dir, job_id):
            state = load_checkpoint(
                self.data_dir, job_id, SEGMENT_EXPORT_STATE_CHECKPOINT
            )
            if not state or state.get("status") not in {"queued", "preparing"}:
                return {"status": "skipped"}
            state["status"] = "preparing"
            state["updated_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            save_checkpoint(
                self.data_dir, job_id, SEGMENT_EXPORT_STATE_CHECKPOINT, state
            )

        from . import pipeline

        plan = SegmentEditPlan(
            plan_version=int(state["captured_plan_version"]),
            applied_plan_version=int(state["captured_applied_plan_version"]),
            applied_segments=state["captured_applied_segments"],
            draft_segments=state["captured_segments"],
        )
        backup = state.get("backup_checkpoints") or {}
        source_cp = backup.get("duration_repair") or backup.get("tts") or {}
        source_segments = [
            deepcopy(segment) for segment in source_cp.get("segments") or []
        ]
        source_by_index = {
            int(segment.get("index", position)): segment
            for position, segment in enumerate(source_segments)
        }
        target_positions = {
            segment.segment_id: position
            for position, segment in enumerate(plan.draft_segments)
        }

        def seed_segment(editable) -> dict[str, Any]:
            source = (
                source_by_index.get(editable.source_segment_index)
                if editable.source_segment_index is not None
                else None
            )
            seeded = deepcopy(source or {})
            index = target_positions[editable.segment_id]
            seeded.update(
                {
                    "segment_id": editable.segment_id,
                    "index": index,
                    "start": editable.start_ms / 1000.0,
                    "end": editable.end_ms / 1000.0,
                    "original_duration": (
                        editable.end_ms - editable.start_ms
                    )
                    / 1000.0,
                    "duration_budget": (
                        editable.end_ms - editable.start_ms
                    )
                    / 1000.0,
                    "translation": editable.spoken_text,
                    "tts_spoken_text": editable.spoken_text,
                }
            )
            return seeded

        staging_id = (
            f"{job_id}__segment_export_{state['captured_plan_version']}"
        )
        tts_ids = set(state.get("requires_tts_segment_ids") or [])
        tts_inputs = [
            seed_segment(segment)
            for segment in plan.draft_segments
            if segment.segment_id in tts_ids
        ]
        synthesized_by_id: dict[str, dict[str, Any]] = {}
        if tts_inputs:
            save_checkpoint(
                self.data_dir,
                staging_id,
                "translate",
                {"schema_version": 1, "segments": tts_inputs},
            )
            tts_result = pipeline.tts_step(
                staging_id,
                self.runner.config,
                self.database,
                self.runner,
            )
            save_checkpoint(self.data_dir, staging_id, "tts", tts_result)
            synthesized_by_id = {
                str(segment["segment_id"]): segment
                for segment in tts_result.get("segments") or []
            }

        target_before_repair = materialize_target_segments(
            plan,
            source_segments=source_segments,
            synthesized_by_id=synthesized_by_id,
            repaired_by_id={},
        )
        duration_ids = set(
            state.get("requires_duration_check_segment_ids") or []
        )
        duration_inputs = [
            deepcopy(segment)
            for segment in target_before_repair
            if segment["segment_id"] in duration_ids
        ]
        repaired_by_id: dict[str, dict[str, Any]] = {}
        if duration_inputs:
            save_checkpoint(
                self.data_dir,
                staging_id,
                "tts",
                {"schema_version": 1, "segments": duration_inputs},
            )
            duration_result = pipeline.duration_repair_step(
                staging_id,
                self.runner.config,
                self.database,
                self.runner,
            )
            save_checkpoint(
                self.data_dir, staging_id, "duration_repair", duration_result
            )
            repaired_by_id = {
                str(segment["segment_id"]): segment
                for segment in duration_result.get("segments") or []
            }

        target = materialize_target_segments(
            plan,
            source_segments=source_segments,
            synthesized_by_id=synthesized_by_id,
            repaired_by_id=repaired_by_id,
        )
        for segment in target:
            segment_id = str(segment["segment_id"])
            raw_path = str(segment.get("tts_raw_path") or "")
            repaired_path = str(segment.get("tts_path") or raw_path)
            if not raw_path or not repaired_path:
                raise SegmentArtifactUnavailableError(
                    f"Segment {segment_id} has no reusable audio artifact."
                )
            if not _audio_artifact_usable(Path(raw_path)) or not _audio_artifact_usable(
                Path(repaired_path)
            ):
                raise SegmentArtifactUnavailableError(
                    f"Segment {segment_id} reusable audio is missing or empty."
                )
            if segment_id not in tts_ids:
                self._assert_manifest_reuse_consistent(
                    job_id,
                    segment_id=segment_id,
                    raw_path=raw_path,
                    repaired_path=repaired_path,
                )

        from .timing_placement import (
            compute_placement_starts,
            enforce_zero_overlap_placements,
            schedule_soft_placements,
        )

        compute_placement_starts(target)
        schedule_soft_placements(target)
        enforce_zero_overlap_placements(target)

        audio_manifest = {
            "schema_version": 1,
            "job_id": job_id,
            "plan_version": int(state["captured_plan_version"]),
            "entries": {
                str(segment["segment_id"]): {
                    "segment_id": segment["segment_id"],
                    "index": int(segment["index"]),
                    "raw_wav_path": segment.get("tts_raw_path"),
                    "repaired_wav_path": segment.get("tts_path")
                    or segment.get("tts_raw_path"),
                    "spoken_text": segment.get("tts_spoken_text")
                    or segment.get("translation"),
                    "start_ms": int(round(float(segment.get("start") or 0) * 1000)),
                    "end_ms": int(round(float(segment.get("end") or 0) * 1000)),
                }
                for segment in target
            },
        }
        save_checkpoint(
            self.data_dir,
            job_id,
            SEGMENT_AUDIO_MANIFEST_CHECKPOINT,
            audio_manifest,
        )

        completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_checkpoint(
            self.data_dir,
            job_id,
            "tts",
            {
                "schema_version": 1,
                "job_id": job_id,
                "step_name": "tts",
                "completed_at": completed_at,
                "segments": target_before_repair,
            },
        )
        save_checkpoint(
            self.data_dir,
            job_id,
            "duration_repair",
            {
                "schema_version": 2,
                "job_id": job_id,
                "step_name": "duration_repair",
                "completed_at": completed_at,
                "segments": target,
                "release_eligible": True,
                "timing_review_segments": [],
                "timing_overflow_count": 0,
                "voiced_overlap_count": 0,
            },
        )
        state["status"] = "prepared"
        state["candidate_segments"] = target
        state["staging_job_id"] = staging_id
        state["updated_at"] = completed_at
        save_checkpoint(
            self.data_dir, job_id, SEGMENT_EXPORT_STATE_CHECKPOINT, state
        )
        return {
            "status": "prepared",
            "plan_version": state["captured_plan_version"],
            "requires_tts_segment_ids": sorted(tts_ids),
            "requires_duration_check_segment_ids": sorted(duration_ids),
        }


    def finalize_success(self, job_id: str) -> dict[str, Any]:
        with _job_lock(self.data_dir, job_id):
            state = load_checkpoint(
                self.data_dir, job_id, SEGMENT_EXPORT_STATE_CHECKPOINT
            )
            if not state or state.get("status") not in {
                "prepared",
                "running",
                "queued",
            }:
                return {"status": "skipped"}

            current = self.segment_edits._load_canonical(job_id)
            if current is None:
                raise CorruptSegmentEditPlanError(
                    f"Segment edit plan for job {job_id} is missing during export finalize."
                )

            captured_version = int(state["captured_plan_version"])
            captured_segments = state["captured_segments"]
            if current.plan_version == captured_version:
                # Draft unchanged during export: promote applied to the same draft.
                updated = current.model_copy(
                    update={
                        "applied_plan_version": captured_version,
                        "applied_segments": [
                            segment.model_copy(deep=True)
                            for segment in current.draft_segments
                        ],
                    },
                    deep=True,
                )
            else:
                # Newer draft exists: promote only the captured applied snapshot.
                updated = current.model_copy(
                    update={
                        "applied_plan_version": captured_version,
                        "applied_segments": [
                            EditableSegment.model_validate(segment)
                            for segment in captured_segments
                        ],
                    },
                    deep=True,
                )

            save_checkpoint(
                self.data_dir,
                job_id,
                SEGMENT_EDIT_CHECKPOINT,
                updated.model_dump(mode="json"),
            )
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            state["status"] = "succeeded"
            state["updated_at"] = now
            state["error"] = None
            save_checkpoint(
                self.data_dir, job_id, SEGMENT_EXPORT_STATE_CHECKPOINT, state
            )
            with self.database.connection:
                self.database.connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'completed', current_step = NULL,
                        last_error_code = NULL, last_error_message = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, job_id),
                )
            self._cleanup_staging_artifacts(job_id, state)
            return {
                "status": "succeeded",
                "plan_version": updated.plan_version,
                "applied_plan_version": updated.applied_plan_version,
            }

    def finalize_failure(
        self, job_id: str, *, error: str | None = None
    ) -> dict[str, Any]:
        with _job_lock(self.data_dir, job_id):
            state = load_checkpoint(
                self.data_dir, job_id, SEGMENT_EXPORT_STATE_CHECKPOINT
            )
            if not state or state.get("status") in {
                "succeeded",
                "failed",
                None,
            }:
                return {"status": "skipped"}

            backup = state.get("backup_checkpoints") or {}
            for step, payload in backup.items():
                if payload is None:
                    continue
                save_checkpoint(self.data_dir, job_id, step, payload)

            previous_output = state.get("previous_output_path")
            if previous_output and Path(previous_output).is_file():
                target = (
                    Path(self.data_dir)
                    / "jobs"
                    / job_id
                    / "output"
                    / "dubbed.mp4"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(previous_output, target)

            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            state["status"] = "failed"
            state["error"] = error
            state["updated_at"] = now
            save_checkpoint(
                self.data_dir, job_id, SEGMENT_EXPORT_STATE_CHECKPOINT, state
            )
            with self.database.connection:
                self.database.connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'completed', current_step = NULL, updated_at = ?
                    WHERE id = ?
                    """,
                    (now, job_id),
                )
            return {"status": "failed", "error": error}

    def recover_interrupted(
        self, job_id: str | None = None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Restore backups for orphaned active segment exports after crash/restart."""
        if job_id is not None:
            return self._recover_one(job_id)
        recovered: list[dict[str, Any]] = []
        for candidate_id in self._iter_job_ids():
            result = self._recover_one(candidate_id)
            if result.get("status") == "failed":
                recovered.append({"job_id": candidate_id, **result})
        return recovered

    def _recover_one(self, job_id: str) -> dict[str, Any]:
        state = load_checkpoint(
            self.data_dir, job_id, SEGMENT_EXPORT_STATE_CHECKPOINT
        )
        if not state or state.get("status") not in ACTIVE_EXPORT_STATUSES:
            return {"status": "skipped"}
        try:
            job = self.jobs.get(job_id)
            job_status = str(job.status)
        except Exception:
            job_status = "interrupted"
        # Live runner owns in-process running jobs; everything else with an
        # active export after interrupt/restart must be rolled back.
        if job_status == "running":
            return {"status": "skipped"}
        if job_status not in RECOVERABLE_JOB_STATUSES:
            return {"status": "skipped"}
        return self.finalize_failure(
            job_id, error="SEGMENT_EXPORT_INTERRUPTED"
        )

    def _iter_job_ids(self) -> list[str]:
        rows = self.database.connection.execute(
            "SELECT id FROM jobs ORDER BY updated_at DESC"
        ).fetchall()
        return [str(row["id"]) for row in rows]

    def _assert_manifest_reuse_consistent(
        self,
        job_id: str,
        *,
        segment_id: str,
        raw_path: str,
        repaired_path: str,
    ) -> None:
        manifest = load_checkpoint(
            self.data_dir, job_id, SEGMENT_AUDIO_MANIFEST_CHECKPOINT
        )
        if not manifest:
            return
        entry = (manifest.get("entries") or {}).get(segment_id)
        if not entry:
            return
        manifest_raw = str(entry.get("raw_wav_path") or "")
        manifest_repaired = str(
            entry.get("repaired_wav_path") or entry.get("raw_wav_path") or ""
        )
        if manifest_raw and not _paths_equivalent(manifest_raw, raw_path):
            raise SegmentArtifactUnavailableError(
                f"Segment {segment_id} manifest raw path does not match reusable audio."
            )
        if manifest_repaired and not _paths_equivalent(
            manifest_repaired, repaired_path
        ):
            raise SegmentArtifactUnavailableError(
                f"Segment {segment_id} manifest repaired path does not match reusable audio."
            )
        if not _audio_artifact_usable(Path(manifest_raw)) or not _audio_artifact_usable(
            Path(manifest_repaired)
        ):
            raise SegmentArtifactUnavailableError(
                f"Segment {segment_id} manifest audio is missing or empty."
            )

    def _cleanup_staging_artifacts(
        self, job_id: str, state: Mapping[str, Any]
    ) -> None:
        staging_id = state.get("staging_job_id")
        if not staging_id:
            staging_id = (
                f"{job_id}__segment_export_{state.get('captured_plan_version')}"
            )
        staging_dir = Path(self.data_dir) / "jobs" / str(staging_id)
        job_dir = Path(self.data_dir) / "jobs" / job_id
        try:
            if (
                staging_dir.is_dir()
                and staging_dir.resolve() != job_dir.resolve()
                and staging_dir.name.startswith(f"{job_id}__segment_export_")
            ):
                shutil.rmtree(staging_dir, ignore_errors=True)
        except OSError:
            pass
