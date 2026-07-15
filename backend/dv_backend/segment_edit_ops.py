from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from threading import Lock, RLock
from typing import Any
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict

from .checkpoints import checkpoint_path, load_checkpoint, save_checkpoint
from .segment_edit_plan import (
    EditableSegment,
    PlanVersionConflictError,
    SegmentEditPlan,
    diff_applied_to_draft,
    replace_draft,
    validate_segments,
)


SEGMENT_EDIT_CHECKPOINT = "segment_edit_plan"
LEGACY_SOURCE_STEPS = ("align_final_dub", "duration_repair", "tts", "translate")
SEGMENT_NAMESPACE = UUID("248d73d4-c219-45f6-9404-d3263ebc56cf")

_LOCKS_GUARD = Lock()
_JOB_LOCKS: dict[str, RLock] = {}


class InvalidJobStateError(ValueError):
    def __init__(self, job_id: str, status: str) -> None:
        self.job_id = job_id
        self.status = status
        super().__init__(f"Job {job_id} must be completed before editing; status={status}.")


class SegmentSourceUnavailableError(ValueError):
    pass


class CorruptSegmentEditPlanError(ValueError):
    pass


class UnknownSegmentIdError(ValueError):
    def __init__(self, segment_id: str) -> None:
        self.segment_id = segment_id
        super().__init__(f"Unknown segment_id: {segment_id}.")


class EditableSegmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str | None = None
    start_ms: int
    end_ms: int
    spoken_text: str


class SaveSegmentEditPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_plan_version: int
    segments: list[EditableSegmentInput]


def _job_lock(data_dir: Path, job_id: str) -> RLock:
    key = f"{data_dir.resolve()}::{job_id}"
    with _LOCKS_GUARD:
        return _JOB_LOCKS.setdefault(key, RLock())


def _canonical_milliseconds(
    segment: dict[str, Any],
) -> tuple[int, int] | None:
    if segment.get("start_ms") is not None and segment.get("end_ms") is not None:
        try:
            return int(segment["start_ms"]), int(segment["end_ms"])
        except (TypeError, ValueError):
            return None
    if segment.get("start") is None or segment.get("end") is None:
        return None
    try:
        return (
            int(round(float(segment["start"]) * 1000)),
            int(round(float(segment["end"]) * 1000)),
        )
    except (TypeError, ValueError):
        return None


def _spoken_text(segment: dict[str, Any]) -> str:
    for key in ("tts_spoken_text", "spoken_text", "translation"):
        value = str(segment.get(key) or "")
        if value.strip():
            return value
    return ""


def _migrate_legacy_segments(
    job_id: str,
    source_step: str,
    raw_segments: Any,
) -> list[EditableSegment] | None:
    if not isinstance(raw_segments, list) or not raw_segments:
        return None
    if not all(isinstance(item, dict) for item in raw_segments):
        return None

    candidate_ids = [str(item.get("segment_id") or "").strip() for item in raw_segments]
    id_counts = Counter(candidate for candidate in candidate_ids if candidate)
    migrated: list[EditableSegment] = []
    for position, raw in enumerate(raw_segments):
        timing = _canonical_milliseconds(raw)
        spoken = _spoken_text(raw)
        if timing is None or not spoken.strip():
            return None
        candidate = candidate_ids[position]
        segment_id = (
            candidate
            if candidate and id_counts[candidate] == 1
            else str(
                uuid5(
                    SEGMENT_NAMESPACE,
                    f"{job_id}:legacy:{source_step}:{position}",
                )
            )
        )
        migrated.append(
            EditableSegment(
                segment_id=segment_id,
                start_ms=timing[0],
                end_ms=timing[1],
                spoken_text=spoken,
                source_text=(
                    str(raw.get("source_text") or raw.get("text") or "") or None
                ),
                origin="pipeline",
                source_segment_index=position,
            )
        )
    try:
        validate_segments(migrated)
    except ValueError:
        return None
    return migrated


class SegmentEditPlanService:
    def __init__(self, data_dir: Path, jobs: Any) -> None:
        self.data_dir = Path(data_dir)
        self.jobs = jobs

    def _require_completed(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job.status != "completed":
            raise InvalidJobStateError(job_id, str(job.status))

    def _load_canonical(self, job_id: str) -> SegmentEditPlan | None:
        path = checkpoint_path(self.data_dir, job_id, SEGMENT_EDIT_CHECKPOINT)
        raw = load_checkpoint(self.data_dir, job_id, SEGMENT_EDIT_CHECKPOINT)
        if raw is None:
            if path.exists():
                raise CorruptSegmentEditPlanError(
                    f"Segment edit plan for job {job_id} is unreadable."
                )
            return None
        try:
            plan = SegmentEditPlan.model_validate(raw)
            validate_segments(plan.applied_segments)
            validate_segments(plan.draft_segments)
            return plan
        except Exception as exc:
            raise CorruptSegmentEditPlanError(
                f"Segment edit plan for job {job_id} is invalid."
            ) from exc

    def _initialize_from_legacy(self, job_id: str) -> SegmentEditPlan:
        for source_step in LEGACY_SOURCE_STEPS:
            checkpoint = load_checkpoint(self.data_dir, job_id, source_step)
            migrated = _migrate_legacy_segments(
                job_id,
                source_step,
                (checkpoint or {}).get("segments"),
            )
            if migrated is None:
                continue
            plan = SegmentEditPlan(
                plan_version=1,
                applied_plan_version=1,
                applied_segments=[
                    segment.model_copy(deep=True) for segment in migrated
                ],
                draft_segments=[
                    segment.model_copy(deep=True) for segment in migrated
                ],
            )
            save_checkpoint(
                self.data_dir,
                job_id,
                SEGMENT_EDIT_CHECKPOINT,
                plan.model_dump(mode="json"),
            )
            return plan
        raise SegmentSourceUnavailableError(
            f"Completed job {job_id} has no editable segment checkpoint."
        )

    def get_or_create(self, job_id: str) -> SegmentEditPlan:
        with _job_lock(self.data_dir, job_id):
            self._require_completed(job_id)
            return self._load_canonical(job_id) or self._initialize_from_legacy(job_id)

    def _materialize_proposal(
        self,
        current: SegmentEditPlan,
        inputs: Sequence[EditableSegmentInput],
    ) -> list[EditableSegment]:
        current_by_id = {
            segment.segment_id: segment for segment in current.draft_segments
        }
        proposal: list[EditableSegment] = []
        for item in inputs:
            if item.segment_id is None:
                proposal.append(
                    EditableSegment(
                        segment_id=str(uuid4()),
                        start_ms=item.start_ms,
                        end_ms=item.end_ms,
                        spoken_text=item.spoken_text,
                        source_text=None,
                        origin="user",
                        source_segment_index=None,
                    )
                )
                continue
            existing = current_by_id.get(item.segment_id)
            if existing is None:
                raise UnknownSegmentIdError(item.segment_id)
            proposal.append(
                existing.model_copy(
                    update={
                        "start_ms": item.start_ms,
                        "end_ms": item.end_ms,
                        "spoken_text": item.spoken_text,
                    },
                    deep=True,
                )
            )
        validate_segments(proposal)
        return proposal

    def save_draft(
        self,
        job_id: str,
        *,
        expected_plan_version: int,
        segments: Sequence[EditableSegmentInput],
    ) -> SegmentEditPlan:
        with _job_lock(self.data_dir, job_id):
            self._require_completed(job_id)
            current = self._load_canonical(job_id) or self._initialize_from_legacy(
                job_id
            )
            if expected_plan_version != current.plan_version:
                raise PlanVersionConflictError(
                    expected=expected_plan_version,
                    current=current.plan_version,
                )
            proposal = self._materialize_proposal(current, segments)
            updated = replace_draft(
                current,
                proposal,
                expected_plan_version=expected_plan_version,
            )
            if updated.plan_version == current.plan_version:
                return updated
            save_checkpoint(
                self.data_dir,
                job_id,
                SEGMENT_EDIT_CHECKPOINT,
                updated.model_dump(mode="json"),
            )
            return updated

    @staticmethod
    def response_payload(plan: SegmentEditPlan) -> dict[str, Any]:
        diff = diff_applied_to_draft(
            plan.applied_segments,
            plan.draft_segments,
        )
        return {
            "schema_version": plan.schema_version,
            "plan_version": plan.plan_version,
            "applied_plan_version": plan.applied_plan_version,
            "draft_segments": [
                segment.model_dump(mode="json") for segment in plan.draft_segments
            ],
            "diff": {
                "has_changes": diff.has_changes,
                "structural_changed": diff.structural_changed,
                "deltas": [
                    delta.model_dump(mode="json") for delta in diff.deltas
                ],
                "requires_tts_segment_ids": sorted(
                    diff.requires_tts_segment_ids
                ),
                "requires_duration_check_segment_ids": sorted(
                    diff.requires_duration_check_segment_ids
                ),
                "reusable_tts_segment_ids": sorted(
                    diff.reusable_tts_segment_ids
                ),
                "deleted_segment_ids": sorted(diff.deleted_segment_ids),
            },
        }
