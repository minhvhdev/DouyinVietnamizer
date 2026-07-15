from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PlanVersionConflictError(ValueError):
    """Raised when a draft replacement is based on a stale plan version."""

    def __init__(self, *, expected: int, current: int) -> None:
        self.expected = expected
        self.current = current
        super().__init__(
            f"Segment edit plan version conflict: expected {expected}, current {current}."
        )


class EditableSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str
    start_ms: int
    end_ms: int
    spoken_text: str
    source_text: str | None = None
    origin: Literal["pipeline", "user"]
    source_segment_index: int | None = None


class SegmentEditPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    plan_version: int
    applied_plan_version: int
    applied_segments: list[EditableSegment]
    draft_segments: list[EditableSegment]


class SegmentDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str
    added: bool = False
    deleted: bool = False
    text_changed: bool = False
    timing_changed: bool = False
    order_changed: bool = False


class PlanDiff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deltas: list[SegmentDelta] = Field(default_factory=list)
    requires_tts_segment_ids: set[str] = Field(default_factory=set)
    requires_duration_check_segment_ids: set[str] = Field(default_factory=set)
    reusable_tts_segment_ids: set[str] = Field(default_factory=set)
    deleted_segment_ids: set[str] = Field(default_factory=set)
    structural_changed: bool = False
    has_changes: bool = False


def validate_segments(segments: Sequence[EditableSegment]) -> None:
    """Validate an editable snapshot without normalizing or reordering it."""

    seen_ids: set[str] = set()
    for segment in segments:
        if not segment.segment_id.strip():
            raise ValueError("segment_id cannot be blank.")
        if segment.segment_id in seen_ids:
            raise ValueError(f"Duplicate segment_id: {segment.segment_id}.")
        seen_ids.add(segment.segment_id)

        if segment.start_ms < 0:
            raise ValueError(
                f"Segment {segment.segment_id} start_ms cannot be negative."
            )
        if segment.end_ms <= segment.start_ms:
            raise ValueError(
                f"Segment {segment.segment_id} end_ms must be greater than start_ms."
            )
        if not segment.spoken_text.strip():
            raise ValueError(f"Segment {segment.segment_id} spoken_text cannot be blank.")
        if segment.origin == "pipeline" and segment.source_segment_index is None:
            raise ValueError(
                f"Pipeline segment {segment.segment_id} requires source_segment_index."
            )
        if segment.origin == "user" and segment.source_segment_index is not None:
            raise ValueError(
                f"User segment {segment.segment_id} cannot set source_segment_index."
            )


def diff_applied_to_draft(
    applied: Sequence[EditableSegment],
    draft: Sequence[EditableSegment],
) -> PlanDiff:
    """Return a deterministic, identity-based selective-processing diff."""

    validate_segments(applied)
    validate_segments(draft)

    applied_by_id = {segment.segment_id: segment for segment in applied}
    draft_by_id = {segment.segment_id: segment for segment in draft}
    applied_positions = {
        segment.segment_id: position for position, segment in enumerate(applied)
    }
    draft_positions = {
        segment.segment_id: position for position, segment in enumerate(draft)
    }

    deltas: list[SegmentDelta] = []
    requires_tts: set[str] = set()
    requires_duration_check: set[str] = set()
    reusable_tts: set[str] = set()
    deleted_ids: set[str] = set()
    structural_changed = False

    for old in applied:
        segment_id = old.segment_id
        new = draft_by_id.get(segment_id)
        if new is None:
            deltas.append(SegmentDelta(segment_id=segment_id, deleted=True))
            deleted_ids.add(segment_id)
            structural_changed = True
            continue

        text_changed = old.spoken_text != new.spoken_text
        timing_changed = (
            old.start_ms != new.start_ms or old.end_ms != new.end_ms
        )
        order_changed = applied_positions[segment_id] != draft_positions[segment_id]

        if text_changed:
            requires_tts.add(segment_id)
            requires_duration_check.add(segment_id)
        else:
            reusable_tts.add(segment_id)
        if timing_changed:
            requires_duration_check.add(segment_id)
        if order_changed:
            structural_changed = True

        if text_changed or timing_changed or order_changed:
            deltas.append(
                SegmentDelta(
                    segment_id=segment_id,
                    text_changed=text_changed,
                    timing_changed=timing_changed,
                    order_changed=order_changed,
                )
            )

    for new in draft:
        segment_id = new.segment_id
        if segment_id in applied_by_id:
            continue
        deltas.append(SegmentDelta(segment_id=segment_id, added=True))
        requires_tts.add(segment_id)
        requires_duration_check.add(segment_id)
        structural_changed = True

    return PlanDiff(
        deltas=deltas,
        requires_tts_segment_ids=requires_tts,
        requires_duration_check_segment_ids=requires_duration_check,
        reusable_tts_segment_ids=reusable_tts,
        deleted_segment_ids=deleted_ids,
        structural_changed=structural_changed,
        has_changes=bool(deltas),
    )


def replace_draft(
    plan: SegmentEditPlan,
    proposed_segments: Sequence[EditableSegment],
    expected_plan_version: int,
) -> SegmentEditPlan:
    """Replace a draft with optimistic concurrency and no input mutation."""

    if expected_plan_version != plan.plan_version:
        raise PlanVersionConflictError(
            expected=expected_plan_version,
            current=plan.plan_version,
        )

    validate_segments(proposed_segments)
    proposed_copy = [
        segment.model_copy(deep=True) for segment in proposed_segments
    ]
    if proposed_copy == plan.draft_segments:
        return plan.model_copy(deep=True)

    return plan.model_copy(
        update={
            "plan_version": plan.plan_version + 1,
            "draft_segments": proposed_copy,
        },
        deep=True,
    )
